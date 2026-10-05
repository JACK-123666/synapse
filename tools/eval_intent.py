"""Synapse 三路融合意图识别 · 离线评测脚本。

评测对象：app/intent/ 下的真实代码（keyword / semantic / vector / blend 四模块）。
不启动 docker / FastAPI，直接驱动真实识别器，输出可信的准确率数字。

覆盖：
    1. 单路 Top-1 准确率：LLM 语义 / 向量相似度 / 关键词投票
    2. 三路融合 Top-1 准确率（走 IntentFusion._compute_weights + _fuse 真实逻辑）
    3. 模糊难例子集准确率（测试集 hard=True 的样本）
    4. 降级鲁棒性：模拟 LLM / 向量 / 双路故障时融合的准确率变化（权重自动重分配）
    5. 混淆矩阵、逐意图准确率、单条延迟（均值 / P95）

用法：
    python tools/eval_intent.py                     # 全量跑（含 LLM 路，需 .env 有 API key）
    python tools/eval_intent.py --no-llm            # 跳过 LLM 路（只测向量+关键词+融合的可用子集）
    python tools/eval_intent.py --sample 20         # 只取前 20 条快速验证
    python tools/eval_intent.py --json out.json     # 结果同时落盘 JSON

说明：
    - LLM 路调用真实 API（默认 DeepSeek），每条样本 1 次调用；融合的降级场景用缓存结果重算，
      不额外产生 LLM 调用。
    - 向量路默认把 ChromaDB 打补丁为内存版（EphemeralClient），使用 ChromaDB 内置本地 embedding
      （all-MiniLM-L6-v2，首次使用会从 HuggingFace 下载模型；已默认指向 hf-mirror 镜像）。
      若模型下载失败，向量路自动标记为不可用，融合器会重分配权重，不影响其余评测。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 国内网络默认走 HF 镜像，提升向量路内置 embedding 模型的下载成功率
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from app import store as app_store  # noqa: E402
from app.config import Settings, get_settings  # noqa: E402
from app.intent.blend import IntentFusion  # noqa: E402
from app.intent.keyword import KeywordIntentRecognizer  # noqa: E402
from app.intent.semantic import LLMIntentRecognizer  # noqa: E402
from app.intent.vector import VectorIntentRecognizer  # noqa: E402
from app.llm.gateway import get_llm_client  # noqa: E402

from intent_testset import TEST_SET  # noqa: E402

INTENTS = ["knowledge_retrieval", "summarize", "small_talk"]


def patch_inmemory_chroma() -> None:
    """把 app.store 的 ChromaDB 客户端替换为内存版，避免依赖 docker 服务。"""
    import chromadb

    app_store._chroma = chromadb.EphemeralClient()
    print("[patch] ChromaDB 已切换为内存版 (EphemeralClient)，向量路无需 docker")


class NullRecognizer:
    """模拟一路识别器故障：恒返回 None。"""

    async def recognize(self, message: str) -> None:
        return None


def top_intent(result: Optional[Dict[str, float]]) -> Optional[str]:
    """从 {意图: 分数} 字典取 Top-1 意图；None 表示该路不可用/失败。"""
    if not result:
        return None
    return max(result, key=result.get)


def _norm(s: str) -> str:
    """归一化：去空白/标点/大小写，便于做子串与 n-gram 比对。"""
    return "".join(ch for ch in s.lower() if not ch.isspace())


def build_leak_phrases(settings: Settings) -> set:
    """收集系统「见过」的示例文本，用于识别测试集泄漏。

    来源：
    1. config.py 中 intent_examples —— 向量意图索引的种子样本
    2. semantic.py 中 LLM few-shot 提示词的示例 —— 直接写进了 prompt
    （keywords 是通用词表，不算泄漏，不参与剔除）
    """
    phrases = set()
    for examples in settings.intent_examples.values():
        for ex in examples:
            phrases.add(_norm(ex))
    # 与 app/intent/semantic.py 中 few-shot 示例保持一致
    few_shot_examples = ["你好", "什么是向量数据库？", "帮我总结一下"]
    for ex in few_shot_examples:
        phrases.add(_norm(ex))
    return phrases


def is_leaked(text: str, phrases: set, ngram: int = 4) -> bool:
    """判断测试文本是否与内置示例泄漏重叠。

    判定规则（取并集，宁可多判为泄漏以保严谨）：
    1. 互为子串（完全被内置示例覆盖，或覆盖内置示例）
    2. 与任一内置示例共享 >= ngram 长度的连续子串（近似改写也算泄漏）
    """
    nt = _norm(text)
    for p in phrases:
        if p in nt or nt in p:
            return True
    if ngram and len(nt) >= ngram:
        for i in range(len(nt) - ngram + 1):
            gram = nt[i : i + ngram]
            if any(gram in p for p in phrases if len(p) >= ngram):
                return True
    return False


async def smoke_test_fusion(fusion: IntentFusion, settings: Settings) -> None:
    """端到端冒烟：走完整的 fusion.recognize()（asyncio.gather 三路），
    与前 3 条样本的融合结果比对，确认与真实代码路径一致。"""
    from intent_testset import TEST_SET as TS

    for item in TS[:3]:
        intent, conf = await fusion.recognize(item["text"])
        print(f"[smoke] '{item['text'][:20]}' -> {intent} (conf={conf:.2f}) gold={item['intent']}")


async def run_eval(args: argparse.Namespace) -> Dict[str, Any]:
    settings = get_settings()
    patch_inmemory_chroma()

    kw = KeywordIntentRecognizer(settings)
    vec = VectorIntentRecognizer(settings)
    sem = LLMIntentRecognizer(settings)
    fusion = IntentFusion(settings)
    await fusion.initialize()
    await vec.initialize()

    await smoke_test_fusion(fusion, settings)  # 端到端冒烟：验证真实 fusion.recognize() 路径

    items = TEST_SET[: args.sample] if args.sample else TEST_SET
    sema = asyncio.Semaphore(args.concurrency)
    leak_phrases = build_leak_phrases(settings)

    # 路径可用性标记（供报告使用）
    path_available = {"llm": True, "vector": True, "keyword": True}

    async def collect(item: Dict[str, str]) -> Dict[str, Any]:
        text, gold = item["text"], item["intent"]
        async with sema:
            kw_r = await kw.recognize(text)
            vec_r = await vec.recognize(text)
            llm_r = await sem.recognize(text) if not args.no_llm else None
        t0 = time.perf_counter()
        fused = fusion._fuse(  # 真实融合逻辑（纯函数）
            llm_result=llm_r,
            vector_result=vec_r,
            keyword_result=kw_r,
            weights=fusion._compute_weights(
                llm_r is not None, vec_r is not None, kw_r is not None
            ),
        )
        fused_top = top_intent(fused) or "small_talk"
        dt = time.perf_counter() - t0
        return {
            "text": text,
            "gold": gold,
            "hard": item.get("hard", False),
            "leaked": is_leaked(text, leak_phrases),
            "llm_r": llm_r,
            "vector_r": vec_r,
            "keyword_r": kw_r,
            "pred": fused_top,
            "latency": dt,
        }

    rows = await asyncio.gather(*(collect(it) for it in items))
    if not rows:
        raise SystemExit("测试集为空")

    # 可用性统计
    for name, key in [("llm", "llm_r"), ("vector", "vector_r"), ("keyword", "keyword_r")]:
        n_ok = sum(1 for r in rows if r[key] is not None)
        path_available[name] = n_ok == len(rows)
        print(
            f"[avail] {name:8s} 可用样本 {n_ok}/{len(rows)}"
            + ("" if n_ok == len(rows) else "  <- 该路部分/全部不可用")
        )

    # ---- 各路径 Top-1 准确率 ----
    def acc_for(path_key: str) -> Dict[str, Any]:
        ok = [r for r in rows if r[path_key] is not None]
        if not ok:
            return {"available": False, "n": 0, "acc": None, "correct": 0}
        correct = sum(1 for r in ok if top_intent(r[path_key]) == r["gold"])
        hard_ok = [r for r in ok if r["hard"]]
        hard_correct = sum(1 for r in hard_ok if top_intent(r[path_key]) == r["gold"])
        return {
            "available": True,
            "n": len(ok),
            "acc": round(correct / len(ok), 4),
            "correct": correct,
            "hard_n": len(hard_ok),
            "hard_acc": round(hard_correct / len(hard_ok), 4) if hard_ok else None,
            "hard_correct": hard_correct,
        }

    per_path = {name: acc_for(f"{name}_r") for name in ("llm", "vector", "keyword")}

    # ---- 融合准确率（基准 + 各故障场景，全部用缓存结果重算，不额外调 LLM）----
    def fused_for(r: Dict[str, Any], llm_ok: bool, vec_ok: bool, kw_ok: bool) -> str:
        llm_r = r["llm_r"] if llm_ok else None
        vec_r = r["vector_r"] if vec_ok else None
        kw_r = r["keyword_r"] if kw_ok else None
        scores = fusion._fuse(
            llm_r, vec_r, kw_r,
            fusion._compute_weights(llm_r is not None, vec_r is not None, kw_r is not None),
        )
        return top_intent(scores) or "small_talk"

    scenarios = {
        "三路融合(基准)": (True, True, True),
        "仅语义+关键词(向量故障)": (True, False, True),
        "仅语义+向量(关键词故障)": (True, True, False),
        "仅向量+关键词(LLM故障)": (False, True, True),
        "仅关键词(双路故障)": (False, False, True),
    }

    def scenario_acc(flags) -> Dict[str, Any]:
        correct = 0
        for r in rows:
            if fused_for(r, *flags) == r["gold"]:
                correct += 1
        return {"acc": round(correct / len(rows), 4), "correct": correct, "n": len(rows)}

    fusion_results = {name: scenario_acc(flags) for name, flags in scenarios.items()}

    # ---- 混淆矩阵（基准融合） ----
    cm = {g: {p: 0 for p in INTENTS} for g in INTENTS}
    for r in rows:
        cm[r["gold"]][fused_for(r, True, True, True)] += 1

    # ---- 延迟统计（融合纯函数耗时，不含 LLM 网络往返） ----
    lats = [r["latency"] for r in rows]
    lat = {
        "avg_ms": round(statistics.mean(lats) * 1000, 3),
        "p95_ms": round(sorted(lats)[max(0, int(len(lats) * 0.95) - 1)] * 1000, 3),
    }

    # ---- 逐意图准确率（基准融合） ----
    per_intent = {}
    for intent in INTENTS:
        subset = [r for r in rows if r["gold"] == intent]
        if not subset:
            continue
        correct = sum(1 for r in subset if fused_for(r, True, True, True) == intent)
        per_intent[intent] = {"n": len(subset), "acc": round(correct / len(subset), 4)}

    # ---- 难例子集（基准融合） ----
    hard_rows = [r for r in rows if r["hard"]]
    hard_correct = sum(1 for r in hard_rows if fused_for(r, True, True, True) == r["gold"])

    # ---- 纯净子集：剔除与内置示例重叠的样本（方法学上更可信的口径） ----
    clean_rows = [r for r in rows if not r["leaked"]]
    clean_correct = sum(1 for r in clean_rows if fused_for(r, True, True, True) == r["gold"])

    return {
        "meta": {
            "settings": {
                "provider": settings.llm_provider,
                "weights": {
                    "llm": settings.intent_llm_weight,
                    "vector": settings.intent_vector_weight,
                    "keyword": settings.intent_keyword_weight,
                },
                "llm_path_used": not args.no_llm,
            },
            "testset": {"n": len(rows), "hard_n": len(hard_rows)},
            "path_available": path_available,
        },
        "per_path": per_path,
        "fusion": fusion_results,
        "confusion_matrix": cm,
        "per_intent": per_intent,
        "hard_subset": {
            "n": len(hard_rows),
            "correct": hard_correct,
            "acc": round(hard_correct / len(hard_rows), 4) if hard_rows else None,
        },
        "clean_subset": {
            "n": len(clean_rows),
            "leaked_n": len(rows) - len(clean_rows),
            "correct": clean_correct,
            "acc": round(clean_correct / len(clean_rows), 4) if clean_rows else None,
        },
        "latency_pure_fusion_ms": lat,
        "per_item": [
            {
                "text": r["text"],
                "gold": r["gold"],
                "hard": r["hard"],
                "leaked": r["leaked"],
                "llm": top_intent(r["llm_r"]),
                "vector": top_intent(r["vector_r"]),
                "keyword": top_intent(r["keyword_r"]),
                "fusion": r["pred"],
            }
            for r in rows
        ],
    }


def print_report(result: Dict[str, Any]) -> None:
    print("\n" + "=" * 62)
    print("Synapse 三路融合意图识别 · 评测报告")
    print("=" * 62)

    meta = result["meta"]
    w = meta["settings"]["weights"]
    print(f"测试集: {meta['testset']['n']} 条（难例 {meta['testset']['hard_n']} 条） | "
          f"LLM Provider: {meta['settings']['provider']}")
    print(f"权重: LLM={w['llm']} 向量={w['vector']} 关键词={w['keyword']}")
    print()

    print("一、单路 Top-1 准确率")
    print(f"  {'路径':<10}{'可用':>6}{'样本':>6}{'正确':>6}{'准确率':>10}{'难例准确率':>12}")
    for name, label in [("llm", "LLM 语义"), ("vector", "向量相似度"), ("keyword", "关键词投票")]:
        p = result["per_path"][name]
        if not p["available"]:
            print(f"  {label:<10}{'否':>6}{'-':>6}{'-':>6}{'-':>10}{'-':>12}  (该路不可用)")
            continue
        hard = f"{p['hard_acc']:.1%}" if p["hard_acc"] is not None else "-"
        print(f"  {label:<10}{'是':>6}{p['n']:>6}{p['correct']:>6}{p['acc']:>10.1%}{hard:>12}")
    print()

    print("二、三路融合 Top-1 准确率（真实 _compute_weights + _fuse 逻辑）")
    for name, f in result["fusion"].items():
        print(f"  {name:<22}  {f['acc']:>7.1%}   ({f['correct']}/{f['n']})")
    print()

    hs = result["hard_subset"]
    print(f"三、模糊难例子集（{hs['n']} 条，关键词路难以命中的样本）")
    if hs["acc"] is not None:
        print(f"  融合准确率: {hs['acc']:.1%}  ({hs['correct']}/{hs['n']})")
    print()

    cs = result["clean_subset"]
    print(f"四、纯净子集（剔除 {cs['leaked_n']} 条与内置示例重叠的样本，剩 {cs['n']} 条）")
    if cs["acc"] is not None:
        print(f"  融合准确率: {cs['acc']:.1%}  ({cs['correct']}/{cs['n']})  <- 推荐作为对外口径")
    print()

    print("五、逐意图准确率（基准融合）")
    for intent, v in result["per_intent"].items():
        print(f"  {intent:<24}{v['n']:>4} 条  准确率 {v['acc']:>7.1%}")
    print()

    print("六、混淆矩阵（行=金标准, 列=融合预测）")
    intents = ["knowledge_retrieval", "summarize", "small_talk"]
    # f-string 表达式内不能含反斜杠（Python 3.12 以下），表头先单独取出
    header = "金\\预"
    print(f"  {header:<22}" + "".join(f"{i[:14]:>16}" for i in intents))
    for g in intents:
        row = result["confusion_matrix"][g]
        print(f"  {g:<22}" + "".join(f"{row[p]:>16}" for p in intents))
    print()

    lat = result["latency_pure_fusion_ms"]
    print(f"七、延迟（融合纯逻辑，不含 LLM 网络往返）  avg {lat['avg_ms']}ms / p95 {lat['p95_ms']}ms")
    print("=" * 62)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synapse 意图识别评测")
    parser.add_argument("--sample", type=int, default=0, help="只取前 N 条样本（默认全部）")
    parser.add_argument("--no-llm", action="store_true", help="跳过 LLM 语义路（不调用 API）")
    parser.add_argument("--concurrency", type=int, default=5, help="LLM 并发数（默认 5）")
    parser.add_argument("--json", type=str, default="", help="结果输出到 JSON 文件")
    args = parser.parse_args()

    if args.sample:
        print(f"[info] --sample={args.sample}：只评测前 {args.sample} 条")

    result = asyncio.run(run_eval(args))
    print_report(result)

    if args.json:
        out = Path(args.json)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[out] 结果已写入 {out}")


if __name__ == "__main__":
    main()
