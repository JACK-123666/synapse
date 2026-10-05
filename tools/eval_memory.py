"""Synapse 三级分层记忆 · Token 消耗评测。

对比同一段 N 轮对话在两种场景下的「累计历史上下文 Token」：

    A. 基线（无记忆管理）：每轮把全部历史消息原样塞进 prompt，随轮数线性增长
    B. Synapse 三级记忆：Redis 短期窗口 + 超阈值异步压缩为摘要存 ChromaDB + 按需召回

驱动真实代码模块：
    - app.memory.recent.ShortTermMemory    （真实滑动窗口 + LTRIM 裁剪）
    - app.memory.compress.MemoryCompressor （真实压缩判定 should_compress）
    - app.memory.archive.LongTermMemory    （真实召回 recall + 摘要存取）
    - app.memory.recent._estimate_tokens   （系统内置中英文 Token 估算器，与压缩判定同一把尺子）

为保证可复现 / 低成本，做了 3 处明确标注的 mock（都不影响 Token 计数字数）：
    1. 短期记忆底层：内存版 Redis（实现 rpush/lrange/ltrim/expire/delete/get/set）
    2. 摘要生成：固定长度 stub（系统设计约束 ≤200 字；实际 LLM 摘要通常 150-250 字）
    3. 长期记忆 embedding：确定性 mock 向量（仅影响召回相似度排序，不影响文本长度）

用法：
    python tools/eval_memory.py [--rounds 20] [--json out.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import store as app_store  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.llm.gateway import get_llm_client  # noqa: E402
from app.memory.archive import LongTermMemory, get_long_term_memory  # noqa: E402
from app.memory.compress import MemoryCompressor  # noqa: E402
from app.memory.recent import ShortTermMemory, _estimate_tokens  # noqa: E402

# ---------------- mock：内存版 Redis ----------------

class FakeRedis:
    """实现 recent/profile 用到的 Redis 命令子集（列表 + 字符串）。"""

    def __init__(self) -> None:
        self.data: Dict[str, Any] = {}

    async def rpush(self, key: str, *values: str) -> int:
        lst = self.data.setdefault(key, [])
        lst.extend(values)
        return len(lst)

    async def lrange(self, key: str, start: int, end: int) -> List[str]:
        lst = self.data.get(key, [])
        return lst[start:] if end == -1 else lst[start : end + 1]

    async def ltrim(self, key: str, start: int, end: int) -> bool:
        lst = self.data.get(key, [])
        self.data[key] = lst[start:] if end == -1 else lst[start : end + 1]
        return True

    async def expire(self, key: str, ttl: int) -> bool:
        return True

    async def delete(self, key: str) -> int:
        self.data.pop(key, None)
        return 1

    async def get(self, key: str) -> Any:
        return self.data.get(key)

    async def set(self, key: str, value: str) -> bool:
        self.data[key] = value
        return True

    async def ping(self) -> bool:
        return True


# ---------------- mock：确定性 embedding 向量 ----------------

DIM = 384


def mock_vector(text: str) -> List[float]:
    """由文本哈希生成确定性向量（仅用于相似度排序，不影响 token 计数）。"""
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    v = [int(h[i : i + 2], 16) / 255.0 for i in range(0, min(len(h), DIM * 2), 2)]
    while len(v) < DIM:
        v.append(0.0)
    return v[:DIM]


async def _mock_embed(text: str) -> List[float]:
    return mock_vector(text)


async def _mock_embed_batch(texts: List[str]) -> List[List[float]]:
    return [mock_vector(t) for t in texts]


# ---------------- mock：固定长度摘要 stub ----------------

async def fake_summary(dialogue_text: str) -> str:
    """生成确定性、接近系统约束（≤200 字）的摘要文本。"""
    n_msg = dialogue_text.count("用户:") + dialogue_text.count("助手:")
    return (
        f"（对话摘要）本轮共 {n_msg} 条消息。用户围绕向量数据库、RAG 检索、"
        "多 Agent 架构与记忆管理等主题连续提问，助手基于检索增强分别给出了"
        "概念解释与方案对比，双方确认了技术要点，并记录了后续需要跟进的事项。"
        "对话结论以原文为准。"
    )


# ---------------- 模拟对话 ----------------

USER_MESSAGES = [
    "什么是向量数据库？",
    "它和传统数据库有什么区别？",
    "那 RAG 是什么，怎么工作的？",
    "帮我解释一下 Embedding 的概念",
    "为什么需要重新排序 Rerank？",
    "多 Agent 架构是怎么做任务分发的？",
    "意图识别一般有哪几种方法？",
    "三级记忆是怎么设计的？",
    "短期记忆和长期记忆怎么配合？",
    "Token 消耗太高怎么优化？",
    "异步压缩是什么时候触发的？",
    "Z-score 异常检测的原理是什么？",
    "健康分低于多少会被摘除？",
    "ChromaDB 和 Milvus 该怎么选？",
    "Docker 一键部署怎么配置的？",
    "Prometheus 采集了哪些指标？",
    "Grafana 看板怎么搭建的？",
    "可观测性这块踩过什么坑？",
    "后续打算怎么优化这个项目？",
    "谢谢，今天学到很多！",
]

_TPL = [
    "关于这个问题，核心要点是 {kw}。首先需要理解其基本工作原理，"
    "然后再看具体的实现细节。从架构上看，{kw} 涉及到多个环节的配合，"
    "每个环节都需要做针对性的优化，才能保证整体的效果和稳定性。"
    "如果遇到具体问题，建议先从官方文档入手。",
    "{kw} 可以从两个层面来理解：一是原理层面，二是工程层面。"
    "原理上决定了它的能力边界，工程上决定了它能不能落地。"
    "结合我们的实践来看，关键是做好权衡，避免过度设计。",
    "简单说，{kw} 的核心目标是控制成本、提升效果。实现时要注意几个点："
    "数据质量、参数选择、以及失败时的降级处理。这三者缺一不可。",
]


def build_assistant_reply(rng: random.Random, turn: int) -> str:
    kw = ["记忆压缩", "意图融合", "路由降级", "向量检索", "异常检测", "token 预算", "摘要召回", "画像注入"]
    tpl = rng.choice(_TPL)
    return tpl.format(kw=rng.choice(kw))


def build_conversation(n_rounds: int):
    """生成 (user, assistant) 轮次列表。"""
    rng = random.Random(42)
    conv = []
    for i in range(n_rounds):
        user = USER_MESSAGES[i % len(USER_MESSAGES)]
        assistant = build_assistant_reply(rng, i)
        conv.append((user, assistant))
    return conv


# ---------------- 主流程 ----------------

def patch_mocks() -> None:
    """打补丁：内存 Redis + 确定性 embedding。"""
    app_store._redis = FakeRedis()  # app.store.get_redis 单例返回它
    llm = get_llm_client()
    llm.embed = _mock_embed  # type: ignore[method-assign]
    llm.embed_batch = _mock_embed_batch  # type: ignore[method-assign]


def patch_chroma_ephemeral() -> None:
    """内存版 ChromaDB，避免依赖 docker。"""
    import chromadb

    app_store._chroma = chromadb.EphemeralClient()


def count_tokens_of_messages(messages: List[Dict[str, Any]]) -> int:
    """对消息列表的 content 逐条估算 Token。"""
    return sum(_estimate_tokens(m.get("content", "")) for m in messages)


def count_tokens_of_recall(recall: List[Dict[str, Any]]) -> int:
    """对召回摘要（含格式化前缀）估算 Token，与 knowledge.py _format_recall 一致。"""
    parts = []
    for i, item in enumerate(recall, 1):
        text = item.get("text", "")
        score = item.get("score", 0)
        if text:
            parts.append(f"[{i}] (相似度: {score:.2f}) {text}")
    return _estimate_tokens("\n".join(parts))


async def run(n_rounds: int) -> Dict[str, Any]:
    settings = get_settings()
    patch_chroma_ephemeral()
    patch_mocks()

    short = ShortTermMemory(settings)
    long_mem: LongTermMemory = get_long_term_memory()
    compressor = MemoryCompressor(settings)
    # 摘要生成用 stub（真实 _generate_summary 会调 LLM，这里固定长度保证可复现）
    compressor._generate_summary = fake_summary  # type: ignore[method-assign]

    conv = build_conversation(n_rounds)
    sid = "eval-mem-001"

    per_turn: List[Dict[str, Any]] = []
    all_round_texts: List[str] = []  # 基线用：累计全部历史

    for t, (user, assistant) in enumerate(conv, 1):
        # ---- 当前时刻的上下文构成（含此前所有轮次） ----
        short_msgs = await short.get_messages(sid)
        recall = await long_mem.recall(query_text=user)

        synapse_tokens = count_tokens_of_messages(short_msgs) + count_tokens_of_recall(recall)

        # 基线：把此前全部历史原样塞入
        history_msgs = []
        for u, a in all_round_texts:
            history_msgs.append({"role": "user", "content": u})
            history_msgs.append({"role": "assistant", "content": a})
        baseline_tokens = count_tokens_of_messages(history_msgs)

        # 追加本轮（模拟 chat.py：先取记忆 → dispatch → 再 append）
        all_round_texts.append((user, assistant))
        await short.append(sid, "user", user)
        await short.append(sid, "assistant", assistant)

        # 压缩判定（模拟 chat.py 的异步压缩）
        compressed = False
        if await compressor.should_compress(sid):
            await compressor.compress(sid)
            compressed = True

        per_turn.append({
            "turn": t,
            "rounds_stored": len(all_round_texts),
            "baseline_tokens": baseline_tokens,
            "synapse_tokens": synapse_tokens,
            "short_msgs": len(short_msgs),
            "recall_msgs": len(recall),
            "compressed": compressed,
        })

    total_baseline = sum(x["baseline_tokens"] for x in per_turn)
    total_synapse = sum(x["synapse_tokens"] for x in per_turn)
    reduction = 1.0 - total_synapse / total_baseline if total_baseline else 0.0
    peak_baseline = max(x["baseline_tokens"] for x in per_turn)
    peak_synapse = max(x["synapse_tokens"] for x in per_turn)

    return {
        "meta": {
            "rounds": n_rounds,
            "config": {
                "short_term_max_rounds": settings.short_term_max_rounds,
                "summary_trigger_rounds": settings.summary_trigger_rounds,
                "token_budget": settings.token_budget,
                "long_term_recall_k": settings.long_term_recall_k,
            },
            "token_estimator": "系统内置 _estimate_tokens（中文≈1.5字符/token，英文≈0.75 token/词）",
            "assumptions": [
                "短期记忆用内存版 Redis 模拟（真实 LTRIM 窗口逻辑）",
                "摘要用固定长度 stub（≤200 字约束，实际 LLM 摘要 150-250 字）",
                "长期记忆 embedding 用确定性 mock 向量（不影响文本长度）",
                "会话未启用用户画像（user_id=None，profile 上下文为 0）",
            ],
        },
        "totals": {
            "baseline": total_baseline,
            "synapse": total_synapse,
            "reduction": round(reduction, 4),
            "peak_baseline": peak_baseline,
            "peak_synapse": peak_synapse,
            "peak_reduction": round(1.0 - peak_synapse / peak_baseline, 4),
        },
        "per_turn": per_turn,
    }


def print_report(result: Dict[str, Any]) -> None:
    print("\n" + "=" * 66)
    print("Synapse 三级记忆 Token 消耗 · 评测报告")
    print("=" * 66)
    m = result["meta"]
    cfg = m["config"]
    print(f"对话轮数: {m['rounds']} | 短期窗口: {cfg['short_term_max_rounds']} 轮 | "
          f"压缩阈值: {cfg['summary_trigger_rounds']} 轮 | 召回 Top-{cfg['long_term_recall_k']}")
    print()

    tots = result["totals"]
    print(f"累计历史上下文 Token（{m['rounds']} 轮之和）:")
    print(f"  基线（无记忆管理，每轮全量历史）  {tots['baseline']:>10,}")
    print(f"  Synapse 三级记忆                 {tots['synapse']:>10,}")
    print(f"  降低幅度                          {tots['reduction']:>10.1%}")
    print()
    print(f"单轮峰值上下文 Token:")
    print(f"  基线峰值                          {tots['peak_baseline']:>10,}")
    print(f"  Synapse 峰值                      {tots['peak_synapse']:>10,}")
    print(f"  峰值降幅                          {tots['peak_reduction']:>10.1%}")
    print()

    print("逐轮明细（节选：前 10 轮 + 最后 5 轮）:")
    print(f"  {'轮次':>4} {'基线Token':>10} {'Synapse':>9} {'短期消息数':>9} {'召回数':>6} {'压缩?':>6}")
    rows = result["per_turn"]
    show = rows[:10] + rows[-5:]
    for r in show:
        mark = "✓" if r["compressed"] else ""
        print(f"  {r['turn']:>4} {r['baseline_tokens']:>10,} {r['synapse_tokens']:>9} "
              f"{r['short_msgs']:>9} {r['recall_msgs']:>6} {mark:>6}")
    print("=" * 66)
    print("\n口径说明：")
    for a in m["assumptions"]:
        print(f"  · {a}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synapse 三级记忆 Token 消耗评测")
    parser.add_argument("--rounds", type=int, default=20, help="模拟对话轮数（默认 20）")
    parser.add_argument("--json", type=str, default="", help="结果输出到 JSON 文件")
    args = parser.parse_args()

    result = asyncio_run(args.rounds)
    print_report(result)

    if args.json:
        out = Path(args.json)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[out] 结果已写入 {out}")


# asyncio.run 与模块级 import 兼容性
import asyncio  # noqa: E402


def asyncio_run(n_rounds: int):
    return asyncio.run(run(n_rounds))


if __name__ == "__main__":
    main()
