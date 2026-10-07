"""三路融合意图识别器。

融合策略：将 LLM 语义、向量相似度、关键词投票三路输出归一化后，
按配置权重加权求和，得分最高者作为最终意图。

权重默认：LLM 0.5、向量 0.3、关键词 0.2（可通过环境变量调整）。
任一路识别失败时，其权重自动重分配给其他路。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional, Tuple

from app.config import Settings, get_settings
from app.intent.keyword import KeywordIntentRecognizer
from app.intent.semantic import LLMIntentRecognizer
from app.intent.vector import VectorIntentRecognizer
from app.observability import metrics

logger = logging.getLogger(__name__)


class IntentFusion:
    """三路融合意图识别器。

    工作流程：
    1. 并行（或串行退避）执行三路识别。
    2. 归一化各路的输出分数。
    3. 按权重加权求和。
    4. 返回最高分意图及其置信度。
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings: Settings = settings or get_settings()
        self._llm = LLMIntentRecognizer(settings)
        self._vector = VectorIntentRecognizer(settings)
        self._keyword = KeywordIntentRecognizer(settings)

    async def initialize(self) -> None:
        """初始化向量意图索引（需在 startup 时调用）。"""
        await self._vector.initialize()

    async def refresh(self) -> None:
        """意图目录变化后重建向量索引（插件启停 / 热重载时调用）。"""
        await self._vector.refresh()

    async def recognize(self, message: str) -> Tuple[str, float]:
        """三路融合识别用户消息意图。

        Args:
            message: 用户输入消息

        Returns:
            (意图标签, 融合置信度) 元组
            - 若所有路均失败，返回 ("small_talk", 1.0) 作为默认兜底

        执行顺序按成本从低到高：
        1. 关键词 + 向量两路（本地、便宜）并行执行；
        2. 两路结论一致且都高置信时直接短路，省掉一次 LLM 往返；
        3. 否则再跑 LLM 路，带独立超时（超时视为该路失败，权重转给前两路）；
        4. 三路加权融合取最高分。
        """
        # ---- 第一级：关键词 + 向量（本地，无 LLM 成本）----
        vec_raw, kw_raw = await asyncio.gather(
            self._vector.recognize_detailed(message),
            self._keyword.recognize(message),
            return_exceptions=True,
        )
        if isinstance(vec_raw, BaseException):
            vector_result = self._unwrap(vec_raw, "向量")
            vec_similarity = 0.0
        else:
            vector_result, vec_similarity = vec_raw
        keyword_result = self._unwrap(kw_raw, "关键词")

        # ---- 短路：两路一致且高置信，跳过 LLM ----
        if self._settings.intent_short_circuit:
            short = self._try_short_circuit(keyword_result, vector_result, vec_similarity)
            if short is not None:
                intent, confidence = short
                metrics.record_intent_confidence(intent, confidence)
                logger.info(
                    "融合意图识别(短路): '%s' -> %s (置信度=%.2f, 关键词+向量一致，跳过 LLM)",
                    message[:50], intent, confidence,
                )
                return (intent, confidence)

        # ---- 第二级：LLM 语义路（带独立超时）----
        llm_result = await self._recognize_llm(message)

        # 动态权重分配：失败的路将其权重重新分配给健康路
        weights = self._compute_weights(
            bool(llm_result), bool(vector_result), bool(keyword_result)
        )

        # 加权融合
        fused_scores = self._fuse(
            llm_result=llm_result,
            vector_result=vector_result,
            keyword_result=keyword_result,
            weights=weights,
        )

        # 提取最高分意图
        if not fused_scores:
            logger.warning("融合器: 所有识别器均失败，使用默认意图 small_talk")
            metrics.record_intent_confidence("small_talk", 1.0)
            return ("small_talk", 1.0)

        best_intent, confidence = max(fused_scores.items(), key=lambda x: x[1])
        metrics.record_intent_confidence(best_intent, confidence)
        logger.info(
            "融合意图识别: '%s' -> %s (置信度=%.2f, 三路权重: LLM=%.2f VEC=%.2f KW=%.2f, "
            "向量原始相似度=%.3f)",
            message[:50], best_intent, confidence,
            weights.get("llm", 0), weights.get("vector", 0), weights.get("keyword", 0),
            vec_similarity,
        )
        return (best_intent, confidence)

    @staticmethod
    def _unwrap(result: Any, name: str) -> Optional[Dict[str, float]]:
        """把 gather 的结果规整为分数字典或 None（异常一律记为该路失败）。"""
        if isinstance(result, BaseException):
            logger.warning("融合器: %s 意图识别异常: %s", name, result)
            return None
        return result

    async def _recognize_llm(self, message: str) -> Optional[Dict[str, float]]:
        """执行 LLM 语义路；超时或异常一律视为该路失败。

        这个超时是必需的：LLM 路权重最高（默认 0.5）且位于请求关键路径上，
        一旦挂起，整个请求会一直等到 llm_timeout（默认 60s），
        远超 agent_timeout（默认 30s），降级机制根本来不及介入。
        """
        timeout = self._settings.intent_llm_timeout
        try:
            return await asyncio.wait_for(
                self._llm.recognize(message), timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(
                "融合器: LLM 意图识别超时 (%.1fs)，该路权重转给向量/关键词", timeout
            )
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("融合器: LLM 意图识别异常: %s", exc)
            return None

    def _try_short_circuit(
        self,
        keyword_result: Optional[Dict[str, float]],
        vector_result: Optional[Dict[str, float]],
        vec_top_similarity: float,
    ) -> Optional[Tuple[str, float]]:
        """关键词与向量两路结论一致、且各自证据都够强时，直接判定，跳过 LLM 路。

        两道判据：
        - 关键词：命中要集中在同一个意图上（归一化分数 >= min_score）；
        - 向量：最近一条意图示例的**原始余弦相似度** >= min_similarity。
          这里刻意不用归一化分布：Top-K 横跨多个意图时它会被摊薄，
          即使语义上极其接近也拿不到高分（实测几乎永远卡在 0.4~0.7）。

        一旦消息有歧义（例如同时出现「总结」和「文档」），关键词命中被摊薄到
        多个意图上，第一道判据自然不通过，仍会走完整三路融合。
        """
        if not keyword_result or not vector_result:
            return None

        kw_top = max(keyword_result, key=keyword_result.get)
        vec_top = max(vector_result, key=vector_result.get)
        if kw_top != vec_top:
            return None

        if keyword_result[kw_top] < self._settings.intent_short_circuit_min_score:
            return None
        if vec_top_similarity < self._settings.intent_short_circuit_min_similarity:
            return None

        # 用「LLM 路缺席」的权重重分配结果折算，口径与完整路径一致
        weights = self._compute_weights(False, True, True)
        fused = (
            keyword_result[kw_top] * weights["keyword"]
            + vec_top_similarity * weights["vector"]
        )
        return kw_top, fused

    def _compute_weights(
        self,
        llm_ok: bool,
        vector_ok: bool,
        keyword_ok: bool,
    ) -> Dict[str, float]:
        """计算动态权重。

        当某路识别器失败时，将其权重按比例重新分配给其他路。
        若全部失败，返回均匀权重（虽然最终会走默认兜底）。
        """
        base = {
            "llm": self._settings.intent_llm_weight,
            "vector": self._settings.intent_vector_weight,
            "keyword": self._settings.intent_keyword_weight,
        }

        # 统计可用路数及其权重
        available = {
            k: v
            for k, v, ok in [
                ("llm", base["llm"], llm_ok),
                ("vector", base["vector"], vector_ok),
                ("keyword", base["keyword"], keyword_ok),
            ]
            if ok
        }

        if not available:
            return {"llm": 0.0, "vector": 0.0, "keyword": 0.0}

        total_weight = sum(available.values())
        # 归一化，确保和为 1.0
        redistributed: Dict[str, float] = {}
        for key in base:
            if key in available:
                redistributed[key] = available[key] / total_weight
            else:
                redistributed[key] = 0.0

        # 修正浮点累积误差
        actual_sum = sum(redistributed.values())
        if actual_sum > 0 and abs(actual_sum - 1.0) > 0.001:
            redistributed = {
                k: v / actual_sum for k, v in redistributed.items()
            }

        return redistributed

    def _fuse(
        self,
        llm_result: Optional[Dict[str, float]],
        vector_result: Optional[Dict[str, float]],
        keyword_result: Optional[Dict[str, float]],
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """加权融合三路识别结果。

        对每条路的输出乘以权重后累加，得到融合后的意图分数分布。

        Args:
            llm_result: LLM 识别结果
            vector_result: 向量识别结果
            keyword_result: 关键词识别结果
            weights: 动态权重 {"llm": w1, "vector": w2, "keyword": w3}

        Returns:
            融合后的 {意图: 加权分数}
        """
        fused: Dict[str, float] = {}

        # 按权重累加各识别器的分数
        contributions = [
            (llm_result, weights.get("llm", 0)),
            (vector_result, weights.get("vector", 0)),
            (keyword_result, weights.get("keyword", 0)),
        ]

        for result, weight in contributions:
            if result is None:
                continue
            for intent, score in result.items():
                fused[intent] = fused.get(intent, 0.0) + score * weight

        # 归一化
        total = sum(fused.values())
        if total > 0:
            fused = {k: v / total for k, v in fused.items()}

        return fused


# 全局单例

_instance: Optional[IntentFusion] = None


def get_intent_fusion() -> IntentFusion:
    """获取融合意图识别器单例。"""
    global _instance
    if _instance is None:
        _instance = IntentFusion()
    return _instance
