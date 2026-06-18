"""三路融合意图识别器。

融合策略：将 LLM 语义、向量相似度、关键词投票三路输出归一化后，
按配置权重加权求和，得分最高者作为最终意图。

权重默认：LLM 0.5、向量 0.3、关键词 0.2（可通过环境变量调整）。
任一路识别失败时，其权重自动重分配给其他路。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional, Tuple

from app.config import Settings, get_settings
from app.intent.keyword_intent import KeywordIntentRecognizer
from app.intent.llm_intent import LLMIntentRecognizer
from app.intent.vector_intent import VectorIntentRecognizer
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

    async def recognize(self, message: str) -> Tuple[str, float]:
        """三路融合识别用户消息意图。

        Args:
            message: 用户输入消息

        Returns:
            (意图标签, 融合置信度) 元组
            - 若所有路均失败，返回 ("small_talk", 1.0) 作为默认兜底
        """
        # 1. 三路并行执行：LLM 最慢，向量和关键词不依赖 LLM 结果，
        #    通过 asyncio.gather 并发跑，总耗时 = max(各路耗时) 而非 sum
        llm_task = self._llm.recognize(message)
        vec_task = self._vector.recognize(message)
        kw_task = self._keyword.recognize(message)

        results = await asyncio.gather(
            llm_task, vec_task, kw_task, return_exceptions=True
        )

        llm_result: Optional[Dict[str, float]] = None
        vector_result: Optional[Dict[str, float]] = None
        keyword_result: Optional[Dict[str, float]] = None

        for i, (result, name) in enumerate(zip(
            results, ["LLM", "向量", "关键词"]
        )):
            if isinstance(result, Exception):
                logger.warning("融合器: %s 意图识别异常: %s", name, result)
            else:
                if i == 0:
                    llm_result = result
                elif i == 1:
                    vector_result = result
                else:
                    keyword_result = result

        # 2. 动态权重分配：任一识别器失败时，将其权重重新分配给其他识别器
        weights = self._compute_weights(
            bool(llm_result), bool(vector_result), bool(keyword_result)
        )

        # 3. 加权融合
        fused_scores = self._fuse(
            llm_result=llm_result,
            vector_result=vector_result,
            keyword_result=keyword_result,
            weights=weights,
        )

        # 4. 提取最高分意图
        if not fused_scores:
            logger.warning("融合器: 所有识别器均失败，使用默认意图 small_talk")
            metrics.record_intent_confidence("small_talk", 1.0)
            return ("small_talk", 1.0)

        best_intent, confidence = max(fused_scores.items(), key=lambda x: x[1])
        metrics.record_intent_confidence(best_intent, confidence)
        logger.info(
            "融合意图识别: '%s' -> %s (置信度=%.2f, 三路权重: LLM=%.2f VEC=%.2f KW=%.2f)",
            message[:50], best_intent, confidence,
            weights.get("llm", 0), weights.get("vector", 0), weights.get("keyword", 0),
        )
        return (best_intent, confidence)

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


# ============================================================
# 全局单例
# ============================================================

_instance: Optional[IntentFusion] = None


def get_intent_fusion() -> IntentFusion:
    """获取融合意图识别器单例。"""
    global _instance
    if _instance is None:
        _instance = IntentFusion()
    return _instance
