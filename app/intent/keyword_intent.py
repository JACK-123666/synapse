"""关键词加权投票意图识别。

基于预定义的意图关键词字典，对用户消息进行关键词匹配。
匹配到的关键词数量作为该意图的投票得分。

阈值最高者得胜，简单且不含外部依赖，适合作为融合策略中的稳定底色。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class KeywordIntentRecognizer:
    """基于关键词匹配的意图识别器。

    使用配置中的 intent_keywords 字典，对用户消息分词匹配。
    为每个意图计算匹配关键词的加权得分（匹配数量 / 该意图总关键词数），
    最后归一化得到各意图的置信度分布。
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings: Settings = settings or get_settings()
        # 预计算各意图的关键词总数，避免每次计算
        self._keyword_counts: Dict[str, int] = {
            intent: len(keywords)
            for intent, keywords in self._settings.intent_keywords.items()
        }

    async def recognize(self, message: str) -> Optional[Dict[str, float]]:
        """识别用户消息的意图（关键词匹配）。

        对每个意图统计其关键词在消息中的命中数，
        归一化后作为置信度分布返回。

        Args:
            message: 用户输入消息

        Returns:
            {意图标签: 得分} 字典，无任何匹配时返回 None
        """
        if not message.strip():
            return None

        # 转为小写便于匹配（中英文混合，小写不影响中文）
        lower_msg = message.lower()

        # 各意图的关键词命中数
        hit_counts: Dict[str, int] = {}
        for intent, keywords in self._settings.intent_keywords.items():
            hits = 0
            for kw in keywords:
                if kw.lower() in lower_msg:
                    hits += 1
            hit_counts[intent] = hits

        total_hits = sum(hit_counts.values())
        if total_hits == 0:
            logger.debug("关键词意图识别: 无关键词命中: '%s'", message[:50])
            return None

        # 归一化为置信度
        scores = {
            intent: count / total_hits
            for intent, count in hit_counts.items()
        }

        logger.debug("关键词意图识别: '%s' -> %s", message[:50], scores)
        return scores
