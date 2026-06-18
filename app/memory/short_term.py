"""短期记忆 - 基于 Redis。

以 session_id 为 key，存储当前会话最近 N 轮对话消息。
设置 TTL 自动过期，避免内存无限增长。
每轮对话包含 user 消息和 assistant 回复。

数据结构：Redis List，使用 LPUSH/RPUSH + LTRIM 维护固定长度窗口。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from redis.asyncio import Redis

from app.config import Settings, get_settings
from app.storage import get_redis

logger = logging.getLogger(__name__)

#: Redis key 前缀
_SHORT_TERM_PREFIX = "synapse:short_term"


class ShortTermMemory:
    """短期记忆管理器。

    基于 Redis List 存储，每个 session 对应一个 key。
    通过 LTRIM 保持固定长度的滑动窗口（最近 N 轮）。

    每条消息格式：
        {"role": "user"|"assistant", "content": "...", "timestamp": 123.0}
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings: Settings = settings or get_settings()

    def _key(self, session_id: str) -> str:
        """构建 Redis key。"""
        return f"{_SHORT_TERM_PREFIX}:{session_id}"

    async def _get_redis(self) -> Redis:
        return await get_redis()

    async def append(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """追加一条消息到短期记忆。

        追加后自动裁剪到最大轮数，并刷新 TTL。

        Args:
            session_id: 会话 ID
            role: 消息角色，user / assistant
            content: 消息内容
        """
        redis = await self._get_redis()
        message: Dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
        }
        key = self._key(session_id)

        # RPUSH 追加到列表尾部
        await redis.rpush(key, json.dumps(message, ensure_ascii=False))

        # LTRIM 裁剪：只保留最近 max_rounds * 2 条（每轮含 user + assistant）
        max_items = self._settings.short_term_max_rounds * 2
        await redis.ltrim(key, -max_items, -1)

        # 刷新 TTL
        await redis.expire(key, self._settings.short_term_ttl)
        logger.debug(
            "短期记忆: session=%s 追加 %s 消息", session_id, role
        )

    async def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """获取当前会话的全部短期记忆消息。

        Args:
            session_id: 会话 ID

        Returns:
            消息列表，按时间顺序排列
        """
        redis = await self._get_redis()
        key = self._key(session_id)
        raw_list = await redis.lrange(key, 0, -1)
        messages: List[Dict[str, Any]] = []
        for raw in raw_list:
            try:
                messages.append(json.loads(raw))
            except json.JSONDecodeError:
                logger.warning("短期记忆: 解析消息失败，跳过: %s", raw)
        return messages

    async def get_round_count(self, session_id: str) -> int:
        """获取当前会话的对话轮数（user + assistant 算一轮）。"""
        messages = await self.get_messages(session_id)
        # 统计 user 消息数即为轮数
        return sum(1 for m in messages if m.get("role") == "user")

    async def estimate_tokens(self, session_id: str) -> int:
        """粗略估算当前短期记忆的 Token 数。

        使用简单的字符数 / 4 作为估算（中英文混合近似）。
        更精确的估算可使用 tiktoken，此处保持轻量。
        """
        messages = await self.get_messages(session_id)
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return total_chars // 4

    async def clear(self, session_id: str) -> None:
        """清空指定会话的短期记忆。

        在摘要压缩完成后调用，将短期记忆内容转存长期记忆后清空。
        """
        redis = await self._get_redis()
        key = self._key(session_id)
        await redis.delete(key)
        logger.info("短期记忆: session=%s 已清空", session_id)

    async def delete(self, session_id: str) -> None:
        """删除会话短期记忆（clear 的别名）。"""
        await self.clear(session_id)


# ============================================================
# 全局单例
# ============================================================

_instance: Optional[ShortTermMemory] = None


def get_short_term_memory() -> ShortTermMemory:
    """获取短期记忆管理器单例。"""
    global _instance
    if _instance is None:
        _instance = ShortTermMemory()
    return _instance
