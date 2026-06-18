"""用户画像管理 - 基于 Redis + ChromaDB。

维护用户的静态/动态特征：
- 偏好标签（如技术栈、语言偏好）
- 常用术语
- 历史交互摘要标签

在构建 LLM prompt 时注入用户画像，实现个性化响应。
静态特征存 Redis（快速读写），动态特征向量存 ChromaDB（语义检索）。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from app.config import Settings, get_settings
from app.storage import get_redis

logger = logging.getLogger(__name__)

#: Redis key 前缀
_PROFILE_PREFIX = "synapse:user_profile"


class UserProfileManager:
    """用户画像管理器。

    数据模型（Redis JSON）：
        {
            "preferences": ["python", "fastapi", ...],   # 偏好标签
            "frequent_terms": ["向量数据库", "RAG", ...], # 常用术语
            "interaction_count": 42,                      # 交互次数
            "custom": {}                                  # 自定义字段
        }
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings: Settings = settings or get_settings()

    def _key(self, user_id: str) -> str:
        """构建 Redis key。"""
        return f"{_PROFILE_PREFIX}:{user_id}"

    async def get_profile(self, user_id: Optional[str]) -> Dict[str, Any]:
        """获取用户画像。

        Args:
            user_id: 用户 ID，为 None 时返回空画像

        Returns:
            用户画像字典
        """
        if not user_id:
            return {}

        from app.storage import get_redis
        redis = await get_redis()
        key = self._key(user_id)
        raw = await redis.get(key)
        if raw is None:
            # 首次访问，初始化空画像
            profile: Dict[str, Any] = {
                "preferences": [],
                "frequent_terms": [],
                "interaction_count": 0,
                "custom": {},
            }
            await self.save_profile(user_id, profile)
            return profile
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("用户画像: 解析失败，重置: user=%s", user_id)
            return {
                "preferences": [],
                "frequent_terms": [],
                "interaction_count": 0,
                "custom": {},
            }

    async def save_profile(
        self, user_id: str, profile: Dict[str, Any]
    ) -> None:
        """保存用户画像。"""
        from app.storage import get_redis
        redis = await get_redis()
        key = self._key(user_id)
        await redis.set(key, json.dumps(profile, ensure_ascii=False))

    async def update_preferences(
        self, user_id: Optional[str], preferences: List[str]
    ) -> None:
        """更新用户偏好标签（去重合并）。"""
        if not user_id:
            return
        profile = await self.get_profile(user_id)
        existing = set(profile.get("preferences", []))
        existing.update(preferences)
        profile["preferences"] = list(existing)
        await self.save_profile(user_id, profile)

    async def add_frequent_terms(
        self, user_id: Optional[str], terms: List[str]
    ) -> None:
        """添加常用术语（去重合并）。"""
        if not user_id:
            return
        profile = await self.get_profile(user_id)
        existing = set(profile.get("frequent_terms", []))
        existing.update(terms)
        # 最多保留 50 个常用术语
        profile["frequent_terms"] = list(existing)[:50]
        await self.save_profile(user_id, profile)

    async def increment_interaction(self, user_id: Optional[str]) -> None:
        """递增用户交互次数。"""
        if not user_id:
            return
        profile = await self.get_profile(user_id)
        profile["interaction_count"] = profile.get("interaction_count", 0) + 1
        await self.save_profile(user_id, profile)

    async def build_prompt_context(
        self, user_id: Optional[str]
    ) -> str:
        """构建注入 LLM prompt 的用户画像上下文文本。

        Args:
            user_id: 用户 ID

        Returns:
            格式化的用户画像描述文本，无画像时返回空字符串
        """
        profile = await self.get_profile(user_id)
        if not profile or profile.get("interaction_count", 0) == 0:
            return ""

        parts: List[str] = []
        prefs = profile.get("preferences", [])
        if prefs:
            parts.append(f"用户偏好: {', '.join(prefs)}")

        terms = profile.get("frequent_terms", [])
        if terms:
            parts.append(f"常用术语: {', '.join(terms)}")

        count = profile.get("interaction_count", 0)
        parts.append(f"历史交互次数: {count}")

        return "\n".join(parts)


# ============================================================
# 全局单例
# ============================================================

_instance: Optional[UserProfileManager] = None


def get_user_profile_manager() -> UserProfileManager:
    """获取用户画像管理器单例。"""
    global _instance
    if _instance is None:
        _instance = UserProfileManager()
    return _instance
