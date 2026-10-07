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
from typing import Any, Callable, Dict, List, Optional

from redis.exceptions import WatchError

from app.config import Settings, get_settings
from app.store import get_redis

logger = logging.getLogger(__name__)

#: Redis key 前缀
_PROFILE_PREFIX = "synapse:user_profile"
#: 画像并发更新冲突时的最大重试次数
_MAX_UPDATE_RETRIES = 5
#: frequent_terms 里最多保留多少个词（按出现次数排序，超出丢弃低频词）
_MAX_TRACKED_TERMS = 100
#: 注入 prompt 的术语数量上限 —— 太多既费 token 又稀释重点
_MAX_PROMPT_TERMS = 8
#: 至少出现这么多次才认为"常用"。一次性出现的短语（"我叫小明"）不入画像，
#: 这是让画像从"噪声收集器"变成"真实特征"的关键一步。
_MIN_TERM_FREQ = 2
#: 常见停用词：出现频率再高也不代表用户特征
_STOPWORDS = {
    "你好", "您好", "谢谢", "多谢", "再见", "请问", "帮我", "帮忙", "一下",
    "什么", "怎么", "如何", "为什么", "可以", "能否", "是否", "这个", "那个",
    "现在", "今天", "明天", "昨天", "我们", "你们", "他们", "自己", "知道",
    "hi", "hello", "hey", "thanks", "thank", "please", "the", "and", "you",
    "for", "with", "this", "that", "what", "how", "why",
}


def _clean_terms(terms: List[str]) -> List[str]:
    """过滤掉停用词与长度不合理的片段。"""
    out: List[str] = []
    for raw in terms:
        term = str(raw).strip()
        if not (2 <= len(term) <= 16):
            continue
        if term.lower() in _STOPWORDS:
            continue
        out.append(term)
    return out


def _term_counts(profile: Dict[str, Any]) -> Dict[str, int]:
    """读取词频表，兼容早期把 frequent_terms 存成 list 的数据。"""
    counts = profile.get("frequent_terms") or {}
    if isinstance(counts, list):
        return {str(t): 1 for t in counts}
    if isinstance(counts, dict):
        return {str(k): int(v) for k, v in counts.items()}
    return {}


class UserProfileManager:
    """用户画像管理器。

    数据模型（Redis JSON）：
        {
            "preferences": ["用中文回答", "简洁一点"],       # 偏好标签（显式设置）
            "frequent_terms": {"向量数据库": 7, "RAG": 5},  # 术语 -> 出现次数
            "interaction_count": 42,                        # 交互次数（仅统计，不注入 prompt）
            "custom": {}                                    # 自定义字段
        }

    注入 prompt 的原则：只放真正能改善回答的信息。因此交互次数不注入，
    术语也要求出现次数达到 _MIN_TERM_FREQ 且只取 Top-N。
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

        from app.store import get_redis
        redis = await get_redis()
        key = self._key(user_id)
        raw = await redis.get(key)
        if raw is None:
            # 首次访问，初始化空画像
            profile = self._empty_profile()
            await self.save_profile(user_id, profile)
            return profile
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else self._empty_profile()
        except json.JSONDecodeError:
            logger.warning("用户画像: 解析失败，重置: user=%s", user_id)
            return self._empty_profile()

    async def save_profile(
        self, user_id: str, profile: Dict[str, Any]
    ) -> None:
        """保存用户画像。"""
        from app.store import get_redis
        redis = await get_redis()
        key = self._key(user_id)
        await redis.set(key, json.dumps(profile, ensure_ascii=False))

    @staticmethod
    def _empty_profile() -> Dict[str, Any]:
        return {
            "preferences": [],
            # 词频表 {术语: 出现次数}，注入 prompt 时按次数排序取 Top-N
            "frequent_terms": {},
            "interaction_count": 0,
            "custom": {},
        }

    async def _atomic_update(
        self, user_id: str, mutate: Callable[[Dict[str, Any]], None]
    ) -> Dict[str, Any]:
        """原子地读-改-写用户画像（Redis WATCH 乐观锁，冲突时重试）。

        同一用户的并发请求不会互相覆盖更新。
        """
        redis = await get_redis()
        key = self._key(user_id)
        for _ in range(_MAX_UPDATE_RETRIES):
            try:
                async with redis.pipeline(transaction=True) as pipe:
                    await pipe.watch(key)
                    raw = await pipe.get(key)
                    try:
                        profile = json.loads(raw) if raw else self._empty_profile()
                    except json.JSONDecodeError:
                        profile = self._empty_profile()
                    mutate(profile)
                    pipe.multi()
                    pipe.set(key, json.dumps(profile, ensure_ascii=False))
                    await pipe.execute()
                    return profile
            except WatchError:
                continue
            except (AttributeError, NotImplementedError):
                # 不支持事务的 Redis 替身：退化为普通读-改-写
                break
        profile = await self.get_profile(user_id)
        mutate(profile)
        await self.save_profile(user_id, profile)
        return profile

    async def set_preferences(
        self, user_id: Optional[str], preferences: List[str]
    ) -> None:
        """整体替换用户偏好标签（PUT 语义）。"""
        if not user_id:
            return
        cleaned = [str(p).strip() for p in preferences if str(p).strip()]

        def _mutate(profile: Dict[str, Any]) -> None:
            profile["preferences"] = list(dict.fromkeys(cleaned))

        await self._atomic_update(user_id, _mutate)

    async def set_custom(self, user_id: Optional[str], custom: Dict[str, Any]) -> None:
        """整体替换自定义字段。"""
        if not user_id:
            return

        def _mutate(profile: Dict[str, Any]) -> None:
            profile["custom"] = dict(custom or {})

        await self._atomic_update(user_id, _mutate)

    async def add_frequent_terms(
        self, user_id: Optional[str], terms: List[str]
    ) -> None:
        """累加术语的出现次数（而不是简单去重追加）。

        只有反复出现的词才留得下来：先过一遍停用词与长度过滤，
        注入 prompt 时还要求次数达到 _MIN_TERM_FREQ。
        """
        if not user_id:
            return
        cleaned = _clean_terms(terms)
        if not cleaned:
            return

        def _mutate(profile: Dict[str, Any]) -> None:
            counts = _term_counts(profile)
            for term in cleaned:
                counts[term] = counts.get(term, 0) + 1
            if len(counts) > _MAX_TRACKED_TERMS:
                counts = dict(
                    sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:_MAX_TRACKED_TERMS]
                )
            profile["frequent_terms"] = counts

        await self._atomic_update(user_id, _mutate)

    async def clear(self, user_id: Optional[str]) -> None:
        """删除用户画像。"""
        if not user_id:
            return
        redis = await get_redis()
        await redis.delete(self._key(user_id))
        logger.info("用户画像: 已清除 user=%s", user_id)

    async def increment_interaction(self, user_id: Optional[str]) -> None:
        """递增用户交互次数。"""
        if not user_id:
            return

        def _mutate(profile: Dict[str, Any]) -> None:
            profile["interaction_count"] = profile.get("interaction_count", 0) + 1

        await self._atomic_update(user_id, _mutate)

    async def build_prompt_context(
        self, user_id: Optional[str]
    ) -> str:
        """构建注入 LLM prompt 的用户画像上下文文本。

        Args:
            user_id: 用户 ID

        Returns:
            格式化的用户画像描述文本，无画像时返回空字符串
        """
        if not user_id:
            return ""
        profile = await self.get_profile(user_id)

        parts: List[str] = []
        prefs = profile.get("preferences") or []
        if prefs:
            parts.append("用户偏好: " + ", ".join(str(p) for p in prefs[:10]))

        # 只取反复出现的术语，按次数排序取 Top-N。
        # 刻意不注入"历史交互次数"之类的统计量——它对模型生成没有帮助，纯费 token。
        counts = _term_counts(profile)
        ranked = sorted(
            ((t, c) for t, c in counts.items() if c >= _MIN_TERM_FREQ),
            key=lambda kv: kv[1],
            reverse=True,
        )[:_MAX_PROMPT_TERMS]
        if ranked:
            parts.append("用户常提到: " + ", ".join(t for t, _ in ranked))

        return "\n".join(parts)


# 全局单例

_instance: Optional[UserProfileManager] = None


def get_user_profile_manager() -> UserProfileManager:
    """获取用户画像管理器单例。"""
    global _instance
    if _instance is None:
        _instance = UserProfileManager()
    return _instance
