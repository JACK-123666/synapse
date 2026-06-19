"""记忆摘要压缩器。

当短期记忆达到阈值（轮数或 Token 数）时，触发 LLM 生成对话摘要，
将摘要存入长期记忆（ChromaDB），然后清空短期记忆。

核心效果：通过摘要压缩和按需召回，大幅降低传递给 LLM 的上下文 Token 数，
解决长对话导致会话记忆膨胀的问题。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.config import Settings, get_settings
from app.llm.gateway import LLMError, get_llm_client
from app.memory.archive import LongTermMemory, get_long_term_memory
from app.memory.recent import ShortTermMemory, get_short_term_memory
from app.observability import metrics

logger = logging.getLogger(__name__)


class MemoryCompressor:
    """记忆摘要压缩器。

    工作流程：
    1. 检查短期记忆是否达到压缩阈值（轮数 >= summary_trigger_rounds
       或 Token 估算 >= token_budget）。
    2. 若达到阈值，取出全部短期记忆消息，拼接为对话文本。
    3. 调用 LLM 生成摘要。
    4. 将摘要存入长期记忆（ChromaDB session_summaries 集合）。
    5. 清空短期记忆。

    压缩是幂等的：即使 LLM 调用失败，也只会记录错误，不会清空短期记忆，
    下次请求会再次尝试压缩。
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings: Settings = settings or get_settings()
        self._short_term: ShortTermMemory = get_short_term_memory()
        self._long_term: LongTermMemory = get_long_term_memory()

    async def should_compress(self, session_id: str) -> bool:
        """判断是否需要触发压缩。

        当短期记忆轮数 >= summary_trigger_rounds，
        或 Token 估算 >= token_budget 时返回 True。

        Args:
            session_id: 会话 ID

        Returns:
            是否需要压缩
        """
        round_count = await self._short_term.get_round_count(session_id)
        if round_count >= self._settings.summary_trigger_rounds:
            logger.info(
                "压缩判断: session=%s 轮数 %d >= 阈值 %d，触发压缩",
                session_id, round_count, self._settings.summary_trigger_rounds,
            )
            return True

        token_est = await self._short_term.estimate_tokens(session_id)
        if token_est >= self._settings.token_budget:
            logger.info(
                "压缩判断: session=%s Token 估算 %d >= 预算 %d，触发压缩",
                session_id, token_est, self._settings.token_budget,
            )
            return True

        return False

    async def compress(
        self,
        session_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """执行摘要压缩。

        Args:
            session_id: 会话 ID
            user_id: 用户 ID（写入长期记忆元数据）

        Returns:
            生成的摘要文本，压缩失败时返回 None
        """
        # 取出短期记忆消息
        messages = await self._short_term.get_messages(session_id)
        if not messages:
            logger.debug("压缩: session=%s 无短期记忆，跳过", session_id)
            return None

        # 拼接对话文本
        dialogue_text = self._format_dialogue(messages)

        # 调用 LLM 生成摘要
        try:
            summary = await self._generate_summary(dialogue_text)
        except LLMError as exc:
            logger.error("压缩: session=%s LLM 摘要生成失败: %s", session_id, exc)
            return None

        # 存入长期记忆
        try:
            await self._long_term.store_summary(
                session_id=session_id,
                summary=summary,
                user_id=user_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("压缩: session=%s 长期记忆存储失败: %s", session_id, exc)
            return None

        # 清空短期记忆
        await self._short_term.clear(session_id)

        # 记录指标
        metrics.record_compression()

        logger.info(
            "压缩完成: session=%s 摘要长度=%d，短期记忆已清空",
            session_id, len(summary),
        )
        return summary

    async def compress_if_needed(
        self,
        session_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """检查并按需执行压缩（便捷方法）。

        Returns:
            压缩后的摘要文本（如果触发了压缩），否则 None
        """
        if await self.should_compress(session_id):
            return await self.compress(session_id, user_id)
        return None

    def _format_dialogue(self, messages: List[Dict[str, Any]]) -> str:
        """将短期记忆消息列表格式化为对话文本。

        Args:
            messages: 消息列表

        Returns:
            格式化的对话文本
        """
        lines: List[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            label = "用户" if role == "user" else "助手"
            lines.append(f"{label}: {content}")
        return "\n".join(lines)

    async def _generate_summary(self, dialogue_text: str) -> str:
        """调用 LLM 生成对话摘要。

        使用结构化 prompt，要求 LLM 提取关键信息、话题、结论。

        Args:
            dialogue_text: 对话文本

        Returns:
            摘要文本

        Raises:
            LLMError: LLM 调用失败
        """
        system_prompt = (
            "你是一个对话摘要专家。请将以下对话压缩为简洁的摘要，"
            "保留关键信息、讨论的话题、达成的结论和待办事项。"
            "摘要应控制在 200 字以内，便于后续检索召回。"
        )
        user_prompt = f"请总结以下对话:\n\n{dialogue_text}"

        llm = get_llm_client()
        summary = await llm.chat(
            messages=[{"role": "user", "content": user_prompt}],
            system=system_prompt,
            temperature=0.3,
            max_tokens=512,
        )
        return summary.strip()


# 全局单例

_instance: Optional[MemoryCompressor] = None


def get_memory_compressor() -> MemoryCompressor:
    """获取记忆压缩器单例。"""
    global _instance
    if _instance is None:
        _instance = MemoryCompressor()
    return _instance
