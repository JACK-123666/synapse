"""摘要压缩 Agent。

对长文本或对话历史进行智能摘要，提取关键信息、结论和动作项。
适用于：用户明确要求总结、或记忆压缩器触发自动摘要。

适用意图：summarize
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from app.agents.base import AgentContext, AgentResponse, BaseAgent
from app.llm.client import LLMError, get_llm_client

logger = logging.getLogger(__name__)


class SummarizationAgent(BaseAgent):
    """摘要压缩 Agent。

    将对话历史或用户提供的长文本压缩为结构化摘要，
    保留话题、关键结论、决策和待办事项。

    也可被 MemoryCompressor 内部调用，作为自动压缩的执行 Agent。
    """

    agent_id: str = "summarize_agent"
    description: str = "智能对话与文本摘要 Agent"

    async def execute(self, context: AgentContext) -> AgentResponse:
        """执行摘要生成。

        Args:
            context: 执行上下文，包含短期记忆（待摘要的对话）和用户消息

        Returns:
            包含结构化摘要的响应
        """
        # 1. 收集待摘要内容
        content_to_summarize = self._gather_content(context)

        if not content_to_summarize:
            return AgentResponse(
                reply="当前没有足够的内容可以总结，请先进行一些对话。",
                metadata={"mode": "summarize", "content_length": 0},
            )

        # 2. 构建摘要 prompt
        system_prompt = (
            "你是一个专业的摘要专家。请对提供的对话/文本生成结构化摘要，包含以下部分：\n"
            "1. 【话题】讨论的主题\n"
            "2. 【关键点】重要信息和结论\n"
            "3. 【决策】已做出的决定\n"
            "4. 【待办】后续需要跟进的事项\n"
            "如某部分不适用，可标注「无」。摘要应简明扼要。"
        )

        user_prompt = f"请总结以下内容:\n\n{content_to_summarize}"

        # 3. 调用 LLM
        try:
            llm = get_llm_client()
            reply = await llm.chat(
                messages=[{"role": "user", "content": user_prompt}],
                system=system_prompt,
                temperature=0.3,
                max_tokens=1024,
            )
        except LLMError as exc:
            logger.error("摘要 Agent: LLM 调用失败: %s", exc)
            raise

        metadata: Dict[str, Any] = {
            "mode": "summarize",
            "content_length": len(content_to_summarize),
            "summary_length": len(reply),
        }

        return AgentResponse(reply=reply.strip(), metadata=metadata)

    def _gather_content(self, context: AgentContext) -> str:
        """收集待摘要的内容。

        优先级：
        1. 如果短期记忆中有多轮对话，摘要对话历史
        2. 如果用户消息本身是长文本，直接摘要用户消息
        3. 两者都包含

        Returns:
            拼接后的待摘要文本
        """
        parts: List[str] = []

        # 短期记忆中的对话
        if context.short_term_memory:
            dialogue_lines: List[str] = []
            for msg in context.short_term_memory:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                label = "用户" if role == "user" else "助手"
                dialogue_lines.append(f"{label}: {content}")
            if dialogue_lines:
                parts.append("【对话历史】\n" + "\n".join(dialogue_lines))

        # 用户当前消息（可能是待摘要的长文本）
        if context.message.strip():
            parts.append(f"【当前内容】\n{context.message}")

        return "\n\n".join(parts)
