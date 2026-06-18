"""知识检索 Agent。

从 ChromaDB 知识库中语义检索相关文档片段，
结合短期/长期记忆和用户画像，调用 LLM 生成知识驱动的回复。

适用意图：knowledge_retrieval
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.agents.base import AgentContext, AgentResponse, BaseAgent
from app.llm.client import LLMError, get_llm_client
from app.memory.long_term import get_long_term_memory

logger = logging.getLogger(__name__)


class RetrievalAgent(BaseAgent):
    """知识检索 Agent。

    工作流程：
    1. 从 ChromaDB 知识库检索与用户消息语义相似的知识文档。
    2. 构建增强 prompt：知识上下文 + 长期记忆召回 + 短期对话 + 用户画像。
    3. 调用 LLM 生成基于检索增强的回复。
    """

    agent_id: str = "retrieval_agent"
    description: str = "基于向量检索的知识问答 Agent"

    async def execute(self, context: AgentContext) -> AgentResponse:
        """执行知识检索与回答。

        Args:
            context: 执行上下文

        Returns:
            包含知识驱动的回复和检索来源元数据
        """
        long_term = get_long_term_memory()

        # 1. 知识库语义检索
        knowledge_results: List[Dict[str, Any]] = []
        knowledge_text: str = ""
        try:
            knowledge_results = await long_term.search_knowledge(
                query_text=context.message,
                top_k=5,
            )
            if knowledge_results:
                snippets = [r["text"] for r in knowledge_results if r.get("text")]
                knowledge_text = "\n---\n".join(snippets)
                logger.info(
                    "检索 Agent: session=%s 检索到 %d 条知识",
                    context.session_id, len(knowledge_results),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("检索 Agent: 知识库检索异常: %s", exc)

        # 2. 构建增强 prompt
        system_prompt = self._build_system_prompt(
            knowledge_text=knowledge_text,
            recall_text=self._format_recall(context.long_term_recall),
            user_profile_text=context.user_profile_context,
        )

        # 3. 构建消息列表
        messages = self._build_messages(
            short_term=context.short_term_memory,
            current_message=context.message,
        )

        # 4. 调用 LLM 生成回复
        try:
            llm = get_llm_client()
            reply = await llm.chat(
                messages=messages,
                system=system_prompt,
                temperature=0.5,
                max_tokens=2048,
            )
        except LLMError as exc:
            logger.error("检索 Agent: LLM 调用失败: %s", exc)
            raise

        # 5. 构建元数据
        metadata: Dict[str, Any] = {
            "mode": "knowledge_retrieval",
            "sources": [
                {"text": r["text"][:200], "score": r["score"]}
                for r in knowledge_results[:3]
            ],
            "recall_count": len(context.long_term_recall),
        }

        return AgentResponse(reply=reply.strip(), metadata=metadata)

    def _build_system_prompt(
        self,
        knowledge_text: str,
        recall_text: str,
        user_profile_text: str,
    ) -> str:
        """构建系统提示词，注入检索到的知识、历史摘要和用户画像。"""
        parts: List[str] = [
            "你是一个知识渊博的 AI 助手，用检索到的参考资料回答问题。",
            "请基于提供的上下文给出准确、有用的回答，如果上下文不足则诚实说明。",
        ]

        if knowledge_text:
            parts.append(f"\n【参考知识库】\n{knowledge_text}")

        if recall_text:
            parts.append(f"\n【历史相关摘要】\n{recall_text}")

        if user_profile_text:
            parts.append(f"\n【用户画像】\n{user_profile_text}")

        parts.append(
            "\n【要求】回答简洁清晰，必要时分点说明。若涉及技术细节请确保准确。"
        )
        return "\n".join(parts)

    def _build_messages(
        self,
        short_term: List[Dict[str, Any]],
        current_message: str,
    ) -> List[Dict[str, str]]:
        """构建 LLM 消息列表：短期记忆 + 当前消息。"""
        messages: List[Dict[str, str]] = []
        # 包含最近的对话历史
        for msg in short_term:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            messages.append({"role": role, "content": content})
        # 当前消息
        messages.append({"role": "user", "content": current_message})
        return messages

    @staticmethod
    def _format_recall(recall: List[Dict[str, Any]]) -> str:
        """格式化长期记忆召回摘要。"""
        if not recall:
            return ""
        parts = []
        for i, item in enumerate(recall, 1):
            text = item.get("text", "")
            score = item.get("score", 0)
            if text:
                parts.append(f"[{i}] (相似度: {score:.2f}) {text}")
        return "\n".join(parts)
