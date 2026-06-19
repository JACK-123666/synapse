"""知识检索 Agent。

从 ChromaDB 知识库中语义检索相关文档片段，
结合短期/长期记忆和用户画像，调用 LLM 生成知识驱动的回复。

适用意图：knowledge_retrieval
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.agents.base import AgentContext, AgentResponse, BaseAgent
from app.llm.gateway import LLMError, get_llm_client
from app.memory.archive import get_long_term_memory
from app.tools.search import web_search

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
        """执行知识检索 / 闲聊回复。

        small_talk：直接以自然友好的方式聊天，不走知识检索。
        knowledge_retrieval：从知识库检索 + long‑term 召回，LLM 综合回答。
        summarize：委托上下文中的记忆做摘要（实际由 SummarizationAgent 处理，
        此处兜底处理未路由到 summarize 的情况）。

        Args:
            context: 执行上下文

        Returns:
            知识驱动 / 闲聊回复
        """
        intent = context.intent
        long_term = get_long_term_memory()

        # ---- knowledge_retrieval / summarize：检索知识 ----
        knowledge_text: str = ""
        knowledge_results: List[Dict[str, Any]] = []
        if intent in ("knowledge_retrieval", "summarize"):
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

            # ---- 知识库不足时，联网搜索（受 context.web_search 控制） ----
            web_results: List[Dict[str, str]] = []
            web_text: str = ""
            if not knowledge_results and intent == "knowledge_retrieval" and context.web_search:
                try:
                    web_results = await web_search(context.message, max_results=5)
                    if web_results:
                        web_parts = [
                            f"- [{r['title']}]({r['url']})\n  {r['snippet']}"
                            for r in web_results
                        ]
                        web_text = "\n\n".join(web_parts)
                        logger.info(
                            "检索 Agent: session=%s 联网搜索到 %d 条结果",
                            context.session_id, len(web_results),
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("检索 Agent: 联网搜索异常: %s", exc)

        # ---- 按意图选择 system prompt ----
        if intent == "small_talk":
            system_prompt = self._build_chat_prompt(
                recall_text=self._format_recall(context.long_term_recall),
                user_profile_text=context.user_profile_context,
            )
        elif intent == "summarize":
            system_prompt = self._build_summarize_prompt(
                knowledge_text=knowledge_text,
                recall_text=self._format_recall(context.long_term_recall),
                user_profile_text=context.user_profile_context,
            )
        else:
            # knowledge_retrieval 及其他意图走知识检索 prompt
            system_prompt = self._build_retrieval_prompt(
                knowledge_text=knowledge_text,
                web_text=web_text,
                recall_text=self._format_recall(context.long_term_recall),
                user_profile_text=context.user_profile_context,
            )

        # 构建消息列表
        messages = self._build_messages(
            short_term=context.short_term_memory,
            current_message=context.message,
        )

        # 调用 LLM 生成回复
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

        # 构建元数据
        metadata: Dict[str, Any] = {
            "mode": "knowledge_retrieval",
            "sources": [
                {"text": r["text"][:200], "score": r["score"]}
                for r in knowledge_results[:3]
            ],
            "recall_count": len(context.long_term_recall),
        }

        return AgentResponse(reply=reply.strip(), metadata=metadata)

    def _build_retrieval_prompt(
        self,
        knowledge_text: str,
        web_text: str,
        recall_text: str,
        user_profile_text: str,
    ) -> str:
        """知识检索 system prompt。"""
        parts: List[str] = [
            "你是一个知识渊博的 AI 助手，用检索到的参考资料回答问题。",
            "请基于提供的上下文给出准确、有用的回答，如果上下文不足则诚实说明。",
        ]

        if knowledge_text:
            parts.append(f"\n【本地知识库】\n{knowledge_text}")

        if web_text:
            parts.append(f"\n【联网搜索结果】\n{web_text}")
            parts.append("（以上为实时联网搜索结果，请引用时注明来源链接）")

        if recall_text:
            parts.append(f"\n【历史相关摘要】\n{recall_text}")

        if user_profile_text:
            parts.append(f"\n【用户画像】\n{user_profile_text}")

        parts.append(
            "\n【要求】回答简洁清晰，必要时分点说明。若涉及技术细节请确保准确。"
        )
        return "\n".join(parts)

    def _build_chat_prompt(
        self,
        recall_text: str,
        user_profile_text: str,
    ) -> str:
        """闲聊 system prompt：自然友好，不强制检索。"""
        parts: List[str] = [
            "你是一个友好、善解人意的 AI 聊天助手，用自然口语化的方式与用户交流。",
            "保持对话轻松愉快，适当使用语气词让回复更亲切。",
            "如果用户问具体问题，认真回答；如果是打招呼或闲聊，就轻松回应。",
        ]

        if recall_text:
            parts.append(f"\n【你可能记得的历史对话】\n{recall_text}")

        if user_profile_text:
            parts.append(f"\n【关于这位用户】\n{user_profile_text}")

        return "\n".join(parts)

    def _build_summarize_prompt(
        self,
        knowledge_text: str,
        recall_text: str,
        user_profile_text: str,
    ) -> str:
        """摘要 system prompt（兜底，正常由 SummarizationAgent 处理）。"""
        parts: List[str] = [
            "你是一个专业的摘要与总结助手。请对用户提供的内容进行结构化总结。",
            "包含：核心要点、关键结论、需要跟进的事项。",
        ]

        if knowledge_text:
            parts.append(f"\n【参考资料】\n{knowledge_text}")

        if recall_text:
            parts.append(f"\n【历史上下文】\n{recall_text}")

        if user_profile_text:
            parts.append(f"\n【用户信息】\n{user_profile_text}")

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
