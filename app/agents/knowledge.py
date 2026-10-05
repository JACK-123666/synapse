"""知识检索 Agent。

从 ChromaDB 知识库中语义检索相关文档片段，
结合短期/长期记忆和用户画像，调用 LLM 生成知识驱动的回复。

基于 LangChain create_agent 实现：knowledge_retrieval 且允许联网时，
把 web_search / fetch_url 作为工具交给模型，由模型自主决定是否调用。

适用意图：knowledge_retrieval（兼处理 small_talk、summarize 兜底）
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

from langchain_core.messages import BaseMessage

from app.agents.base import AgentContext
from app.agents.langchain_agent import LangChainAgent, PreparedRun
from app.memory.archive import get_long_term_memory

logger = logging.getLogger(__name__)


class RetrievalAgent(LangChainAgent):
    """知识检索 Agent。

    工作流程：
    1. 从 ChromaDB 知识库（全局知识库 + 当前用户可访问的 RAG 知识库）检索与用户消息语义相似的文档。
    2. 构建增强 prompt：知识上下文 + 长期记忆召回 + 短期对话 + 用户画像。
    3. 调用 LLM 生成基于检索增强的回复（可按需调用联网工具）。
    """

    agent_id: str = "retrieval_agent"
    description: str = "基于向量检索的知识问答 Agent"
    tool_tags = ("web",)
    temperature = 0.5
    max_tokens = 2048

    async def prepare(self, context: AgentContext) -> PreparedRun:
        """执行知识检索 / 闲聊回复前的准备。

        small_talk：直接以自然友好的方式聊天，不走知识检索。
        knowledge_retrieval：从知识库检索 + long‑term 召回，LLM 综合回答。
        summarize：委托上下文中的记忆做摘要（实际由 SummarizationAgent 处理，
        此处兜底处理未路由到 summarize 的情况）。
        """
        intent = context.intent
        long_term = get_long_term_memory()
        web_text: str = ""

        # ---- knowledge_retrieval / summarize：检索知识 ----
        knowledge_text: str = ""
        knowledge_results: List[Dict[str, Any]] = []
        if intent in ("knowledge_retrieval", "summarize"):
            try:
                knowledge_results = await long_term.search_knowledge(
                    query_text=context.message,
                    top_k=5,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("检索 Agent: 知识库检索异常: %s", exc)

            # 当前用户可访问的 RAG 知识库
            try:
                from app.capabilities.knowledge.service import get_knowledge_service

                kb_results = await get_knowledge_service().search_for_current_user(
                    context.message
                )
                knowledge_results.extend(kb_results)
            except Exception as exc:  # noqa: BLE001
                logger.warning("检索 Agent: RAG 知识库检索异常: %s", exc)

            if knowledge_results:
                knowledge_results.sort(key=lambda r: r.get("score", 0), reverse=True)
                snippets = [self._format_snippet(r) for r in knowledge_results if r.get("text")]
                knowledge_text = "\n---\n".join(snippets)
                logger.info(
                    "检索 Agent: session=%s 检索到 %d 条知识",
                    context.session_id, len(knowledge_results),
                )

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

        # 联网搜索：knowledge_retrieval 且允许联网时，交由模型通过 Function Calling 决定是否调用
        tools = []
        if intent == "knowledge_retrieval" and context.web_search:
            tools = self.select_tools(context)

        metadata: Dict[str, Any] = {
            "mode": "knowledge_retrieval",
            "sources": [
                {
                    "text": r["text"][:200],
                    "score": r["score"],
                    **{k: r[k] for k in ("kb", "filename", "doc_id") if k in r},
                }
                for r in knowledge_results[:3]
            ],
            "recall_count": len(context.long_term_recall),
        }
        return PreparedRun(system_prompt=system_prompt, tools=tools, metadata=metadata)

    def build_metadata(
        self,
        context: AgentContext,
        prepared: PreparedRun,
        new_messages: Sequence[BaseMessage],
    ) -> Dict[str, Any]:
        metadata = super().build_metadata(context, prepared, new_messages)
        metadata["mode"] = "knowledge_retrieval"
        web_results: List[Dict[str, str]] = []
        for artifact in metadata.get("_artifacts", {}).get("web_search", []):
            if isinstance(artifact, list):
                web_results.extend(artifact)
        metadata["web_sources"] = web_results[:5]
        return metadata

    @staticmethod
    def _format_snippet(result: Dict[str, Any]) -> str:
        """知识片段前加上来源（知识库 / 文件名），方便模型引用。"""
        text = result.get("text", "")
        source = " / ".join(
            str(result[k]) for k in ("kb", "filename") if result.get(k)
        )
        return f"[来源: {source}]\n{text}" if source else text

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
            parts.append("（引用本地知识库内容时请注明来源）")

        if web_text:
            parts.append(f"\n【联网搜索结果】\n{web_text}")
            parts.append("（以上为实时联网搜索结果，请引用时注明来源链接）")

        if recall_text:
            parts.append(f"\n【历史相关摘要】\n{recall_text}")

        if user_profile_text:
            parts.append(f"\n【用户画像】\n{user_profile_text}")

        parts.append(
            "\n【要求】回答简洁清晰，必要时分点说明。若涉及技术细节请确保准确。"
            "本地资料不足或需要最新信息时，可调用联网搜索 / 网页抓取工具，并注明来源链接。"
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
