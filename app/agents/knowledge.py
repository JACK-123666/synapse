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

    #: 三种意图各自的静态系统提示词；动态内容（检索结果 / 记忆 / 画像）走 context_blocks
    _RETRIEVAL_PROMPT = (
        "你是一个知识渊博的 AI 助手，用检索到的参考资料回答问题。\n"
        "请基于提供的上下文给出准确、有用的回答，如果上下文不足则诚实说明。\n"
        "引用【本地知识库检索结果】或联网结果时请注明来源。\n"
        "本地资料不足或需要最新信息时，可调用联网搜索 / 网页抓取工具。\n"
        "回答简洁清晰，必要时分点说明；涉及技术细节请确保准确。"
    )
    _CHAT_PROMPT = (
        "你是一个友好、善解人意的 AI 聊天助手，用自然口语化的方式与用户交流。\n"
        "保持对话轻松愉快，适当使用语气词让回复更亲切。\n"
        "如果用户问具体问题，认真回答；如果是打招呼或闲聊，就轻松回应。"
    )
    _SUMMARIZE_PROMPT = (
        "你是一个专业的摘要与总结助手。请对用户提供的内容进行结构化总结。\n"
        "包含：核心要点、关键结论、需要跟进的事项。"
    )

    async def prepare(self, context: AgentContext) -> PreparedRun:
        """执行知识检索 / 闲聊回复前的准备。

        small_talk：直接以自然友好的方式聊天，不走知识检索。
        knowledge_retrieval：从知识库检索，结果作为上下文块注入消息序列。
        summarize：委托上下文中的记忆做摘要（正常由 SummarizationAgent 处理，
        此处兜底处理未路由到 summarize 的情况）。
        """
        intent = context.intent

        # ---- knowledge_retrieval / summarize：检索知识 ----
        knowledge_results: List[Dict[str, Any]] = []
        if intent in ("knowledge_retrieval", "summarize"):
            # 只查真正的 RAG 知识库（每个库一个 kb_<id> 集合）。
            # 此前这里还会多查一次遗留的全局 knowledge_base 集合，但它没有任何
            # 写入入口、永远为空，等于每个知识问题白付一次 embedding + 一次 Chroma 往返。
            # top_k 不传，由 KnowledgeService 读 RAG_TOP_K 配置。
            try:
                from app.capabilities.knowledge.service import get_knowledge_service

                kb_results = await get_knowledge_service().search_for_current_user(
                    context.message
                )
                knowledge_results.extend(kb_results)
            except Exception as exc:  # noqa: BLE001
                logger.warning("检索 Agent: RAG 知识库检索异常: %s", exc)

        # 检索结果按分数排序，打包成上下文块交给 build_context_block
        context_blocks: List[str] = []
        if knowledge_results:
            knowledge_results.sort(key=lambda r: r.get("score", 0), reverse=True)
            snippets = [self._format_snippet(r) for r in knowledge_results if r.get("text")]
            if snippets:
                context_blocks.append(
                    "【本地知识库检索结果】\n" + "\n---\n".join(snippets)
                )
                logger.info(
                    "检索 Agent: session=%s 检索到 %d 条知识",
                    context.session_id, len(knowledge_results),
                )

        # 系统提示词只按意图选择静态文案，逐字节稳定 → Agent 图可缓存、prefix cache 可命中
        system_prompt = {
            "small_talk": self._CHAT_PROMPT,
            "summarize": self._SUMMARIZE_PROMPT,
        }.get(intent, self._RETRIEVAL_PROMPT)

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
        return PreparedRun(
            system_prompt=system_prompt,
            tools=tools,
            metadata=metadata,
            context_blocks=context_blocks,
        )

    def build_metadata(
        self,
        context: AgentContext,
        prepared: PreparedRun,
        new_messages: Sequence[BaseMessage],
    ) -> Dict[str, Any]:
        """在通用元数据之上补充知识来源与联网引用。"""
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
