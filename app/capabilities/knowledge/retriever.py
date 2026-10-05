"""LangChain 检索器适配。

chromadb 固定在 0.4.24（langchain-chroma 需要 chromadb>=1.3.5），
因此自写 BaseRetriever，复用 KnowledgeService 的权限与检索逻辑，
可直接用于 LangChain 的 RAG 链（如 create_retrieval_chain）。
"""

from __future__ import annotations

from typing import List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document as LCDocument
from langchain_core.retrievers import BaseRetriever

from app.capabilities.knowledge.service import Principal, get_knowledge_service


class KnowledgeRetriever(BaseRetriever):
    """按用户权限检索知识库的 LangChain 检索器。"""

    user_id: str
    is_admin: bool = False
    kb_ids: Optional[List[str]] = None
    k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[LCDocument]:
        raise NotImplementedError("KnowledgeRetriever 仅支持异步调用：请使用 ainvoke")

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[LCDocument]:
        results = await get_knowledge_service().search(
            Principal(self.user_id, self.is_admin), query, kb_ids=self.kb_ids, top_k=self.k
        )
        return [
            LCDocument(
                page_content=r["text"],
                metadata={k: v for k, v in r.items() if k != "text"},
            )
            for r in results
        ]
