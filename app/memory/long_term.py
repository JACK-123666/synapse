"""长期情景记忆 - 基于 ChromaDB。

存储历史对话摘要的向量，支持相似情景召回。
当短期记忆触发压缩时，生成的摘要经 embedding 后存入此集合。
处理新消息时，检索与之相似的历史摘要，拼接到 LLM 上下文，
实现跨会话的记忆连续性，同时控制 Token 消耗。

ChromaDB 客户端是同步的，所有操作通过 asyncio.to_thread 包装为异步。
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from app.config import Settings, get_settings
from app.llm.client import LLMError, get_llm_client
from app.storage import get_chroma

logger = logging.getLogger(__name__)


class LongTermMemory:
    """长期情景记忆管理器。

    使用 ChromaDB 的 session_summaries 集合存储对话摘要向量。
    每条记录的元数据包含 session_id、timestamp、user_id 等。

    注意：ChromaDB 在使用自定义 embedding 时，需在创建集合时指定
    embedding_function。本实现通过 LLMClient.embed 获取向量，
    再用 collection.add(ids, embeddings, documents, metadatas) 写入，
    检索时用 collection.query(query_embeddings=...) 做相似度查询。
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings: Settings = settings or get_settings()
        self._collection = None

    def _get_collection(self):
        """获取或创建 ChromaDB 集合（懒加载）。"""
        if self._collection is None:
            client = get_chroma()
            # get_or_create 避免重复创建报错
            self._collection = client.get_or_create_collection(
                name=self._settings.chroma_collection_summary,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "长期记忆: ChromaDB 集合 '%s' 已就绪",
                self._settings.chroma_collection_summary,
            )
        return self._collection

    async def store_summary(
        self,
        session_id: str,
        summary: str,
        user_id: Optional[str] = None,
    ) -> str:
        """存储一条对话摘要到长期记忆。

        将摘要文本 embedding 后存入 ChromaDB。

        Args:
            session_id: 会话 ID
            summary: 摘要文本
            user_id: 用户 ID（可选）

        Returns:
            存储记录的唯一 ID
        """
        # 获取 embedding 向量
        try:
            llm = get_llm_client()
            embedding = await llm.embed(summary)
        except LLMError as exc:
            logger.error("长期记忆: 摘要 embedding 失败: %s", exc)
            raise

        record_id = f"summary_{uuid.uuid4().hex[:12]}"
        metadata: Dict[str, Any] = {
            "session_id": session_id,
            "timestamp": time.time(),
            "user_id": user_id or "anonymous",
        }

        collection = self._get_collection()
        # ChromaDB 同步操作，用 to_thread 包装
        await asyncio.to_thread(
            collection.add,
            ids=[record_id],
            embeddings=[embedding],
            documents=[summary],
            metadatas=[metadata],
        )
        logger.info(
            "长期记忆: session=%s 摘要已存储 (id=%s, 长度=%d)",
            session_id, record_id, len(summary),
        )
        return record_id

    async def recall(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """召回与查询文本相似的历史摘要。

        将查询文本 embedding 后，在 ChromaDB 中做余弦相似度检索。

        Args:
            query_text: 查询文本（通常是用户当前消息）
            top_k: 返回 Top-K 条，默认使用配置值
            user_id: 可选，限定召回某用户的摘要

        Returns:
            相似摘要列表，每项包含 text, score, metadata
        """
        k = top_k or self._settings.long_term_recall_k

        try:
            llm = get_llm_client()
            query_embedding = await llm.embed(query_text)
        except LLMError as exc:
            logger.error("长期记忆: 查询 embedding 失败: %s", exc)
            return []

        collection = self._get_collection()

        # 构建 where 条件（可选按 user_id 过滤）
        where_filter: Optional[Dict[str, Any]] = None
        if user_id:
            where_filter = {"user_id": user_id}

        try:
            results = await asyncio.to_thread(
                collection.query,
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("长期记忆: ChromaDB 检索失败: %s", exc)
            return []

        return self._parse_results(results)

    @staticmethod
    def _parse_results(results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """解析 ChromaDB 查询结果为统一格式。"""
        parsed: List[Dict[str, Any]] = []
        documents = results.get("documents", [[]])
        metadatas = results.get("metadatas", [[]])
        distances = results.get("distances", [[]])

        if not documents or not documents[0]:
            return parsed

        for doc, meta, dist in zip(
            documents[0], metadatas[0] if metadatas else [{}] * len(documents[0]),
            distances[0] if distances else [0.0] * len(documents[0]),
        ):
            # ChromaDB cosine distance 越小越相似，转换为相似度分数
            similarity = max(0.0, 1.0 - dist)
            parsed.append({
                "text": doc,
                "score": similarity,
                "metadata": meta or {},
            })
        return parsed

    async def store_knowledge(
        self,
        knowledge_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """存储知识库条目（供知识检索 Agent 使用）。

        Args:
            knowledge_id: 知识条目 ID
            content: 知识内容文本
            metadata: 附加元数据
        """
        try:
            llm = get_llm_client()
            embedding = await llm.embed(content)
        except LLMError as exc:
            logger.error("长期记忆: 知识 embedding 失败: %s", exc)
            raise

        client = get_chroma()
        collection = client.get_or_create_collection(
            name=self._settings.chroma_collection_knowledge,
            metadata={"hnsw:space": "cosine"},
        )
        await asyncio.to_thread(
            collection.add,
            ids=[knowledge_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata or {}],
        )
        logger.info("长期记忆: 知识条目 '%s' 已存储", knowledge_id)

    async def search_knowledge(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """知识库语义检索。

        Args:
            query_text: 查询文本
            top_k: 返回 Top-K 条

        Returns:
            相似知识条目列表
        """
        try:
            llm = get_llm_client()
            query_embedding = await llm.embed(query_text)
        except LLMError as exc:
            logger.error("长期记忆: 知识检索 embedding 失败: %s", exc)
            return []

        client = get_chroma()
        try:
            collection = client.get_or_create_collection(
                name=self._settings.chroma_collection_knowledge,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("长期记忆: 获取知识库集合失败: %s", exc)
            return []

        try:
            results = await asyncio.to_thread(
                collection.query,
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("长期记忆: 知识库检索失败: %s", exc)
            return []

        return self._parse_results(results)


# ============================================================
# 全局单例
# ============================================================

_instance: Optional[LongTermMemory] = None


def get_long_term_memory() -> LongTermMemory:
    """获取长期记忆管理器单例。"""
    global _instance
    if _instance is None:
        _instance = LongTermMemory()
    return _instance
