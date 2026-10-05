"""知识库（RAG）服务。

- 元数据（知识库、文档）存关系型数据库，向量存 ChromaDB（每个知识库一个集合）
- 文档解析 → RecursiveCharacterTextSplitter 切片 → LLMClient.embed_batch 向量化 → 写入 Chroma
- 提取出的纯文本另存到 data_dir/kb_texts，用于更换 embedding 模型后重建索引
- 权限：所有者与管理员可写；私有知识库仅所有者可读，共享知识库所有用户可读
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import func, or_, select

from app.capabilities.knowledge.loaders import extract_text
from app.config import get_settings
from app.core.context import get_request_context
from app.core.db import session_scope
from app.llm.gateway import get_llm_client
from app.models import Document, KnowledgeBase, new_id, utcnow
from app.store import get_chroma

logger = logging.getLogger(__name__)

_EMBED_BATCH = 64
_VISIBILITIES = ("private", "shared")


class KnowledgeError(ValueError):
    """知识库操作错误（不存在、参数非法等）。"""


@dataclass
class Principal:
    """执行操作的用户。"""

    user_id: str
    is_admin: bool = False


def current_principal() -> Principal:
    ctx = get_request_context()
    return Principal(user_id=ctx.user_id, is_admin=ctx.is_admin)


def kb_to_dict(kb: KnowledgeBase, document_count: Optional[int] = None) -> Dict[str, Any]:
    data = {
        "id": kb.id,
        "name": kb.name,
        "description": kb.description,
        "owner_id": kb.owner_id,
        "visibility": kb.visibility,
        "created_at": kb.created_at.isoformat() if kb.created_at else None,
    }
    if document_count is not None:
        data["document_count"] = document_count
    return data


def doc_to_dict(doc: Document) -> Dict[str, Any]:
    return {
        "id": doc.id,
        "kb_id": doc.kb_id,
        "filename": doc.filename,
        "source": doc.source,
        "source_uri": doc.source_uri,
        "chunk_count": doc.chunk_count,
        "char_count": doc.char_count,
        "status": doc.status,
        "error": doc.error,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
    }


class KnowledgeService:
    """知识库服务。"""

    def __init__(self) -> None:
        self._settings = get_settings()

    # ---- 权限 ----

    @staticmethod
    def _can_read(kb: KnowledgeBase, who: Principal) -> bool:
        return who.is_admin or kb.owner_id == who.user_id or kb.visibility == "shared"

    @staticmethod
    def _can_write(kb: KnowledgeBase, who: Principal) -> bool:
        return who.is_admin or kb.owner_id == who.user_id

    async def _get_kb(self, session, kb_id: str, who: Principal, write: bool = False) -> KnowledgeBase:
        kb = await session.get(KnowledgeBase, kb_id)
        if kb is None or not self._can_read(kb, who):
            raise KnowledgeError("知识库不存在")
        if write and not self._can_write(kb, who):
            raise PermissionError("没有修改该知识库的权限")
        return kb

    # ---- Chroma ----

    def _collection(self, collection_name: str):
        return get_chroma().get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def _splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self._settings.rag_chunk_size,
            chunk_overlap=self._settings.rag_chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", ". ", " ", ""],
        )

    def _text_path(self, doc_id: str) -> Path:
        return self._settings.data_path("kb_texts", f"{doc_id}.txt")

    # ---- 知识库 CRUD ----

    async def create_kb(
        self,
        who: Principal,
        name: str,
        description: str = "",
        visibility: str = "private",
    ) -> Dict[str, Any]:
        name = name.strip()
        if not name:
            raise KnowledgeError("知识库名称不能为空")
        if visibility not in _VISIBILITIES:
            raise KnowledgeError(f"visibility 只能是 {_VISIBILITIES}")
        kb_id = new_id()
        async with session_scope() as session:
            exists = (
                await session.execute(
                    select(KnowledgeBase.id).where(
                        KnowledgeBase.owner_id == who.user_id, KnowledgeBase.name == name
                    )
                )
            ).first()
            if exists:
                raise KnowledgeError(f"知识库「{name}」已存在")
            kb = KnowledgeBase(
                id=kb_id,
                name=name,
                description=description,
                owner_id=who.user_id,
                visibility=visibility,
                collection_name=f"kb_{kb_id}",
            )
            session.add(kb)
            await session.flush()
            return kb_to_dict(kb, document_count=0)

    async def list_kbs(self, who: Principal) -> List[Dict[str, Any]]:
        async with session_scope() as session:
            stmt = select(KnowledgeBase).order_by(KnowledgeBase.created_at)
            if not who.is_admin:
                stmt = stmt.where(
                    or_(KnowledgeBase.owner_id == who.user_id, KnowledgeBase.visibility == "shared")
                )
            kbs = (await session.execute(stmt)).scalars().all()
            counts = dict(
                (
                    await session.execute(
                        select(Document.kb_id, func.count()).group_by(Document.kb_id)
                    )
                ).all()
            )
            return [kb_to_dict(kb, counts.get(kb.id, 0)) for kb in kbs]

    async def get_kb(self, who: Principal, kb_id: str) -> Dict[str, Any]:
        async with session_scope() as session:
            kb = await self._get_kb(session, kb_id, who)
            count = (
                await session.execute(
                    select(func.count()).select_from(Document).where(Document.kb_id == kb_id)
                )
            ).scalar_one()
            return kb_to_dict(kb, count)

    async def find_kb(self, who: Principal, name_or_id: str) -> Optional[KnowledgeBase]:
        """按名称或 ID 查找可访问的知识库（自己的优先）。"""
        key = name_or_id.strip()
        async with session_scope() as session:
            kb = await session.get(KnowledgeBase, key)
            if kb is not None and self._can_read(kb, who):
                return kb
            rows = (
                await session.execute(select(KnowledgeBase).where(KnowledgeBase.name == key))
            ).scalars().all()
            readable = [k for k in rows if self._can_read(k, who)]
            readable.sort(key=lambda k: k.owner_id != who.user_id)
            return readable[0] if readable else None

    async def update_kb(
        self,
        who: Principal,
        kb_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        visibility: Optional[str] = None,
    ) -> Dict[str, Any]:
        async with session_scope() as session:
            kb = await self._get_kb(session, kb_id, who, write=True)
            if name is not None and name.strip():
                kb.name = name.strip()
            if description is not None:
                kb.description = description
            if visibility is not None:
                if visibility not in _VISIBILITIES:
                    raise KnowledgeError(f"visibility 只能是 {_VISIBILITIES}")
                kb.visibility = visibility
            return kb_to_dict(kb)

    async def delete_kb(self, who: Principal, kb_id: str) -> None:
        async with session_scope() as session:
            kb = await self._get_kb(session, kb_id, who, write=True)
            collection_name = kb.collection_name
            doc_ids = (
                await session.execute(select(Document.id).where(Document.kb_id == kb_id))
            ).scalars().all()
            await session.delete(kb)
        try:
            await asyncio.to_thread(get_chroma().delete_collection, collection_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("删除知识库向量集合失败（可忽略）: %s", exc)
        for doc_id in doc_ids:
            self._text_path(doc_id).unlink(missing_ok=True)

    # ---- 文档 ----

    async def list_documents(self, who: Principal, kb_id: str) -> List[Dict[str, Any]]:
        async with session_scope() as session:
            await self._get_kb(session, kb_id, who)
            docs = (
                await session.execute(
                    select(Document).where(Document.kb_id == kb_id).order_by(Document.created_at)
                )
            ).scalars().all()
            return [doc_to_dict(d) for d in docs]

    async def ingest_file(
        self, who: Principal, kb_id: str, filename: str, data: bytes
    ) -> Dict[str, Any]:
        max_bytes = self._settings.upload_max_mb * 1024 * 1024
        if len(data) > max_bytes:
            raise KnowledgeError(f"文件超过 {self._settings.upload_max_mb}MB 上限")
        text = await asyncio.to_thread(extract_text, filename, data)
        return await self.ingest_text(who, kb_id, filename, text, source="upload")

    async def ingest_text(
        self,
        who: Principal,
        kb_id: str,
        title: str,
        text: str,
        *,
        source: str = "text",
        source_uri: str = "",
        replace_same_uri: bool = False,
    ) -> Dict[str, Any]:
        """切片、向量化并写入知识库，返回文档信息。"""
        text = (text or "").strip()
        if not text:
            raise KnowledgeError("文档内容为空，无法入库")

        async with session_scope() as session:
            kb = await self._get_kb(session, kb_id, who, write=True)
            collection_name = kb.collection_name
            kb_name = kb.name
            old_ids: List[str] = []
            if replace_same_uri and source_uri:
                old_ids = list(
                    (
                        await session.execute(
                            select(Document.id).where(
                                Document.kb_id == kb_id, Document.source_uri == source_uri
                            )
                        )
                    ).scalars().all()
                )
            doc = Document(
                kb_id=kb_id,
                filename=title[:500] or "untitled",
                source=source,
                source_uri=source_uri,
                status="processing",
                char_count=len(text),
            )
            session.add(doc)
            await session.flush()
            doc_id = doc.id

        for old_id in old_ids:
            await self.delete_document(who, old_id)

        try:
            chunk_count = await self._index_text(collection_name, kb_id, kb_name, doc_id, title, text)
            self._text_path(doc_id).write_text(text, encoding="utf-8")
            status, error = "ready", ""
        except Exception as exc:  # noqa: BLE001
            logger.error("知识库入库失败: doc=%s %s", doc_id, exc)
            chunk_count, status, error = 0, "failed", str(exc)[:1000]

        async with session_scope() as session:
            doc = await session.get(Document, doc_id)
            doc.chunk_count = chunk_count
            doc.status = status
            doc.error = error
            doc.updated_at = utcnow()
            result = doc_to_dict(doc)
        if status == "failed":
            raise KnowledgeError(f"入库失败: {error}")
        logger.info("知识库入库: kb=%s doc=%s 片段=%d", kb_name, title, chunk_count)
        return result

    async def ingest_url(self, who: Principal, kb_id: str, url: str) -> Dict[str, Any]:
        from app.capabilities.web.fetch import fetch_page

        page = await fetch_page(url)
        title = page.title or page.final_url
        return await self.ingest_text(
            who, kb_id, title, page.text,
            source="url", source_uri=page.final_url, replace_same_uri=True,
        )

    async def _index_text(
        self,
        collection_name: str,
        kb_id: str,
        kb_name: str,
        doc_id: str,
        title: str,
        text: str,
    ) -> int:
        chunks = [c for c in self._splitter().split_text(text) if c.strip()]
        if not chunks:
            return 0
        llm = get_llm_client()
        embeddings: List[List[float]] = []
        for i in range(0, len(chunks), _EMBED_BATCH):
            embeddings.extend(await llm.embed_batch(chunks[i:i + _EMBED_BATCH]))
        collection = await asyncio.to_thread(self._collection, collection_name)
        await asyncio.to_thread(
            collection.add,
            ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
            embeddings=embeddings,
            documents=chunks,
            metadatas=[
                {"doc_id": doc_id, "kb_id": kb_id, "kb": kb_name, "filename": title[:200], "chunk": i}
                for i in range(len(chunks))
            ],
        )
        return len(chunks)

    async def delete_document(self, who: Principal, doc_id: str) -> None:
        async with session_scope() as session:
            doc = await session.get(Document, doc_id)
            if doc is None:
                raise KnowledgeError("文档不存在")
            kb = await self._get_kb(session, doc.kb_id, who, write=True)
            collection_name = kb.collection_name
            await session.delete(doc)
        try:
            collection = await asyncio.to_thread(self._collection, collection_name)
            await asyncio.to_thread(collection.delete, where={"doc_id": doc_id})
        except Exception as exc:  # noqa: BLE001
            logger.warning("删除文档向量失败（可忽略）: %s", exc)
        self._text_path(doc_id).unlink(missing_ok=True)

    async def reindex_kb(self, who: Principal, kb_id: str) -> Dict[str, Any]:
        """用保存的纯文本重建整个知识库的向量（更换 embedding 模型后使用）。"""
        async with session_scope() as session:
            kb = await self._get_kb(session, kb_id, who, write=True)
            collection_name, kb_name = kb.collection_name, kb.name
            docs = (
                await session.execute(select(Document).where(Document.kb_id == kb_id))
            ).scalars().all()
            doc_infos = [(d.id, d.filename) for d in docs]
        try:
            await asyncio.to_thread(get_chroma().delete_collection, collection_name)
        except Exception:  # noqa: BLE001
            pass
        done, failed = 0, 0
        for doc_id, filename in doc_infos:
            path = self._text_path(doc_id)
            status, count, error = "failed", 0, "原始文本缺失，请重新上传"
            if path.exists():
                try:
                    text = path.read_text(encoding="utf-8")
                    count = await self._index_text(collection_name, kb_id, kb_name, doc_id, filename, text)
                    status, error = "ready", ""
                except Exception as exc:  # noqa: BLE001
                    error = str(exc)[:1000]
            async with session_scope() as session:
                doc = await session.get(Document, doc_id)
                if doc is not None:
                    doc.status, doc.chunk_count, doc.error = status, count, error
                    doc.updated_at = utcnow()
            if status == "ready":
                done += 1
            else:
                failed += 1
        return {"reindexed": done, "failed": failed}

    # ---- 检索 ----

    async def readable_kbs(self, who: Principal, kb_ids: Optional[Iterable[str]] = None) -> List[KnowledgeBase]:
        async with session_scope() as session:
            stmt = select(KnowledgeBase)
            if kb_ids is not None:
                stmt = stmt.where(KnowledgeBase.id.in_(list(kb_ids)))
            kbs = (await session.execute(stmt)).scalars().all()
            return [kb for kb in kbs if self._can_read(kb, who)]

    async def search(
        self,
        who: Principal,
        query: str,
        kb_ids: Optional[Iterable[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """在可访问的知识库中检索，返回 [{text, score, kb, kb_id, filename, doc_id, chunk}]。"""
        if not query.strip():
            return []
        kbs = await self.readable_kbs(who, kb_ids)
        if not kbs:
            return []
        k = top_k or self._settings.rag_top_k
        query_embedding = await get_llm_client().embed(query)

        def _query_all() -> List[Dict[str, Any]]:
            results: List[Dict[str, Any]] = []
            for kb in kbs:
                try:
                    collection = self._collection(kb.collection_name)
                    count = collection.count()
                    if count == 0:
                        continue
                    res = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=min(k, count),
                        include=["documents", "metadatas", "distances"],
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("知识库 %s 检索失败: %s", kb.name, exc)
                    continue
                docs = (res.get("documents") or [[]])[0]
                metas = (res.get("metadatas") or [[]])[0]
                dists = (res.get("distances") or [[]])[0]
                for text, meta, dist in zip(docs, metas, dists):
                    meta = meta or {}
                    results.append({
                        "text": text,
                        "score": max(0.0, 1.0 - float(dist)),
                        "kb": kb.name,
                        "kb_id": kb.id,
                        "filename": meta.get("filename", ""),
                        "doc_id": meta.get("doc_id", ""),
                        "chunk": meta.get("chunk", 0),
                    })
            return results

        results = await asyncio.to_thread(_query_all)
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:k]

    # ---- 供 Agent / 工具使用（基于请求上下文中的当前用户） ----

    async def search_for_current_user(
        self, query: str, kb_name: Optional[str] = None, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        who = current_principal()
        kb_ids: Optional[List[str]] = None
        if kb_name:
            kb = await self.find_kb(who, kb_name)
            if kb is None:
                raise KnowledgeError(f"找不到知识库「{kb_name}」")
            kb_ids = [kb.id]
        return await self.search(who, query, kb_ids=kb_ids, top_k=top_k)

    async def get_or_create_kb_for_current_user(self, name: str) -> KnowledgeBase:
        who = current_principal()
        kb = await self.find_kb(who, name)
        if kb is not None:
            if not self._can_write(kb, who):
                raise PermissionError(f"没有写入知识库「{name}」的权限")
            return kb
        created = await self.create_kb(who, name, description="由助手自动创建")
        found = await self.find_kb(who, created["id"])
        assert found is not None
        return found

    async def ingest_url_for_current_user(self, kb_id: str, url: str) -> Dict[str, Any]:
        return await self.ingest_url(current_principal(), kb_id, url)


_service: Optional[KnowledgeService] = None


def get_knowledge_service() -> KnowledgeService:
    global _service
    if _service is None:
        _service = KnowledgeService()
    return _service
