"""知识库（RAG）API。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.capabilities.knowledge.loaders import UnsupportedFileType
from app.capabilities.knowledge.service import (
    KnowledgeError,
    Principal,
    get_knowledge_service,
)
from app.capabilities.web.fetch import FetchError
from app.core.deps import CurrentUser, get_current_user

router = APIRouter(prefix="/knowledge-bases", tags=["知识库"])


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    description: str = ""
    visibility: str = Field(default="private", description="private 仅自己 / shared 所有用户可检索")


class KnowledgeBaseUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    visibility: Optional[str] = None


class TextIngest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    text: str = Field(..., min_length=1)


class UrlIngest(BaseModel):
    url: str = Field(..., min_length=8)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    kb_ids: Optional[List[str]] = Field(default=None, description="限定知识库；留空检索全部可访问的")
    top_k: int = Field(default=5, ge=1, le=50)


def _who(user: CurrentUser) -> Principal:
    return Principal(user_id=user.id, is_admin=user.is_admin)


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, PermissionError):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, KnowledgeError) and "不存在" in str(exc):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


_ERRORS = (KnowledgeError, PermissionError, UnsupportedFileType, FetchError)


@router.get("", summary="知识库列表（自己的 + 共享的）")
async def list_kbs(user: CurrentUser = Depends(get_current_user)) -> List[Dict[str, Any]]:
    return await get_knowledge_service().list_kbs(_who(user))


@router.post("", summary="创建知识库")
async def create_kb(
    req: KnowledgeBaseCreate, user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    try:
        return await get_knowledge_service().create_kb(
            _who(user), req.name, req.description, req.visibility
        )
    except _ERRORS as exc:
        raise _http_error(exc) from exc


@router.post("/search", summary="检索知识库")
async def search(
    req: SearchRequest, user: CurrentUser = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    return await get_knowledge_service().search(
        _who(user), req.query, kb_ids=req.kb_ids, top_k=req.top_k
    )


@router.get("/{kb_id}", summary="知识库详情")
async def get_kb(kb_id: str, user: CurrentUser = Depends(get_current_user)) -> Dict[str, Any]:
    try:
        return await get_knowledge_service().get_kb(_who(user), kb_id)
    except _ERRORS as exc:
        raise _http_error(exc) from exc


@router.patch("/{kb_id}", summary="修改知识库")
async def update_kb(
    kb_id: str, req: KnowledgeBaseUpdate, user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    try:
        return await get_knowledge_service().update_kb(
            _who(user), kb_id,
            name=req.name, description=req.description, visibility=req.visibility,
        )
    except _ERRORS as exc:
        raise _http_error(exc) from exc


@router.delete("/{kb_id}", summary="删除知识库（含全部文档与向量）")
async def delete_kb(kb_id: str, user: CurrentUser = Depends(get_current_user)) -> Dict[str, Any]:
    try:
        await get_knowledge_service().delete_kb(_who(user), kb_id)
    except _ERRORS as exc:
        raise _http_error(exc) from exc
    return {"ok": True}


@router.get("/{kb_id}/documents", summary="文档列表")
async def list_documents(
    kb_id: str, user: CurrentUser = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    try:
        return await get_knowledge_service().list_documents(_who(user), kb_id)
    except _ERRORS as exc:
        raise _http_error(exc) from exc


@router.post("/{kb_id}/documents", summary="上传文档（txt/md/pdf/docx/html/代码文件）")
async def upload_document(
    kb_id: str,
    file: UploadFile = File(...),
    user: CurrentUser = Depends(get_current_user),
) -> Dict[str, Any]:
    data = await file.read()
    try:
        return await get_knowledge_service().ingest_file(
            _who(user), kb_id, file.filename or "upload.txt", data
        )
    except _ERRORS as exc:
        raise _http_error(exc) from exc


@router.post("/{kb_id}/texts", summary="直接写入一段文本")
async def ingest_text(
    kb_id: str, req: TextIngest, user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    try:
        return await get_knowledge_service().ingest_text(_who(user), kb_id, req.title, req.text)
    except _ERRORS as exc:
        raise _http_error(exc) from exc


@router.post("/{kb_id}/urls", summary="抓取网页并入库")
async def ingest_url(
    kb_id: str, req: UrlIngest, user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    try:
        return await get_knowledge_service().ingest_url(_who(user), kb_id, req.url)
    except _ERRORS as exc:
        raise _http_error(exc) from exc


@router.delete("/{kb_id}/documents/{doc_id}", summary="删除文档")
async def delete_document(
    kb_id: str, doc_id: str, user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    try:
        await get_knowledge_service().delete_document(_who(user), doc_id)
    except _ERRORS as exc:
        raise _http_error(exc) from exc
    return {"ok": True}


@router.post("/{kb_id}/reindex", summary="重建向量索引（更换 embedding 模型后使用）")
async def reindex(kb_id: str, user: CurrentUser = Depends(get_current_user)) -> Dict[str, Any]:
    try:
        return await get_knowledge_service().reindex_kb(_who(user), kb_id)
    except _ERRORS as exc:
        raise _http_error(exc) from exc
