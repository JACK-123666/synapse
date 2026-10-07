"""代码仓库 API：仓库连接管理、提交 / 变更日志、索引入知识库、写操作确认。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.capabilities.knowledge.service import KnowledgeError, Principal
from app.capabilities.repo.providers import RepoError
from app.capabilities.repo.service import get_repo_service
from app.core.deps import CurrentUser, get_current_user

router = APIRouter(prefix="/repos", tags=["代码仓库"])


class RepoCreate(BaseModel):
    """新增仓库连接的请求体；token 会加密后存储。"""
    name: str = Field(..., min_length=1, max_length=128, description="连接名称（对话中用它指代仓库）")
    provider: str = Field(..., description="github / gitlab / local")
    repo: str = Field(..., description="github/gitlab: owner/repo；local: 本地路径或 clone 地址")
    token: str = Field(default="", description="访问令牌（加密存储，可选）")
    base_url: str = Field(default="", description="自建 GitHub Enterprise / GitLab 的 API 地址")
    default_branch: str = ""
    allow_write: bool = Field(default=False, description="是否允许写操作（仅管理员可开启）")


class RepoUpdate(BaseModel):
    """修改仓库连接的请求体。"""
    name: Optional[str] = None
    repo: Optional[str] = None
    token: Optional[str] = None
    base_url: Optional[str] = None
    default_branch: Optional[str] = None
    allow_write: Optional[bool] = None


class IndexRequest(BaseModel):
    """把仓库代码索引进知识库的请求体。"""
    knowledge_base: str = Field(..., min_length=1, description="目标知识库名称（不存在时自动创建）")
    path_prefix: str = ""
    ref: str = ""
    max_files: Optional[int] = Field(default=None, ge=1, le=5000)


def _who(user: CurrentUser) -> Principal:
    return Principal(user_id=user.id, is_admin=user.is_admin)


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, PermissionError):
        return HTTPException(status_code=403, detail=str(exc))
    if "不存在" in str(exc) or "找不到" in str(exc):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


_ERRORS = (RepoError, PermissionError, KnowledgeError)


# ---- 写操作确认（固定路径需放在 /{conn_id} 之前） ----


@router.get("/actions", summary="待确认 / 已处理的写操作")
async def list_actions(
    status: Optional[str] = Query(None, description="pending / done / rejected / failed"),
    user: CurrentUser = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """列出等待用户确认的仓库写操作。"""
    return await get_repo_service().list_actions(_who(user), status)


@router.post("/actions/{action_id}/confirm", summary="确认并执行写操作")
async def confirm_action(action_id: str, user: CurrentUser = Depends(get_current_user)) -> Dict[str, Any]:
    """确认并执行一个待确认的写操作。"""
    try:
        return await get_repo_service().confirm_action(_who(user), action_id)
    except _ERRORS as exc:
        raise _http_error(exc) from exc


@router.post("/actions/{action_id}/reject", summary="拒绝写操作")
async def reject_action(action_id: str, user: CurrentUser = Depends(get_current_user)) -> Dict[str, Any]:
    """拒绝一个待确认的写操作。"""
    try:
        return await get_repo_service().reject_action(_who(user), action_id)
    except _ERRORS as exc:
        raise _http_error(exc) from exc


# ---- 仓库连接 ----


@router.get("", summary="仓库连接列表")
async def list_repos(user: CurrentUser = Depends(get_current_user)) -> List[Dict[str, Any]]:
    """列出当前用户连接的代码仓库。"""
    return await get_repo_service().list_connections(_who(user))


@router.post("", summary="添加仓库连接")
async def create_repo(req: RepoCreate, user: CurrentUser = Depends(get_current_user)) -> Dict[str, Any]:
    """新增一个仓库连接（GitHub / GitLab / 本地 Git）。"""
    try:
        return await get_repo_service().create_connection(_who(user), **req.model_dump())
    except _ERRORS as exc:
        raise _http_error(exc) from exc


@router.patch("/{conn_id}", summary="修改仓库连接")
async def update_repo(
    conn_id: str, req: RepoUpdate, user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """修改仓库连接的地址、令牌或写权限。"""
    try:
        return await get_repo_service().update_connection(
            _who(user), conn_id, **req.model_dump(exclude_none=True)
        )
    except _ERRORS as exc:
        raise _http_error(exc) from exc


@router.delete("/{conn_id}", summary="删除仓库连接")
async def delete_repo(conn_id: str, user: CurrentUser = Depends(get_current_user)) -> Dict[str, Any]:
    """删除仓库连接。"""
    try:
        await get_repo_service().delete_connection(_who(user), conn_id)
    except _ERRORS as exc:
        raise _http_error(exc) from exc
    return {"ok": True}


@router.post("/{conn_id}/test", summary="测试连接（读取最近一条提交）")
async def test_repo(conn_id: str, user: CurrentUser = Depends(get_current_user)) -> Dict[str, Any]:
    """测试仓库连接是否可用。"""
    try:
        conn, provider = await get_repo_service().provider(_who(user), conn_id)
        commits = await provider.list_commits(branch=conn.default_branch, limit=1)
    except _ERRORS as exc:
        raise _http_error(exc) from exc
    return {"ok": True, "latest_commit": commits[0] if commits else None}


@router.get("/{conn_id}/commits", summary="提交记录")
async def list_commits(
    conn_id: str,
    limit: int = Query(20, ge=1, le=100),
    branch: str = "",
    since: str = "",
    user: CurrentUser = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """查看仓库的提交记录。"""
    try:
        conn, provider = await get_repo_service().provider(_who(user), conn_id)
        return await provider.list_commits(branch=branch or conn.default_branch, limit=limit, since=since)
    except _ERRORS as exc:
        raise _http_error(exc) from exc


@router.get("/{conn_id}/changelog", summary="生成变更日志（Markdown）")
async def changelog(
    conn_id: str,
    since: str = "",
    limit: int = Query(50, ge=1, le=200),
    branch: str = "",
    user: CurrentUser = Depends(get_current_user),
) -> Dict[str, Any]:
    """根据提交记录生成分组变更日志。"""
    try:
        text = await get_repo_service().changelog(
            _who(user), conn_id, since=since, limit=limit, branch=branch
        )
    except _ERRORS as exc:
        raise _http_error(exc) from exc
    return {"markdown": text}


@router.post("/{conn_id}/index", summary="把仓库代码索引进知识库")
async def index_repo(
    conn_id: str, req: IndexRequest, user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """把仓库文件索引进知识库，供代码问答使用。"""
    try:
        return await get_repo_service().index_to_knowledge(
            _who(user), conn_id, req.knowledge_base,
            path_prefix=req.path_prefix, ref=req.ref, max_files=req.max_files,
        )
    except _ERRORS as exc:
        raise _http_error(exc) from exc
