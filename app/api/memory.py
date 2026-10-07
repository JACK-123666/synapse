"""记忆 API：长期记忆摘要、会话短期记忆、用户画像。

开启鉴权时只能访问自己的记忆；关闭鉴权时可用 user_id 参数筛选（与 /chat 的 user_id 一致）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.deps import CurrentUser, get_current_user
from app.memory.archive import get_long_term_memory
from app.memory.profile import get_user_profile_manager
from app.memory.recent import get_short_term_memory
from app.services.chat import session_key_for

router = APIRouter(prefix="/memories", tags=["记忆"])


def _memory_user(user: CurrentUser, user_id: Optional[str]) -> Optional[str]:
    return user.id if get_settings().auth_enabled else user_id


@router.get("", summary="列出长期记忆摘要（时间倒序）")
async def list_memories(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user_id: Optional[str] = Query(None, description="关闭鉴权时可按 user_id 筛选"),
    user: CurrentUser = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """按时间倒序列出长期记忆摘要。"""
    return await get_long_term_memory().list_summaries(
        user_id=_memory_user(user, user_id), limit=limit, offset=offset
    )


@router.get("/search", summary="语义检索长期记忆")
async def search_memories(
    q: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=50),
    user_id: Optional[str] = Query(None),
    user: CurrentUser = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """在长期记忆摘要中做语义检索。"""
    return await get_long_term_memory().recall(
        query_text=q, top_k=top_k, user_id=_memory_user(user, user_id)
    )


@router.get("/sessions", summary="列出会话（历史对话）")
async def list_sessions(
    limit: int = Query(50, ge=1, le=200),
    user: CurrentUser = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """按最近活跃倒序列出会话，供前端渲染历史对话列表。

    合并两个来源，缺一不可：
    - 短期记忆里还活着的会话 —— 有完整消息，可以直接恢复对话；
    - 只剩长期摘要的会话 —— 短期已过期（TTL 24h）或服务重启过，
      但压缩出来的摘要还在，仍应出现在列表里。
    """
    settings = get_settings()
    prefix = f"{user.id}:" if settings.auth_enabled else ""

    def strip(sid: str) -> str:
        """去掉开启鉴权时加上的用户前缀，返回前端使用的原始会话 ID。"""
        return sid[len(prefix):] if prefix and sid.startswith(prefix) else sid

    sessions: Dict[str, Dict[str, Any]] = {}

    for item in await get_short_term_memory().list_sessions(limit=200):
        raw_sid = item["session_id"]
        if prefix and not raw_sid.startswith(prefix):
            continue
        item["session_id"] = strip(raw_sid)
        item["has_summary"] = False
        sessions[item["session_id"]] = item

    for summary in await get_long_term_memory().list_summaries(
        user_id=user.id if settings.auth_enabled else None, limit=200
    ):
        meta = summary.get("metadata") or {}
        raw_sid = meta.get("session_id")
        if not raw_sid:
            continue
        sid = strip(raw_sid)
        ts = float(meta.get("timestamp") or 0)
        entry = sessions.get(sid)
        if entry is None:
            sessions[sid] = {
                "session_id": sid,
                "message_count": 0,
                "updated_at": ts,
                "preview": str(summary.get("text") or "")[:80],
                "has_summary": True,
            }
        else:
            entry["has_summary"] = True
            entry["updated_at"] = max(float(entry.get("updated_at") or 0), ts)

    ordered = sorted(sessions.values(), key=lambda s: s.get("updated_at") or 0, reverse=True)
    return ordered[:limit]


@router.get("/sessions/{session_id}", summary="查看会话短期记忆")
async def get_session(
    session_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """查看某个会话的短期记忆（最近若干条消息）。"""
    return await get_short_term_memory().get_messages(session_key_for(user.id, session_id))


@router.delete("/sessions/{session_id}", summary="删除会话")
async def clear_session(
    session_id: str,
    purge: bool = Query(False, description="同时删除该会话产生的长期记忆摘要"),
    user: CurrentUser = Depends(get_current_user),
) -> Dict[str, Any]:
    """删除一个会话。

    默认只清短期记忆（会话从列表里消失，但压缩出的长期摘要仍留在记忆里）；
    purge=true 时连同它的长期摘要一起删除，用于"彻底删掉这段对话"。
    """
    key = session_key_for(user.id, session_id)
    await get_short_term_memory().clear(key)

    removed = 0
    if purge:
        long_term = get_long_term_memory()
        owner = user.id if get_settings().auth_enabled else None
        for summary in await long_term.list_summaries(user_id=owner, limit=500):
            if (summary.get("metadata") or {}).get("session_id") == key:
                if await long_term.delete_summary(summary["id"], user_id=owner):
                    removed += 1

    return {"ok": True, "summaries_deleted": removed}


@router.get("/profile", summary="查看用户画像")
async def get_profile(
    user_id: Optional[str] = Query(None),
    user: CurrentUser = Depends(get_current_user),
) -> Dict[str, Any]:
    """查看用户画像（偏好、高频术语、交互次数）。"""
    target = _memory_user(user, user_id)
    if not target:
        raise HTTPException(status_code=400, detail="关闭鉴权时请提供 user_id")
    return await get_user_profile_manager().get_profile(target)


class ProfileUpdate(BaseModel):
    """用户画像的可编辑字段。"""

    preferences: Optional[List[str]] = Field(default=None, description="偏好标签，整体替换")
    custom: Optional[Dict[str, Any]] = Field(default=None, description="自定义字段，整体替换")


@router.put("/profile", summary="更新用户画像")
async def put_profile(
    payload: ProfileUpdate,
    user_id: Optional[str] = Query(None),
    user: CurrentUser = Depends(get_current_user),
) -> Dict[str, Any]:
    """整体替换画像中的指定字段；没传的字段保持不动。"""
    target = _memory_user(user, user_id)
    if not target:
        raise HTTPException(status_code=400, detail="关闭鉴权时请提供 user_id")
    manager = get_user_profile_manager()
    if payload.preferences is not None:
        await manager.set_preferences(target, payload.preferences)
    if payload.custom is not None:
        await manager.set_custom(target, payload.custom)
    return await manager.get_profile(target)


@router.delete("/profile", summary="清除用户画像")
async def delete_profile(
    user_id: Optional[str] = Query(None),
    user: CurrentUser = Depends(get_current_user),
) -> Dict[str, Any]:
    """删除整个用户画像（偏好、术语、计数全部清空）。"""
    target = _memory_user(user, user_id)
    if not target:
        raise HTTPException(status_code=400, detail="关闭鉴权时请提供 user_id")
    await get_user_profile_manager().clear(target)
    return {"ok": True}


# 注意：这条通配路由必须放在所有静态 DELETE 路径之后。
# FastAPI 按注册顺序匹配，/{record_id} 会把 /memories/profile 也吃掉。
@router.delete("/{record_id}", summary="删除一条长期记忆摘要")
async def delete_memory(
    record_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> Dict[str, Any]:
    """删除一条长期记忆摘要。"""
    owner = user.id if get_settings().auth_enabled and not user.is_admin else None
    if not await get_long_term_memory().delete_summary(record_id, user_id=owner):
        raise HTTPException(status_code=404, detail="记忆不存在")
    return {"ok": True}
