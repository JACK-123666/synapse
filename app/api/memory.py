"""记忆 API：长期记忆摘要、会话短期记忆、用户画像。

开启鉴权时只能访问自己的记忆；关闭鉴权时可用 user_id 参数筛选（与 /chat 的 user_id 一致）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

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
    return await get_long_term_memory().recall(
        query_text=q, top_k=top_k, user_id=_memory_user(user, user_id)
    )


@router.delete("/{record_id}", summary="删除一条长期记忆摘要")
async def delete_memory(
    record_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> Dict[str, Any]:
    owner = user.id if get_settings().auth_enabled and not user.is_admin else None
    if not await get_long_term_memory().delete_summary(record_id, user_id=owner):
        raise HTTPException(status_code=404, detail="记忆不存在")
    return {"ok": True}


@router.get("/sessions/{session_id}", summary="查看会话短期记忆")
async def get_session(
    session_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    return await get_short_term_memory().get_messages(session_key_for(user.id, session_id))


@router.delete("/sessions/{session_id}", summary="清空会话短期记忆")
async def clear_session(
    session_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> Dict[str, Any]:
    await get_short_term_memory().clear(session_key_for(user.id, session_id))
    return {"ok": True}


@router.get("/profile", summary="查看用户画像")
async def get_profile(
    user_id: Optional[str] = Query(None),
    user: CurrentUser = Depends(get_current_user),
) -> Dict[str, Any]:
    target = _memory_user(user, user_id)
    if not target:
        raise HTTPException(status_code=400, detail="关闭鉴权时请提供 user_id")
    return await get_user_profile_manager().get_profile(target)
