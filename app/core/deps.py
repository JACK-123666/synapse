"""FastAPI 依赖：当前用户解析与权限校验。

鉴权方式（AUTH_ENABLED=true 时）：
- Authorization: Bearer <JWT>        登录接口签发
- Authorization: Bearer <API Key>    以 syn- 开头
- X-API-Key: <API Key>

AUTH_ENABLED=false（个人部署）时，所有请求视为本地管理员。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import jwt
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select

from app.config import get_settings
from app.core.context import LOCAL_ADMIN_ID
from app.core.db import session_scope
from app.core.security import decode_access_token, hash_api_key, looks_like_api_key

logger = logging.getLogger(__name__)


@dataclass
class CurrentUser:
    """已认证的当前用户。"""

    id: str
    username: str
    role: str

    @property
    def is_admin(self) -> bool:
        """该用户是否管理员。"""
        return self.role == "admin"


def local_admin() -> CurrentUser:
    """构造本地管理员身份；关闭鉴权时所有请求都使用它。"""
    return CurrentUser(id=LOCAL_ADMIN_ID, username=get_settings().admin_username, role="admin")


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


async def _user_from_api_key(plain: str) -> CurrentUser:
    from app.models import ApiKey, User, utcnow

    async with session_scope() as session:
        row = (
            await session.execute(
                select(ApiKey, User)
                .join(User, User.id == ApiKey.user_id)
                .where(ApiKey.key_hash == hash_api_key(plain))
            )
        ).first()
        if row is None:
            raise _unauthorized("API Key 无效")
        api_key, user = row
        if api_key.revoked or not user.is_active:
            raise _unauthorized("API Key 已吊销或用户已停用")
        api_key.last_used_at = utcnow()
        return CurrentUser(id=user.id, username=user.username, role=user.role)


async def _user_from_jwt(token: str) -> CurrentUser:
    from app.models import User

    try:
        payload = decode_access_token(token)
    except jwt.ExpiredSignatureError as exc:
        raise _unauthorized("登录已过期，请重新登录") from exc
    except jwt.PyJWTError as exc:
        raise _unauthorized("令牌无效") from exc

    async with session_scope() as session:
        user = await session.get(User, payload.get("sub"))
        if user is None or not user.is_active:
            raise _unauthorized("用户不存在或已停用")
        return CurrentUser(id=user.id, username=user.username, role=user.role)


async def get_current_user(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> CurrentUser:
    """解析当前用户；鉴权关闭时返回本地管理员。"""
    if not get_settings().auth_enabled:
        return local_admin()

    token: Optional[str] = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()

    if x_api_key:
        return await _user_from_api_key(x_api_key.strip())
    if token and looks_like_api_key(token):
        return await _user_from_api_key(token)
    if token:
        return await _user_from_jwt(token)
    raise _unauthorized("未登录：请提供 Bearer Token 或 X-API-Key")


async def require_admin(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """仅允许管理员。"""
    if not user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="需要管理员权限")
    return user
