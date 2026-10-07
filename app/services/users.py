"""用户与 API Key 管理。"""

from __future__ import annotations

import logging
import secrets
from typing import Dict, List, Optional

from sqlalchemy import func, select

from app.config import get_settings
from app.core.context import LOCAL_ADMIN_ID
from app.core.db import session_scope
from app.core.security import (
    create_access_token,
    generate_api_key,
    hash_password,
    verify_password,
)
from app.models import ApiKey, User

logger = logging.getLogger(__name__)

VALID_ROLES = ("admin", "user")


class UserError(ValueError):
    """用户操作错误（参数非法、重名、权限不足等）。"""


def user_to_dict(user: User) -> Dict[str, object]:
    """把用户对象转成对外字典（不含密码哈希）。"""
    return {
        "id": user.id,
        "username": user.username,
        "role": user.role,
        "is_active": user.is_active,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


async def bootstrap_admin() -> None:
    """确保初始管理员存在（固定 ID，关闭鉴权时即本地管理员）。

    - 首次启动：ADMIN_PASSWORD 留空则随机生成并打印到日志
    - 之后启动：若配置了 ADMIN_PASSWORD 且与当前密码不同，则重置为该密码
    """
    settings = get_settings()
    async with session_scope() as session:
        admin = await session.get(User, LOCAL_ADMIN_ID)
        if admin is None:
            password = settings.admin_password or secrets.token_urlsafe(12)
            session.add(
                User(
                    id=LOCAL_ADMIN_ID,
                    username=settings.admin_username,
                    password_hash=hash_password(password),
                    role="admin",
                )
            )
            if not settings.admin_password:
                logger.warning(
                    "已创建初始管理员 '%s'，随机密码: %s （请登录后修改，或设置 ADMIN_PASSWORD 后重启）",
                    settings.admin_username, password,
                )
            else:
                logger.info("已创建初始管理员 '%s'", settings.admin_username)
        elif settings.admin_password and not verify_password(
            settings.admin_password, admin.password_hash
        ):
            admin.password_hash = hash_password(settings.admin_password)
            logger.info("已按 ADMIN_PASSWORD 重置管理员 '%s' 的密码", admin.username)


async def authenticate(username: str, password: str) -> Optional[User]:
    """校验用户名与密码。成功返回用户对象，失败返回 None。"""
    async with session_scope() as session:
        user = (
            await session.execute(select(User).where(User.username == username))
        ).scalar_one_or_none()
        if user is None or not user.is_active:
            return None
        if not verify_password(password, user.password_hash):
            return None
        return user


async def login(username: str, password: str) -> Optional[Dict[str, object]]:
    """登录：校验通过后签发 JWT 并返回用户信息。"""
    user = await authenticate(username, password)
    if user is None:
        return None
    token, expires_in = create_access_token(user.id, user.username, user.role)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": expires_in,
        "user": user_to_dict(user),
    }


async def list_users() -> List[Dict[str, object]]:
    """列出全部用户。"""
    async with session_scope() as session:
        rows = (await session.execute(select(User).order_by(User.created_at))).scalars().all()
        return [user_to_dict(u) for u in rows]


async def get_user(user_id: str) -> Optional[Dict[str, object]]:
    """按 ID 查询用户。"""
    async with session_scope() as session:
        user = await session.get(User, user_id)
        return user_to_dict(user) if user else None


async def create_user(username: str, password: str, role: str = "user") -> Dict[str, object]:
    """创建用户；用户名重复时抛 UserError。"""
    username = username.strip()
    if not username or len(password) < 6:
        raise UserError("用户名不能为空，密码至少 6 位")
    if role not in VALID_ROLES:
        raise UserError(f"角色只能是 {VALID_ROLES}")
    async with session_scope() as session:
        exists = (
            await session.execute(select(User.id).where(User.username == username))
        ).first()
        if exists:
            raise UserError(f"用户名 '{username}' 已存在")
        user = User(username=username, password_hash=hash_password(password), role=role)
        session.add(user)
        await session.flush()
        return user_to_dict(user)


async def update_user(
    user_id: str,
    *,
    role: Optional[str] = None,
    is_active: Optional[bool] = None,
    password: Optional[str] = None,
) -> Dict[str, object]:
    """修改用户的角色、启用状态或密码。"""
    async with session_scope() as session:
        user = await session.get(User, user_id)
        if user is None:
            raise UserError("用户不存在")
        if role is not None:
            if role not in VALID_ROLES:
                raise UserError(f"角色只能是 {VALID_ROLES}")
            if user_id == LOCAL_ADMIN_ID and role != "admin":
                raise UserError("初始管理员不能降级")
            user.role = role
        if is_active is not None:
            if user_id == LOCAL_ADMIN_ID and not is_active:
                raise UserError("初始管理员不能停用")
            user.is_active = is_active
        if password is not None:
            if len(password) < 6:
                raise UserError("密码至少 6 位")
            user.password_hash = hash_password(password)
        return user_to_dict(user)


async def delete_user(user_id: str) -> None:
    """删除用户；不允许删除自己，避免把管理员删空。"""
    if user_id == LOCAL_ADMIN_ID:
        raise UserError("初始管理员不能删除")
    async with session_scope() as session:
        user = await session.get(User, user_id)
        if user is None:
            raise UserError("用户不存在")
        admin_count = (
            await session.execute(select(func.count()).select_from(User).where(User.role == "admin"))
        ).scalar_one()
        if user.role == "admin" and admin_count <= 1:
            raise UserError("至少保留一个管理员")
        await session.delete(user)


# ---- API Key ----


def _key_to_dict(key: ApiKey) -> Dict[str, object]:
    return {
        "id": key.id,
        "name": key.name,
        "prefix": key.prefix,
        "revoked": key.revoked,
        "created_at": key.created_at.isoformat() if key.created_at else None,
        "last_used_at": key.last_used_at.isoformat() if key.last_used_at else None,
    }


async def create_api_key(user_id: str, name: str = "default") -> Dict[str, object]:
    """创建 API Key；明文只在返回值中出现这一次。"""
    plain, prefix, key_hash = generate_api_key()
    async with session_scope() as session:
        key = ApiKey(user_id=user_id, name=name or "default", prefix=prefix, key_hash=key_hash)
        session.add(key)
        await session.flush()
        data = _key_to_dict(key)
    data["api_key"] = plain
    return data


async def list_api_keys(user_id: str) -> List[Dict[str, object]]:
    """列出某个用户的 API Key。"""
    async with session_scope() as session:
        rows = (
            await session.execute(
                select(ApiKey).where(ApiKey.user_id == user_id).order_by(ApiKey.created_at)
            )
        ).scalars().all()
        return [_key_to_dict(k) for k in rows]


async def revoke_api_key(user_id: str, key_id: str, is_admin: bool = False) -> None:
    """吊销 API Key；非管理员只能吊销自己的。"""
    async with session_scope() as session:
        key = await session.get(ApiKey, key_id)
        if key is None or (key.user_id != user_id and not is_admin):
            raise UserError("API Key 不存在")
        key.revoked = True
