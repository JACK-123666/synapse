"""鉴权 API：登录、当前用户、API Key 管理。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.deps import CurrentUser, get_current_user, require_admin
from app.services import users as user_service

router = APIRouter(prefix="/auth", tags=["鉴权"])


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class PasswordChangeRequest(BaseModel):
    old_password: str
    new_password: str = Field(..., min_length=6)


class ApiKeyCreateRequest(BaseModel):
    name: str = Field(default="default", max_length=64)


@router.get("/config", summary="鉴权配置（公开）")
async def auth_config() -> Dict[str, Any]:
    """前端据此决定是否显示登录框。"""
    return {"auth_enabled": get_settings().auth_enabled}


@router.post("/login", summary="用户名密码登录，返回 JWT")
async def login(req: LoginRequest) -> Dict[str, Any]:
    result = await user_service.login(req.username, req.password)
    if result is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
    return result


@router.get("/me", summary="当前用户")
async def me(user: CurrentUser = Depends(get_current_user)) -> Dict[str, Any]:
    return {"id": user.id, "username": user.username, "role": user.role}


@router.post("/password", summary="修改自己的密码")
async def change_password(
    req: PasswordChangeRequest,
    user: CurrentUser = Depends(get_current_user),
) -> Dict[str, Any]:
    if await user_service.authenticate(user.username, req.old_password) is None:
        raise HTTPException(status_code=400, detail="原密码错误")
    try:
        await user_service.update_user(user.id, password=req.new_password)
    except user_service.UserError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@router.get("/api-keys", summary="我的 API Key 列表")
async def list_api_keys(user: CurrentUser = Depends(get_current_user)) -> List[Dict[str, Any]]:
    return await user_service.list_api_keys(user.id)


@router.post("/api-keys", summary="创建 API Key（明文只返回这一次）")
async def create_api_key(
    req: ApiKeyCreateRequest,
    user: CurrentUser = Depends(get_current_user),
) -> Dict[str, Any]:
    return await user_service.create_api_key(user.id, req.name)


@router.delete("/api-keys/{key_id}", summary="吊销 API Key")
async def revoke_api_key(
    key_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> Dict[str, Any]:
    try:
        await user_service.revoke_api_key(user.id, key_id, is_admin=user.is_admin)
    except user_service.UserError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True}


# ---- 用户管理（管理员） ----

users_router = APIRouter(prefix="/users", tags=["用户管理"])


class UserCreateRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=6)
    role: str = Field(default="user", description="admin / user")


class UserUpdateRequest(BaseModel):
    role: Optional[str] = None
    is_active: Optional[bool] = None
    password: Optional[str] = Field(default=None, min_length=6)


_admin_only = require_admin


@users_router.get("", summary="用户列表")
async def list_users(_: CurrentUser = Depends(_admin_only)) -> List[Dict[str, Any]]:
    return await user_service.list_users()


@users_router.post("", summary="创建用户")
async def create_user(
    req: UserCreateRequest, _: CurrentUser = Depends(_admin_only)
) -> Dict[str, Any]:
    try:
        return await user_service.create_user(req.username, req.password, req.role)
    except user_service.UserError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@users_router.patch("/{user_id}", summary="修改用户（角色 / 启停 / 重置密码）")
async def update_user(
    user_id: str, req: UserUpdateRequest, _: CurrentUser = Depends(_admin_only)
) -> Dict[str, Any]:
    try:
        return await user_service.update_user(
            user_id, role=req.role, is_active=req.is_active, password=req.password
        )
    except user_service.UserError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@users_router.delete("/{user_id}", summary="删除用户")
async def delete_user(user_id: str, _: CurrentUser = Depends(_admin_only)) -> Dict[str, Any]:
    try:
        await user_service.delete_user(user_id)
    except user_service.UserError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}
