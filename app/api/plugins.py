"""插件 API：本地插件启停 / 重载、MCP 服务管理、工具列表与角色白名单。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.deps import CurrentUser, get_current_user, require_admin
from app.plugins.manager import PluginError, get_plugin_manager
from app.plugins.mcp import TRANSPORTS, McpError, get_mcp_manager
from app.services.policies import set_tool_policy
from app.tools.registry import get_tool_registry

router = APIRouter(prefix="/plugins", tags=["插件"])


class McpServerCreate(BaseModel):
    name: str = Field(..., description="服务名（字母、数字、下划线）")
    transport: str = Field(default="streamable_http", description=f"{' / '.join(TRANSPORTS)}")
    config: Dict[str, Any] = Field(
        ...,
        description='stdio: {"command": "...", "args": [...], "env": {...}}；'
        'http: {"url": "...", "headers": {...}}；可选 "intent": {"description", "keywords", "examples"}',
    )
    enabled: bool = True


class McpServerUpdate(BaseModel):
    transport: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = None


class PolicyUpdate(BaseModel):
    patterns: Optional[List[str]] = Field(
        default=None, description="fnmatch 通配模式列表，如 [\"web_*\", \"knowledge_*\"]；null 表示不限制"
    )


def _error(exc: Exception) -> HTTPException:
    status = 404 if "不存在" in str(exc) else 400
    return HTTPException(status_code=status, detail=str(exc))


# ---- 工具与白名单 ----


@router.get("/tools", summary="工具列表（普通用户只显示自己可用的）")
async def list_tools(user: CurrentUser = Depends(get_current_user)) -> List[Dict[str, Any]]:
    registry = get_tool_registry()
    return [
        e.to_dict() for e in registry.entries()
        if user.is_admin or registry.is_allowed(e, user.role)
    ]


@router.get("/policies", summary="角色工具白名单")
async def get_policies(_: CurrentUser = Depends(require_admin)) -> Dict[str, List[str]]:
    return get_tool_registry().get_policies()


@router.put("/policies/{role}", summary="设置角色工具白名单")
async def put_policy(
    role: str, req: PolicyUpdate, _: CurrentUser = Depends(require_admin)
) -> Dict[str, List[str]]:
    if role == "admin":
        raise HTTPException(status_code=400, detail="admin 角色不受白名单限制")
    await set_tool_policy(role, req.patterns)
    return get_tool_registry().get_policies()


# ---- MCP ----


@router.get("/mcp/servers", summary="MCP 服务列表")
async def list_mcp(_: CurrentUser = Depends(require_admin)) -> List[Dict[str, Any]]:
    return await get_mcp_manager().list_servers()


@router.post("/mcp/servers", summary="添加 MCP 服务")
async def add_mcp(req: McpServerCreate, _: CurrentUser = Depends(require_admin)) -> Dict[str, Any]:
    try:
        data = await get_mcp_manager().add_server(req.name, req.transport, req.config, req.enabled)
    except McpError as exc:
        raise _error(exc) from exc
    servers = {s["name"]: s for s in await get_mcp_manager().list_servers()}
    data["status"] = servers.get(req.name, {}).get("status")
    return data


@router.patch("/mcp/servers/{server_id}", summary="修改 MCP 服务")
async def update_mcp(
    server_id: str, req: McpServerUpdate, _: CurrentUser = Depends(require_admin)
) -> Dict[str, Any]:
    try:
        return await get_mcp_manager().update_server(server_id, **req.model_dump(exclude_none=True))
    except McpError as exc:
        raise _error(exc) from exc


@router.delete("/mcp/servers/{server_id}", summary="删除 MCP 服务")
async def delete_mcp(server_id: str, _: CurrentUser = Depends(require_admin)) -> Dict[str, Any]:
    try:
        await get_mcp_manager().delete_server(server_id)
    except McpError as exc:
        raise _error(exc) from exc
    return {"ok": True}


@router.post("/mcp/reload", summary="重新连接全部 MCP 服务")
async def reload_mcp(_: CurrentUser = Depends(require_admin)) -> List[Dict[str, Any]]:
    return await get_mcp_manager().reload_all()


# ---- 本地插件 ----


@router.get("", summary="本地插件列表")
async def list_plugins(_: CurrentUser = Depends(require_admin)) -> List[Dict[str, Any]]:
    return get_plugin_manager().list()


@router.post("/reload", summary="重新扫描并加载全部本地插件")
async def reload_all(_: CurrentUser = Depends(require_admin)) -> List[Dict[str, Any]]:
    return await get_plugin_manager().reload_all()


def _plugin_info(name: str) -> Dict[str, Any]:
    for item in get_plugin_manager().list():
        if item["name"] == name:
            return item
    raise HTTPException(status_code=404, detail=f"插件不存在: {name}")


@router.post("/{name}/enable", summary="启用插件")
async def enable_plugin(name: str, _: CurrentUser = Depends(require_admin)) -> Dict[str, Any]:
    try:
        await get_plugin_manager().enable(name)
    except PluginError as exc:
        raise _error(exc) from exc
    return _plugin_info(name)


@router.post("/{name}/disable", summary="停用插件")
async def disable_plugin(name: str, _: CurrentUser = Depends(require_admin)) -> Dict[str, Any]:
    try:
        await get_plugin_manager().disable(name)
    except PluginError as exc:
        raise _error(exc) from exc
    return _plugin_info(name)


@router.post("/{name}/reload", summary="热重载插件（修改代码后无需重启服务）")
async def reload_plugin(name: str, _: CurrentUser = Depends(require_admin)) -> Dict[str, Any]:
    try:
        await get_plugin_manager().reload(name)
    except PluginError as exc:
        raise _error(exc) from exc
    return _plugin_info(name)
