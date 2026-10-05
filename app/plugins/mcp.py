"""MCP 服务接入（langchain-mcp-adapters）。

每个 MCP 服务注册为一个能力 mcp_<服务名>，工具名统一为 mcp_<服务名>_<工具名>；
通用 Agent（general_task 意图及降级链）可以使用全部 MCP 工具。

服务配置（mcp_servers 表）示例：
    stdio:            {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/data"], "env": {}}
    streamable_http:  {"url": "https://mcp.example.com/mcp", "headers": {"Authorization": "Bearer xxx"}}
    可选字段：
      "intent": {"description": "...", "keywords": [...], "examples": [...]}  为该服务注册专属意图
      "timeout": 30                                                          拉取工具列表的超时（秒）

安全：stdio 会在服务器上启动子进程，只有管理员可以配置 MCP 服务。
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool
from sqlalchemy import select

from app.capabilities.base import Capability, CapabilityTool
from app.capabilities.manager import get_capability_manager
from app.core.db import session_scope
from app.intent.catalog import IntentSpec
from app.models import McpServer

logger = logging.getLogger(__name__)

TRANSPORTS = ("stdio", "streamable_http", "sse", "websocket")
_NAME_RE = re.compile(r"^[A-Za-z0-9_]{1,32}$")


class McpError(Exception):
    """MCP 配置或连接错误。"""


def server_to_dict(row: McpServer) -> Dict[str, Any]:
    config = dict(row.config or {})
    # 不回显敏感信息
    if "headers" in config:
        config["headers"] = {k: "***" for k in config["headers"]}
    if "env" in config:
        config["env"] = {k: "***" for k in config["env"]}
    return {
        "id": row.id,
        "name": row.name,
        "transport": row.transport,
        "config": config,
        "enabled": row.enabled,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def build_connection(transport: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """把服务配置转换为 MultiServerMCPClient 的连接参数。"""
    if transport not in TRANSPORTS:
        raise McpError(f"transport 只能是 {TRANSPORTS}")
    if transport == "stdio":
        if not config.get("command"):
            raise McpError("stdio 服务需要 command")
        conn: Dict[str, Any] = {
            "transport": "stdio",
            "command": config["command"],
            "args": list(config.get("args") or []),
        }
        if config.get("env"):
            conn["env"] = dict(config["env"])
        if config.get("cwd"):
            conn["cwd"] = config["cwd"]
        return conn
    if not str(config.get("url", "")).startswith(("http://", "https://", "ws://", "wss://")):
        raise McpError(f"{transport} 服务需要 url")
    conn = {"transport": transport, "url": config["url"]}
    if config.get("headers") and transport != "websocket":
        conn["headers"] = dict(config["headers"])
    return conn


class McpCapability(Capability):
    """一个 MCP 服务对应的能力。"""

    def __init__(self, server_name: str, tools: List[BaseTool], intent: Optional[Dict[str, Any]] = None) -> None:
        self.name = f"mcp_{server_name}"
        self.description = f"MCP 服务 {server_name}"
        self._server = server_name
        self._tools = tools
        self._intent = intent or {}

    def tools(self):
        decls = []
        for tool in self._tools:
            meta = tool.metadata or {}
            # MCP 工具注解：destructiveHint=True 或 readOnlyHint=False 视为写操作
            write = bool(meta.get("destructiveHint")) or meta.get("readOnlyHint") is False
            decls.append(CapabilityTool(tool, tags=("mcp", f"mcp:{self._server}"), write=write))
        return decls

    def intents(self):
        if not self._intent.get("description"):
            return []
        return [IntentSpec(
            name=self.name,
            description=str(self._intent["description"]),
            keywords=list(self._intent.get("keywords") or []),
            examples=list(self._intent.get("examples") or []),
        )]


class McpManager:
    """MCP 服务管理。"""

    def __init__(self) -> None:
        self._status: Dict[str, Dict[str, Any]] = {}

    # ---- 配置 CRUD ----

    async def list_servers(self) -> List[Dict[str, Any]]:
        async with session_scope() as session:
            rows = (await session.execute(select(McpServer).order_by(McpServer.created_at))).scalars().all()
            result = []
            for row in rows:
                data = server_to_dict(row)
                data["status"] = self._status.get(row.name, {"state": "not_loaded"})
                result.append(data)
            return result

    async def add_server(
        self, name: str, transport: str, config: Dict[str, Any], enabled: bool = True
    ) -> Dict[str, Any]:
        if not _NAME_RE.match(name):
            raise McpError("服务名只能包含字母、数字、下划线（最长 32）")
        build_connection(transport, config)
        async with session_scope() as session:
            exists = (await session.execute(select(McpServer.id).where(McpServer.name == name))).first()
            if exists:
                raise McpError(f"MCP 服务「{name}」已存在")
            row = McpServer(name=name, transport=transport, config=dict(config), enabled=enabled)
            session.add(row)
            await session.flush()
            data = server_to_dict(row)
        if enabled:
            await self.load_server(name)
        return data

    async def update_server(self, server_id: str, **fields: Any) -> Dict[str, Any]:
        async with session_scope() as session:
            row = await session.get(McpServer, server_id)
            if row is None:
                raise McpError("MCP 服务不存在")
            transport = fields.get("transport") or row.transport
            config = fields.get("config") if fields.get("config") is not None else row.config
            build_connection(transport, config)
            row.transport, row.config = transport, dict(config)
            if fields.get("enabled") is not None:
                row.enabled = bool(fields["enabled"])
            name, enabled = row.name, row.enabled
            data = server_to_dict(row)
        await self.unload_server(name)
        if enabled:
            await self.load_server(name)
        return data

    async def delete_server(self, server_id: str) -> None:
        async with session_scope() as session:
            row = await session.get(McpServer, server_id)
            if row is None:
                raise McpError("MCP 服务不存在")
            name = row.name
            await session.delete(row)
        await self.unload_server(name)
        self._status.pop(name, None)

    # ---- 加载 ----

    async def load_server(self, name: str) -> Dict[str, Any]:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        async with session_scope() as session:
            row = (await session.execute(select(McpServer).where(McpServer.name == name))).scalar_one_or_none()
            if row is None:
                raise McpError(f"MCP 服务不存在: {name}")
            transport, config = row.transport, dict(row.config or {})

        connection_name = f"mcp_{name}"
        try:
            client = MultiServerMCPClient(
                {connection_name: build_connection(transport, config)},
                tool_name_prefix=True,
            )
            tools = await asyncio.wait_for(
                client.get_tools(server_name=connection_name),
                timeout=float(config.get("timeout", 30)),
            )
            await get_capability_manager().register(
                McpCapability(name, tools, intent=config.get("intent")), source=f"mcp:{name}"
            )
            self._status[name] = {"state": "loaded", "tools": [t.name for t in tools], "error": ""}
            logger.info("MCP 服务 '%s' 已加载 %d 个工具", name, len(tools))
            if config.get("intent"):
                await self._refresh_intents()
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            self._status[name] = {"state": "error", "tools": [], "error": message[:500]}
            logger.error("MCP 服务 '%s' 加载失败: %s", name, message)
        return self._status[name]

    async def unload_server(self, name: str) -> None:
        await get_capability_manager().unregister(f"mcp_{name}")
        if name in self._status:
            self._status[name] = {"state": "not_loaded", "tools": [], "error": ""}

    async def load_all(self) -> None:
        try:
            async with session_scope() as session:
                names = (
                    await session.execute(select(McpServer.name).where(McpServer.enabled.is_(True)))
                ).scalars().all()
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取 MCP 服务配置失败: %s", exc)
            return
        for name in names:
            await self.load_server(name)

    async def reload_all(self) -> List[Dict[str, Any]]:
        for name in list(self._status):
            await self.unload_server(name)
        await self.load_all()
        return await self.list_servers()

    async def close(self) -> None:
        """适配器默认每次工具调用新建会话，无常驻连接需要关闭。"""
        self._status.clear()

    async def _refresh_intents(self) -> None:
        try:
            from app.intent.blend import get_intent_fusion

            await get_intent_fusion().refresh()
        except Exception as exc:  # noqa: BLE001
            logger.warning("刷新意图索引失败: %s", exc)


_manager: Optional[McpManager] = None


def get_mcp_manager() -> McpManager:
    global _manager
    if _manager is None:
        _manager = McpManager()
    return _manager
