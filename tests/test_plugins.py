"""阶段 6：本地插件与 MCP。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
async def core_registered():
    from app.capabilities.core import CoreCapability
    from app.capabilities.manager import get_capability_manager

    await get_capability_manager().register(CoreCapability())
    yield


@pytest.fixture
def plugin_manager(monkeypatch):
    from app.plugins import manager as plugin_module

    monkeypatch.setattr(plugin_module, "_manager", None)
    manager = plugin_module.get_plugin_manager()

    async def _no_refresh():
        return None

    monkeypatch.setattr(manager, "_refresh_intents", _no_refresh)
    return manager


def _use_plugins_dir(monkeypatch, path: Path) -> None:
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "plugins_dir", str(path))


def _write_plugin(root: Path, name: str, body: str, permissions: str = "[read]") -> Path:
    plugin_dir = root / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(
        f"name: {name}\nversion: 1.0.0\ndescription: test\npermissions: {permissions}\n", encoding="utf-8"
    )
    (plugin_dir / "__init__.py").write_text(body, encoding="utf-8")
    return plugin_dir


_PLUGIN_TEMPLATE = '''
from langchain_core.tools import tool
from app.capabilities.base import Capability, CapabilityTool


@tool("{tool_name}")
def version_tool() -> str:
    """返回版本。"""
    return "{version}"


@tool("{tool_name}_write")
def write_tool() -> str:
    """写操作。"""
    return "written"


class TestPlugin(Capability):
    name = "{name}"

    def tools(self):
        return [version_tool, CapabilityTool(write_tool, write=True)]
'''


# ---- 本地插件 ----


async def test_example_plugin_loads_with_auto_agent(db, core_registered, plugin_manager, monkeypatch):
    from app.intent.catalog import get_intent_catalog
    from app.router.pool import get_agent_registry
    from app.tools.registry import get_tool_registry

    _use_plugins_dir(monkeypatch, ROOT / "plugins")
    await plugin_manager.load_all()
    info = {p["name"]: p for p in plugin_manager.list()}
    assert info["example_plugin"]["loaded"], info["example_plugin"]["error"]
    assert set(info["example_plugin"]["tools"]) == {"get_current_time", "calculator"}

    assert "utility_tools" in get_intent_catalog().names()
    assert get_agent_registry().get_routes()["utility_tools"] == [
        "example_plugin_agent", "general_agent", "fallback_agent",
    ]
    calculator = get_tool_registry().get("calculator")
    assert calculator.invoke({"expression": "(12+30)*4"}) == "(12+30)*4 = 168"
    assert "无法计算" in calculator.invoke({"expression": "__import__('os').system('dir')"})


async def test_disable_persists_and_enable_restores(db, core_registered, plugin_manager, monkeypatch):
    from app.plugins import manager as plugin_module
    from app.tools.registry import get_tool_registry

    _use_plugins_dir(monkeypatch, ROOT / "plugins")
    await plugin_manager.load_all()
    await plugin_manager.disable("example_plugin")
    assert get_tool_registry().get("calculator") is None

    # 新的管理器实例（模拟重启）读取持久化的停用状态
    monkeypatch.setattr(plugin_module, "_manager", None)
    fresh = plugin_module.get_plugin_manager()
    monkeypatch.setattr(fresh, "_refresh_intents", plugin_manager._refresh_intents)
    await fresh.load_all()
    assert get_tool_registry().get("calculator") is None
    assert fresh.list()[0]["enabled"] is False

    await fresh.enable("example_plugin")
    assert get_tool_registry().get("calculator") is not None


async def test_hot_reload_and_write_permission(db, core_registered, plugin_manager, monkeypatch, tmp_path):
    from app.tools.registry import get_tool_registry

    _use_plugins_dir(monkeypatch, tmp_path)
    plugin_dir = _write_plugin(
        tmp_path, "hot", _PLUGIN_TEMPLATE.format(tool_name="hot_version", version="v1", name="hot")
    )
    _write_plugin(tmp_path, "broken", "this is not python")
    await plugin_manager.load_all()

    info = {p["name"]: p for p in plugin_manager.list()}
    assert info["broken"]["loaded"] is False and info["broken"]["error"]
    assert info["hot"]["loaded"] is True
    assert get_tool_registry().get("hot_version").invoke({}) == "v1"
    # 未声明 write 权限：写操作工具不注册
    assert get_tool_registry().get("hot_version_write") is None

    (plugin_dir / "__init__.py").write_text(
        _PLUGIN_TEMPLATE.format(tool_name="hot_version", version="v2", name="hot"), encoding="utf-8"
    )
    (plugin_dir / "plugin.yaml").write_text(
        "name: hot\nversion: 2.0.0\npermissions: [read, write]\n", encoding="utf-8"
    )
    await plugin_manager.reload("hot")
    assert get_tool_registry().get("hot_version").invoke({}) == "v2"
    assert get_tool_registry().entry("hot_version_write").write is True
    assert {p["name"]: p for p in plugin_manager.list()}["hot"]["version"] == "2.0.0"


async def test_plugin_api_admin_only(db, auth_enabled, plugin_manager, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.auth import router as auth_router
    from app.api.auth import users_router
    from app.api.plugins import router

    _use_plugins_dir(monkeypatch, ROOT / "plugins")
    app = FastAPI()
    app.include_router(auth_router)
    app.include_router(users_router)
    app.include_router(router)
    client = TestClient(app)
    admin_token = client.post("/auth/login", json={"username": "admin", "password": "admin-pass"}).json()["access_token"]
    admin = {"Authorization": f"Bearer {admin_token}"}
    client.post("/users", json={"username": "bob", "password": "bob-pass"}, headers=admin)
    bob_token = client.post("/auth/login", json={"username": "bob", "password": "bob-pass"}).json()["access_token"]
    bob = {"Authorization": f"Bearer {bob_token}"}

    assert client.get("/plugins", headers=bob).status_code == 403
    assert client.post("/plugins/reload", headers=admin).status_code == 200
    assert client.put("/plugins/policies/user", json={"patterns": ["get_*"]}, headers=admin).status_code == 200
    tools = [t["name"] for t in client.get("/plugins/tools", headers=bob).json()]
    assert tools == ["get_current_time"]


# ---- MCP ----


async def test_mcp_stdio_server_end_to_end(db, core_registered, monkeypatch):
    from app.plugins import mcp as mcp_module
    from app.tools.registry import get_tool_registry

    monkeypatch.setattr(mcp_module, "_manager", None)
    manager = mcp_module.get_mcp_manager()
    await manager.add_server(
        "echo", "stdio",
        {"command": sys.executable, "args": [str(ROOT / "tests" / "mcp_echo_server.py")], "timeout": 60},
    )
    servers = await manager.list_servers()
    status = servers[0]["status"]
    assert status["state"] == "loaded", status
    assert set(status["tools"]) == {"mcp_echo_echo", "mcp_echo_wipe"}

    registry = get_tool_registry()
    assert registry.entry("mcp_echo_wipe").write is True
    assert registry.entry("mcp_echo_echo").write is False
    result = await registry.get("mcp_echo_echo").ainvoke({"text": "你好"})
    text = result if isinstance(result, str) else str(result)
    assert "echo: 你好" in text

    await manager.delete_server(servers[0]["id"])
    assert registry.get("mcp_echo_echo") is None


def test_mcp_connection_validation():
    from app.plugins.mcp import McpError, build_connection

    assert build_connection("streamable_http", {"url": "https://x/mcp", "headers": {"A": "b"}})["headers"] == {"A": "b"}
    with pytest.raises(McpError):
        build_connection("stdio", {})
    with pytest.raises(McpError):
        build_connection("ftp", {"url": "ftp://x"})
