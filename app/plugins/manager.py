"""本地 Python 插件管理器。

插件目录结构：
    plugins/
      my_plugin/
        plugin.yaml      清单：name / version / description / entry / permissions
        __init__.py      入口模块，定义一个 Capability 子类

plugin.yaml 示例：
    name: my_plugin
    version: 1.0.0
    description: 示例插件
    entry: plugin:MyPlugin        # 模块:类名，模块相对插件目录；省略时在 __init__ 中自动查找 Capability 子类
    permissions: [read]           # 声明 write 才允许注册写操作工具

安全：本地插件是可执行代码，只有管理员能启停 / 重载；单个插件加载失败不影响其他插件。
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import re
import shutil
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from sqlalchemy import select

from app.capabilities.base import Capability, CapabilityTool
from app.capabilities.manager import get_capability_manager
from app.config import get_settings
from app.core.db import session_scope
from app.models import PluginState, utcnow

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_PACKAGE_ROOT = "synapse_plugins"


class PluginError(Exception):
    """插件加载 / 管理错误。"""


@dataclass
class PluginManifest:
    name: str
    path: Path
    version: str = "0.0.0"
    description: str = ""
    author: str = ""
    entry: str = ""
    permissions: List[str] = field(default_factory=lambda: ["read"])

    @classmethod
    def load(cls, plugin_dir: Path) -> "PluginManifest":
        manifest_file = plugin_dir / "plugin.yaml"
        data = yaml.safe_load(manifest_file.read_text(encoding="utf-8")) or {}
        name = str(data.get("name") or plugin_dir.name)
        if not _NAME_RE.match(name):
            raise PluginError(f"插件名不合法: {name}（只能包含字母、数字、下划线、横线）")
        permissions = data.get("permissions") or ["read"]
        if isinstance(permissions, str):
            permissions = [permissions]
        return cls(
            name=name,
            path=plugin_dir,
            version=str(data.get("version", "0.0.0")),
            description=str(data.get("description", "")),
            author=str(data.get("author", "")),
            entry=str(data.get("entry", "")),
            permissions=[str(p) for p in permissions],
        )


@dataclass
class PluginRecord:
    manifest: PluginManifest
    enabled: bool = True
    loaded: bool = False
    error: str = ""
    capability_name: str = ""
    module_name: str = ""


class _PermissionFilteredCapability(Capability):
    """包装插件能力：未声明 write 权限时过滤掉写操作工具。"""

    def __init__(self, inner: Capability, allow_write: bool) -> None:
        self._inner = inner
        self._allow_write = allow_write
        self.name = inner.name
        self.description = inner.description
        self.version = inner.version

    def tools(self):
        result = []
        for decl in self._inner.tools():
            if isinstance(decl, CapabilityTool) and decl.write and not self._allow_write:
                logger.warning(
                    "插件 '%s' 未声明 write 权限，已忽略写操作工具 '%s'", self.name, decl.tool.name
                )
                continue
            result.append(decl)
        return result

    def intents(self):
        return self._inner.intents()

    def agents(self):
        return self._inner.agents()

    def routes(self):
        return self._inner.routes()

    async def startup(self) -> None:
        await self._inner.startup()

    async def shutdown(self) -> None:
        await self._inner.shutdown()


class PluginManager:
    """本地插件管理器。"""

    def __init__(self) -> None:
        self._records: Dict[str, PluginRecord] = {}

    @property
    def plugins_dir(self) -> Path:
        return Path(get_settings().plugins_dir)

    # ---- 发现 ----

    def discover(self) -> Dict[str, PluginManifest]:
        manifests: Dict[str, PluginManifest] = {}
        root = self.plugins_dir
        if not root.is_dir():
            return manifests
        for child in sorted(root.iterdir()):
            if not (child / "plugin.yaml").is_file():
                continue
            try:
                manifest = PluginManifest.load(child)
            except Exception as exc:  # noqa: BLE001
                logger.error("插件清单解析失败 %s: %s", child, exc)
                continue
            if manifest.name in manifests:
                logger.error("插件名重复: %s（%s 被忽略）", manifest.name, child)
                continue
            manifests[manifest.name] = manifest
        return manifests

    async def _states(self) -> Dict[str, bool]:
        try:
            async with session_scope() as session:
                rows = (await session.execute(select(PluginState))).scalars().all()
                return {r.name: r.enabled for r in rows}
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取插件启停状态失败（默认全部启用）: %s", exc)
            return {}

    async def _save_state(self, name: str, enabled: bool) -> None:
        async with session_scope() as session:
            row = await session.get(PluginState, name)
            if row is None:
                session.add(PluginState(name=name, enabled=enabled))
            else:
                row.enabled, row.updated_at = enabled, utcnow()

    # ---- 加载 / 卸载 ----

    def _import_capability(self, manifest: PluginManifest) -> tuple:
        if _PACKAGE_ROOT not in sys.modules:
            root = types.ModuleType(_PACKAGE_ROOT)
            root.__path__ = []  # type: ignore[attr-defined]
            sys.modules[_PACKAGE_ROOT] = root
        package_name = f"{_PACKAGE_ROOT}.{manifest.name.replace('-', '_')}"
        init_file = manifest.path / "__init__.py"
        if not init_file.is_file():
            raise PluginError("插件目录缺少 __init__.py")
        spec = importlib.util.spec_from_file_location(
            package_name, init_file, submodule_search_locations=[str(manifest.path)]
        )
        if spec is None or spec.loader is None:
            raise PluginError("无法加载插件入口模块")

        # 热重载必须读取最新源码：.pyc 只按秒级修改时间与文件大小判断是否过期，
        # 快速编辑时可能复用旧缓存，因此清掉插件的字节码缓存且导入时不再生成
        for cache_dir in manifest.path.rglob("__pycache__"):
            shutil.rmtree(cache_dir, ignore_errors=True)
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            module = importlib.util.module_from_spec(spec)
            sys.modules[package_name] = module
            spec.loader.exec_module(module)

            module_part, _, class_name = manifest.entry.partition(":")
            target = module
            if module_part and module_part != "__init__":
                target = importlib.import_module(f"{package_name}.{module_part}")
        finally:
            sys.dont_write_bytecode = previous
        if class_name:
            cls = getattr(target, class_name, None)
            if not (inspect.isclass(cls) and issubclass(cls, Capability)):
                raise PluginError(f"入口 {manifest.entry} 不是 Capability 子类")
        else:
            candidates = [
                obj for obj in vars(target).values()
                if inspect.isclass(obj) and issubclass(obj, Capability) and obj is not Capability
                and obj.__module__.startswith(package_name)
            ]
            if len(candidates) != 1:
                raise PluginError("请在 plugin.yaml 中用 entry 指定 Capability 子类（模块:类名）")
            cls = candidates[0]
        return cls(), package_name

    def _purge_modules(self, package_name: str) -> None:
        for mod in list(sys.modules):
            if mod == package_name or mod.startswith(package_name + "."):
                sys.modules.pop(mod, None)

    async def load(self, name: str) -> PluginRecord:
        manifests = self.discover()
        manifest = manifests.get(name)
        if manifest is None:
            raise PluginError(f"插件不存在: {name}")
        record = self._records.get(name) or PluginRecord(manifest=manifest)
        record.manifest = manifest
        if record.loaded:
            await self.unload(name, refresh=False)
        try:
            capability, package_name = self._import_capability(manifest)
            if not capability.name:
                capability.name = manifest.name
            if not capability.description:
                capability.description = manifest.description
            capability.version = manifest.version
            wrapped = _PermissionFilteredCapability(capability, "write" in manifest.permissions)
            await get_capability_manager().register(wrapped, source=f"plugin:{manifest.name}")
            record.loaded, record.error = True, ""
            record.capability_name, record.module_name = wrapped.name, package_name
            logger.info("插件 '%s' v%s 已加载", manifest.name, manifest.version)
        except Exception as exc:  # noqa: BLE001
            record.loaded, record.error = False, f"{type(exc).__name__}: {exc}"
            logger.error("插件 '%s' 加载失败: %s", name, exc)
        self._records[name] = record
        await self._refresh_intents()
        return record

    async def unload(self, name: str, refresh: bool = True) -> None:
        record = self._records.get(name)
        if record is None or not record.loaded:
            return
        await get_capability_manager().unregister(record.capability_name)
        self._purge_modules(record.module_name)
        record.loaded = False
        logger.info("插件 '%s' 已卸载", name)
        if refresh:
            await self._refresh_intents()

    async def load_all(self) -> None:
        states = await self._states()
        for name, manifest in self.discover().items():
            enabled = states.get(name, True)
            if enabled:
                await self.load(name)
            else:
                self._records[name] = PluginRecord(manifest=manifest, enabled=False)
        logger.info(
            "本地插件: 发现 %d 个，已加载 %d 个",
            len(self._records), sum(1 for r in self._records.values() if r.loaded),
        )

    async def enable(self, name: str) -> PluginRecord:
        await self._save_state(name, True)
        record = await self.load(name)
        record.enabled = True
        return record

    async def disable(self, name: str) -> PluginRecord:
        if name not in self.discover() and name not in self._records:
            raise PluginError(f"插件不存在: {name}")
        await self._save_state(name, False)
        await self.unload(name)
        record = self._records.get(name) or PluginRecord(manifest=self.discover()[name])
        record.enabled = False
        self._records[name] = record
        return record

    async def reload(self, name: str) -> PluginRecord:
        record = self._records.get(name)
        if record is not None and not record.enabled:
            raise PluginError(f"插件 {name} 已停用，请先启用")
        return await self.load(name)

    async def reload_all(self) -> List[Dict[str, Any]]:
        for name in list(self._records):
            await self.unload(name, refresh=False)
        self._records.clear()
        await self.load_all()
        return self.list()

    async def _refresh_intents(self) -> None:
        """意图目录变化后重建向量意图索引（best-effort）。"""
        try:
            from app.intent.blend import get_intent_fusion

            await get_intent_fusion().refresh()
        except Exception as exc:  # noqa: BLE001
            logger.warning("刷新意图索引失败（不影响插件使用）: %s", exc)

    def list(self) -> List[Dict[str, Any]]:
        manager = get_capability_manager()
        info = {c["name"]: c for c in manager.list()}
        result = []
        for name, record in sorted(self._records.items()):
            cap = info.get(record.capability_name, {})
            result.append({
                "name": name,
                "version": record.manifest.version,
                "description": record.manifest.description,
                "author": record.manifest.author,
                "permissions": record.manifest.permissions,
                "enabled": record.enabled,
                "loaded": record.loaded,
                "error": record.error,
                "tools": cap.get("tools", []),
                "intents": cap.get("intents", []),
                "agents": cap.get("agents", []),
            })
        return result


_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    global _manager
    if _manager is None:
        _manager = PluginManager()
    return _manager
