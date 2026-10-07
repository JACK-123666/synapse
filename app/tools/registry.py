"""Agent 工具注册表与执行器。

- call_tool：工具执行入口（保留兼容；web 能力的 web_search 工具复用它）
- ToolRegistry：LangChain 工具注册表，记录每个工具的来源（内置 / 插件 / MCP）、
  标签、是否写操作、允许的角色，并按角色白名单过滤给 Agent 使用
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any, Dict, FrozenSet, Iterable, List, Optional

from langchain_core.tools import BaseTool

from app.tools.search import web_search

logger = logging.getLogger(__name__)


async def call_tool(
    name: str,
    arguments: Dict[str, Any],
    *,
    collector: Optional[List[Dict[str, str]]] = None,
) -> str:
    """执行 Agent 工具并返回可回填给 LLM 的字符串结果。"""
    if name == "web_search":
        query = str(arguments.get("query", "")).strip()
        max_results = int(arguments.get("max_results", 5))
        if not query:
            return "缺少搜索关键词，无法搜索。"

        results = await web_search(query, max_results=max_results)
        if collector is not None:
            collector.extend(results)

        if not results:
            return "未找到相关结果，请换个关键词或结合已有知识回答。"

        lines = [
            f"- [{r['title']}]({r['url']})\n  {r['snippet']}"
            for r in results
        ]
        return "\n\n".join(lines)

    # 其他工具从 LangChain 工具注册表中查找执行
    registered = get_tool_registry().get(name)
    if registered is not None:
        result = await registered.ainvoke(arguments)
        return result if isinstance(result, str) else str(result)

    raise ValueError(f"未知工具: {name}")


# ---- LangChain 工具注册表 ----


@dataclass
class ToolEntry:
    """注册表中的一个工具。

    Attributes:
        tool: LangChain 工具实例
        source: 来源，builtin / plugin:<名称> / mcp:<服务名>
        capability: 所属能力名
        tags: 标签，Agent 按标签挑选工具
        write: 是否为写操作（有副作用）
        roles: 允许使用的角色；None 表示所有角色
    """

    tool: BaseTool
    source: str = "builtin"
    capability: str = "core"
    tags: FrozenSet[str] = field(default_factory=frozenset)
    write: bool = False
    roles: Optional[FrozenSet[str]] = None

    @property
    def name(self) -> str:
        """工具名。"""
        return self.tool.name

    def to_dict(self) -> Dict[str, Any]:
        """转成对外字典，供 /plugins/tools 展示。"""
        return {
            "name": self.tool.name,
            "description": self.tool.description,
            "source": self.source,
            "capability": self.capability,
            "tags": sorted(self.tags),
            "write": self.write,
            "roles": sorted(self.roles) if self.roles else None,
        }


class ToolRegistry:
    """LangChain 工具注册表（线程安全）。

    角色白名单：管理员可为某个角色设置 fnmatch 通配模式列表（如 ["web_*", "knowledge_*"]），
    该角色只能使用名称匹配的工具；未设置白名单的角色可使用全部工具。
    admin 角色不受白名单限制。
    """

    def __init__(self) -> None:
        self._entries: Dict[str, ToolEntry] = {}
        self._policies: Dict[str, List[str]] = {}
        self._lock = threading.Lock()

    # 注册 / 注销

    def register(
        self,
        tool: BaseTool,
        *,
        source: str = "builtin",
        capability: str = "core",
        tags: Iterable[str] = (),
        write: bool = False,
        roles: Optional[Iterable[str]] = None,
    ) -> ToolEntry:
        """注册一个工具。同名工具被不同来源覆盖时记 warning。"""
        entry = ToolEntry(
            tool=tool,
            source=source,
            capability=capability,
            tags=frozenset(tags),
            write=write,
            roles=frozenset(roles) if roles else None,
        )
        with self._lock:
            if tool.name in self._entries and self._entries[tool.name].source != source:
                logger.warning(
                    "工具注册表: 工具 '%s' 已由 %s 注册，被 %s 覆盖",
                    tool.name, self._entries[tool.name].source, source,
                )
            self._entries[tool.name] = entry
        logger.info("工具注册表: 已注册 '%s' (来源=%s, 写操作=%s)", tool.name, source, write)
        return entry

    def unregister(self, name: str) -> None:
        """按名称注销单个工具。"""
        with self._lock:
            self._entries.pop(name, None)

    def unregister_source(self, source: str) -> int:
        """注销某个来源（插件 / MCP 服务）的全部工具，返回注销数量。"""
        with self._lock:
            names = [n for n, e in self._entries.items() if e.source == source]
            for n in names:
                self._entries.pop(n, None)
        if names:
            logger.info("工具注册表: 已注销来源 %s 的 %d 个工具", source, len(names))
        return len(names)

    # 查询

    def get(self, name: str) -> Optional[BaseTool]:
        """按名称取工具实例。"""
        entry = self._entries.get(name)
        return entry.tool if entry else None

    def entry(self, name: str) -> Optional[ToolEntry]:
        """按名称取工具条目（含来源、标签、写标记、角色限制等元数据）。"""
        return self._entries.get(name)

    def entries(self) -> List[ToolEntry]:
        """返回全部工具条目的快照。"""
        with self._lock:
            return list(self._entries.values())

    # 角色白名单

    def set_policy(self, role: str, patterns: Optional[List[str]]) -> None:
        """设置角色白名单；patterns 为 None 表示取消限制。"""
        with self._lock:
            if patterns is None:
                self._policies.pop(role, None)
            else:
                self._policies[role] = list(patterns)

    def get_policies(self) -> Dict[str, List[str]]:
        """返回各角色的工具白名单。"""
        with self._lock:
            return {k: list(v) for k, v in self._policies.items()}

    def is_allowed(self, entry: ToolEntry, role: str) -> bool:
        """该角色能否使用这个工具：先看工具自身的角色限制，再看角色白名单；admin 不受白名单限制。"""
        if entry.roles and role not in entry.roles:
            return False
        if role == "admin":
            return True
        patterns = self._policies.get(role)
        if patterns is None:
            return True
        return any(fnmatch(entry.tool.name, p) for p in patterns)

    def tools_for(
        self,
        role: str,
        tags: Optional[Iterable[str]] = None,
        names: Optional[Iterable[str]] = None,
        include_write: bool = True,
    ) -> List[BaseTool]:
        """按角色、标签、名称筛选可用工具。

        Args:
            role: 当前用户角色
            tags: 只保留带有任一标签的工具；None 表示不按标签过滤
            names: 只保留这些名称的工具；None 表示不按名称过滤
            include_write: 是否包含写操作工具
        """
        tag_set = set(tags) if tags is not None else None
        name_set = set(names) if names is not None else None
        result: List[BaseTool] = []
        for entry in self.entries():
            if tag_set is not None and not (entry.tags & tag_set):
                continue
            if name_set is not None and entry.tool.name not in name_set:
                continue
            if entry.write and not include_write:
                continue
            if not self.is_allowed(entry, role):
                continue
            result.append(entry.tool)
        return result


_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """获取全局工具注册表单例。"""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


__all__ = [
    "call_tool",
    "ToolEntry",
    "ToolRegistry",
    "get_tool_registry",
]
