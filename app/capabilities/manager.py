"""能力管理器：负责把 Capability 注册到工具注册表、意图目录、Agent 注册表。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import BaseTool

from app.agents.base import AgentContext
from app.agents.langchain_agent import LangChainAgent, clear_agent_cache
from app.capabilities.base import Capability, CapabilityTool
from app.intent.catalog import IntentSpec, get_intent_catalog
from app.router.pool import get_agent_registry
from app.tools.registry import get_tool_registry

logger = logging.getLogger(__name__)

#: 降级链末端，所有自动生成的路由都会追加
_FALLBACK_CHAIN = ["general_agent", "fallback_agent"]


def capability_tag(name: str) -> str:
    """每个能力的工具都会带上的隐式标签。"""
    return f"cap:{name}"


class ToolsetAgent(LangChainAgent):
    """自动生成的 Agent：只使用某个能力的工具。"""

    def __init__(self, capability: Capability, intents: List[IntentSpec]) -> None:
        self.agent_id = f"{capability.name}_agent"
        self.description = capability.description or capability.name
        self.tool_tags = (capability_tag(capability.name),)
        intent_lines = "\n".join(f"- {i.description}" for i in intents)
        parts: List[str] = [
            f"你是 Synapse 智能助手中负责「{self.description}」的助手。",
            f"你擅长处理以下类型的请求：\n{intent_lines}" if intent_lines else "",
            "请优先调用可用工具获取真实结果，不要编造；工具出错时如实告知用户。",
        ]
        # 静态提示词：记忆召回 / 用户画像由 build_context_block 注入消息序列
        self._system_prompt = "\n".join(p for p in parts if p)

    def build_system_prompt(self, context: AgentContext) -> str:
        return self._system_prompt


@dataclass
class _Registered:
    capability: Capability
    source: str
    agent_ids: List[str] = field(default_factory=list)
    route_intents: List[str] = field(default_factory=list)
    tool_names: List[str] = field(default_factory=list)
    intent_names: List[str] = field(default_factory=list)


class CapabilityManager:
    """能力注册与注销。"""

    def __init__(self) -> None:
        self._registered: Dict[str, _Registered] = {}

    async def register(self, capability: Capability, source: Optional[str] = None) -> None:
        """注册能力（同名能力会先注销再注册，用于热重载）。

        Args:
            capability: 能力实例
            source: 来源标识，默认 builtin:<能力名>；插件为 plugin:<插件名>，MCP 为 mcp:<服务名>
        """
        if not capability.name:
            raise ValueError("Capability.name 不能为空")
        if capability.name in self._registered:
            await self.unregister(capability.name)

        source = source or f"builtin:{capability.name}"
        tool_registry = get_tool_registry()
        catalog = get_intent_catalog()
        agent_registry = get_agent_registry()
        record = _Registered(capability=capability, source=source)

        # 1. 工具
        for decl in capability.tools():
            tool, tags, write, roles = self._unpack_tool(decl)
            tool_registry.register(
                tool,
                source=source,
                capability=capability.name,
                tags=(*tags, capability_tag(capability.name)),
                write=write,
                roles=roles,
            )
            record.tool_names.append(tool.name)

        # 2. 意图
        intents = capability.intents()
        for spec in intents:
            if spec.source == "config":
                spec.source = source
            catalog.register(spec)
            record.intent_names.append(spec.name)

        # 3. Agent
        agents = list(capability.agents())
        routes = dict(capability.routes())
        # 已有路由的意图（如向默认意图追加关键词）不自动生成 Agent，避免覆盖原路由
        unrouted = [
            s for s in intents
            if s.name not in routes and not agent_registry.get_agents_for_intent(s.name)
        ]
        if unrouted and record.tool_names:
            auto_agent = ToolsetAgent(capability, unrouted)
            agents.append(auto_agent)
            for spec in unrouted:
                routes[spec.name] = [auto_agent.agent_id, *_FALLBACK_CHAIN]
        elif unrouted:
            for spec in unrouted:
                routes[spec.name] = list(_FALLBACK_CHAIN)

        for agent in agents:
            agent_registry.register_agent(agent)
            record.agent_ids.append(agent.agent_id)

        # 4. 路由
        for intent, agent_ids in routes.items():
            agent_registry.register_route(intent, agent_ids)
            record.route_intents.append(intent)

        self._registered[capability.name] = record
        # 工具集变了，缓存的 Agent 图持有的是旧工具对象，必须作废
        clear_agent_cache()
        try:
            await capability.startup()
        except Exception as exc:  # noqa: BLE001
            logger.error("能力 '%s' startup 失败: %s", capability.name, exc)

        logger.info(
            "能力管理器: 已注册 '%s' (来源=%s, 工具=%d, 意图=%s, Agent=%s)",
            capability.name, source, len(record.tool_names),
            record.intent_names, record.agent_ids,
        )

    async def unregister(self, name: str) -> None:
        """注销能力：移除其工具、意图、路由与 Agent。"""
        record = self._registered.pop(name, None)
        if record is None:
            return
        try:
            await record.capability.shutdown()
        except Exception as exc:  # noqa: BLE001
            logger.warning("能力 '%s' shutdown 失败: %s", name, exc)

        get_tool_registry().unregister_source(record.source)
        get_intent_catalog().unregister_source(record.source)
        agent_registry = get_agent_registry()
        for intent in record.route_intents:
            agent_registry.unregister_route(intent)
        for agent_id in record.agent_ids:
            agent_registry.unregister_agent(agent_id)
        # 工具已下线，缓存的 Agent 图必须作废
        clear_agent_cache()
        logger.info("能力管理器: 已注销 '%s'", name)

    async def shutdown_all(self) -> None:
        for name in list(self._registered):
            record = self._registered[name]
            try:
                await record.capability.shutdown()
            except Exception as exc:  # noqa: BLE001
                logger.warning("能力 '%s' shutdown 失败: %s", name, exc)

    def get(self, name: str) -> Optional[Capability]:
        record = self._registered.get(name)
        return record.capability if record else None

    def list(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": name,
                "description": r.capability.description,
                "version": r.capability.version,
                "source": r.source,
                "tools": r.tool_names,
                "intents": r.intent_names,
                "agents": r.agent_ids,
            }
            for name, r in self._registered.items()
        ]

    @staticmethod
    def _unpack_tool(decl) -> Tuple[BaseTool, Tuple[str, ...], bool, Optional[Tuple[str, ...]]]:
        if isinstance(decl, CapabilityTool):
            roles = tuple(decl.roles) if decl.roles else None
            return decl.tool, tuple(decl.tags), decl.write, roles
        return decl, (), False, None


_manager: Optional[CapabilityManager] = None


def get_capability_manager() -> CapabilityManager:
    global _manager
    if _manager is None:
        _manager = CapabilityManager()
    return _manager
