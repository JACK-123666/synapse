"""能力（Capability）接口。

内置能力（知识库、记忆、网页、代码仓库、定时任务）和外部插件都实现这个接口，
向平台注册四类东西：
- tools()   ：LangChain 工具（可附带标签、写操作标记、角色限制）
- intents() ：意图元数据（描述 / 关键词 / 示例），自动加入三路意图识别
- agents()  ：专属 Agent
- routes()  ：意图 -> Agent 路由（主 Agent 在前，降级 Agent 在后）

只声明意图、不提供 Agent / 路由时，CapabilityManager 会自动生成一个
只使用本能力工具的 Agent，并路由为 [自动 Agent, general_agent, fallback_agent]。
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

from langchain_core.tools import BaseTool

from app.agents.base import BaseAgent
from app.intent.catalog import IntentSpec


@dataclass
class CapabilityTool:
    """带元数据的工具声明。

    Attributes:
        tool: LangChain 工具
        tags: 标签（Agent 按标签挑选工具）
        write: 是否写操作（有外部副作用，可被全局关闭）
        roles: 允许使用的角色；None 表示全部角色
    """

    tool: BaseTool
    tags: Sequence[str] = ()
    write: bool = False
    roles: Optional[Sequence[str]] = None


ToolDecl = Union[BaseTool, CapabilityTool]


class Capability(ABC):
    """能力基类。子类按需覆写下列方法。"""

    #: 能力唯一名称
    name: str = ""
    #: 能力描述
    description: str = ""
    #: 版本号（插件使用）
    version: str = "1.0.0"

    def tools(self) -> List[ToolDecl]:
        return []

    def intents(self) -> List[IntentSpec]:
        return []

    def agents(self) -> List[BaseAgent]:
        return []

    def routes(self) -> Dict[str, List[str]]:
        return {}

    async def startup(self) -> None:
        """注册完成后调用（可做连接初始化、加载定时任务等）。"""

    async def shutdown(self) -> None:
        """注销或应用关闭时调用。"""
