"""核心能力：原有的三个 Agent + 通用 Agent，以及原有的三条意图路由。"""

from __future__ import annotations

from typing import Dict, List

from app.agents.base import BaseAgent
from app.agents.knowledge import RetrievalAgent
from app.agents.langchain_agent import GeneralAgent
from app.agents.safety import FallbackAgent
from app.agents.summary import SummarizationAgent
from app.capabilities.base import Capability
from app.intent.catalog import IntentSpec


class CoreCapability(Capability):
    """核心能力：四个基础 Agent 与三条默认路由。必须最先注册，其他能力的路由依赖它。"""
    name = "core"
    description = "核心对话：知识问答、摘要、闲聊、通用任务"

    def agents(self) -> List[BaseAgent]:
        return [RetrievalAgent(), SummarizationAgent(), FallbackAgent(), GeneralAgent()]

    def intents(self) -> List[IntentSpec]:
        return [
            IntentSpec(
                name="general_task",
                description="需要调用插件或外部工具完成的操作类任务，或无法归入其他意图的任务",
                keywords=["插件", "调用工具", "用工具", "执行一下"],
                examples=[
                    "调用插件帮我处理一下这个任务",
                    "用可用的工具帮我完成这件事",
                    "Use the available tools to get this done",
                ],
            )
        ]

    def routes(self) -> Dict[str, List[str]]:
        return {
            # knowledge_retrieval → RetrievalAgent（主），FallbackAgent（备）
            "knowledge_retrieval": ["retrieval_agent", "fallback_agent"],
            # summarize → SummarizationAgent（主），FallbackAgent（备）
            "summarize": ["summarize_agent", "fallback_agent"],
            # small_talk → RetrievalAgent 兜底处理简单闲聊，FallbackAgent 兜底
            "small_talk": ["retrieval_agent", "fallback_agent"],
            # general_task → 通用 Agent（可用全部有权限的工具），FallbackAgent 兜底
            "general_task": ["general_agent", "fallback_agent"],
        }
