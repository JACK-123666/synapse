"""内置能力。

每个能力实现 Capability 接口，启动时由 CapabilityManager 注册：
- core       核心对话（知识问答 / 摘要 / 闲聊 / 通用任务）
- knowledge  RAG 知识库
- memory     记忆检索
- web        网页抓取与联网搜索
- repo       代码仓库助手
- scheduler  定时任务
"""

from __future__ import annotations

from typing import List

from app.capabilities.base import Capability, CapabilityTool


def builtin_capabilities() -> List[Capability]:
    """按注册顺序返回内置能力（core 必须最先注册，其他能力的路由依赖它的 Agent）。"""
    from app.capabilities.core import CoreCapability
    from app.capabilities.knowledge import KnowledgeCapability
    from app.capabilities.memory import MemoryCapability
    from app.capabilities.repo import RepoCapability
    from app.capabilities.scheduler import SchedulerCapability
    from app.capabilities.web import WebCapability

    return [
        CoreCapability(),
        KnowledgeCapability(),
        MemoryCapability(),
        WebCapability(),
        RepoCapability(),
        SchedulerCapability(),
    ]


__all__ = ["Capability", "CapabilityTool", "builtin_capabilities"]
