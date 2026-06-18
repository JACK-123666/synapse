"""路由调度模块。

维护 Agent 注册信息、权重和健康状态。
根据意图和权重动态分发任务，失败时自动降级到备用 Agent 或兜底逻辑。
"""

from app.router.registry import AgentRegistry, get_agent_registry
from app.router.dispatcher import TaskDispatcher, get_task_dispatcher

__all__ = [
    "AgentRegistry",
    "get_agent_registry",
    "TaskDispatcher",
    "get_task_dispatcher",
]
