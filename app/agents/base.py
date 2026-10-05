"""Agent 抽象基类与数据模型。

定义所有 Agent 的统一接口：
    async execute(context: AgentContext) -> AgentResponse

AgentContext 携带执行所需的全部上下文信息：
- 会话信息（session_id, user_id）
- 当前用户消息与识别意图
- 短期记忆（最近对话）
- 长期记忆召回（相似历史摘要）
- 用户画像上下文

AgentResponse 包含回复文本和元数据（如检索到的知识来源等）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentContext(BaseModel):
    """Agent 执行上下文。

    封装一次请求所需的全部信息，由 API 网关在调用 Agent 前组装。

    Attributes:
        session_id: 会话 ID
        user_id: 用户 ID（可选）
        message: 用户当前消息文本
        intent: 识别出的意图标签
        short_term_memory: 短期记忆消息列表（最近 N 轮）
        long_term_recall: 长期记忆召回的相似历史摘要列表
        user_profile_context: 用户画像上下文文本（注入 prompt）
    """

    session_id: str
    user_id: Optional[str] = None
    message: str
    intent: str = "unknown"
    short_term_memory: List[Dict[str, Any]] = Field(default_factory=list)
    long_term_recall: List[Dict[str, Any]] = Field(default_factory=list)
    user_profile_context: str = ""
    web_search: bool = True  # 是否允许联网搜索
    # 本次请求临时指定的模型（只影响当前请求，不修改全局配置）
    model_override: Optional[str] = None
    # 当前用户角色，用于工具权限过滤
    user_role: str = "admin"
    # 额外限定的工具名列表（None 表示不额外限定）
    allowed_tools: Optional[List[str]] = None
    # 意图识别置信度（流式接口回传用）
    intent_confidence: float = 0.0

    model_config = {"arbitrary_types_allowed": True}


class AgentResponse(BaseModel):
    """Agent 执行响应。

    Attributes:
        reply: 回复文本
        metadata: 附加元数据（如检索来源、Token 使用等）
    """

    reply: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class BaseAgent(ABC):
    """Agent 抽象基类。

    所有 Agent 必须实现 execute 方法，接收 AgentContext，返回 AgentResponse。
    可覆写 stream 方法实现流式输出；健康状态由 observability.health 的
    异常检测器根据实际请求耗时统一判定，Agent 无需自行上报。

    Attributes:
        agent_id: Agent 唯一标识，用于路由注册、指标上报、异常检测
        description: Agent 描述
    """

    agent_id: str = "base_agent"
    description: str = "Base Agent"

    @abstractmethod
    async def execute(self, context: AgentContext) -> AgentResponse:
        """执行 Agent 逻辑。

        Args:
            context: 执行上下文

        Returns:
            Agent 响应

        Raises:
            异常应由上层 dispatcher 捕获并触发降级
        """
        ...

    async def stream(self, context: AgentContext) -> AsyncIterator[Dict[str, Any]]:
        """流式执行，默认实现：执行完整逻辑后一次性输出。

        事件格式：
            {"type": "token", "content": "..."}
            {"type": "tool_start", "name": "...", "args": {...}}
            {"type": "tool_end", "name": "...", "output": "..."}
            {"type": "final", "response": AgentResponse}
        """
        response = await self.execute(context)
        if response.reply:
            yield {"type": "token", "content": response.reply}
        yield {"type": "final", "response": response}
