"""兜底回复 Agent。

所有路由降级的最后保障，无论什么场景都能返回合法回复。
当主 Agent 不可用或执行失败时，由路由调度器自动降级至此 Agent。

设计原则：
- 绝不抛出异常（100% 可用）
- 返回礼貌性回复，引导用户重试或转人工
- 不依赖任何外部服务
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List

from app.agents.base import AgentContext, AgentResponse, BaseAgent

logger = logging.getLogger(__name__)


class FallbackAgent(BaseAgent):
    """兜底回复 Agent。

    作为降级链的最后一环，确保用户总能得到回复。
    返回预设的礼貌性文本，可根据上下文稍作变化。
    """

    agent_id: str = "fallback_agent"
    description: str = "兜底回复 Agent，保证系统始终可响应"

    #: 预设兜底回复模板，随机选取增加自然感
    _FALLBACK_REPLIES: List[str] = [
        "抱歉，我目前无法处理您的请求。请稍后重试，或换个方式描述您的问题。",
        "我遇到了一些技术问题，暂时无法完成您的请求。请稍等片刻后再试。",
        "很抱歉，当前服务出现暂时波动。您可以尝试重新提问，或者联系人工客服获取帮助。",
        "我暂时无法处理这个问题。请尝试用更具体的关键词重新描述，或等待几秒后重试。",
        "系统繁忙，我暂时无法回复。请简化您的问题后重试，谢谢理解。",
    ]

    async def execute(self, context: AgentContext) -> AgentResponse:
        """执行兜底回复。

        不依赖任何外部服务，永远返回合法响应。

        Args:
            context: 执行上下文（大部分信息不会实际使用）

        Returns:
            礼貌性兜底回复
        """
        logger.info(
            "兜底 Agent: session=%s 意图=%s 触发兜底回复",
            context.session_id, context.intent,
        )

        reply = random.choice(self._FALLBACK_REPLIES)
        metadata: Dict[str, Any] = {
            "mode": "fallback",
            "intent": context.intent,
        }

        return AgentResponse(reply=reply, metadata=metadata)

    async def health_check(self) -> bool:
        """兜底 Agent 永远健康。"""
        return True
