"""Agent 执行器模块。

统一接口 BaseAgent.execute(context) -> response，可插拔扩展。
内置三类 Agent：
- RetrievalAgent：知识检索
- SummarizationAgent：摘要压缩
- FallbackAgent：兜底回复
"""

from app.agents.base import AgentContext, AgentResponse, BaseAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.summarize_agent import SummarizationAgent
from app.agents.fallback_agent import FallbackAgent

__all__ = [
    "AgentContext",
    "AgentResponse",
    "BaseAgent",
    "RetrievalAgent",
    "SummarizationAgent",
    "FallbackAgent",
]
