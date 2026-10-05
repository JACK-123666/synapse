"""Agent 执行器模块。

统一接口 BaseAgent.execute(context) -> response，可插拔扩展。
内置 Agent：
- RetrievalAgent：知识检索
- SummarizationAgent：摘要压缩
- FallbackAgent：兜底回复
- GeneralAgent：通用助手（可使用全部有权限的工具）

基于 LangChain 的 Agent 继承 LangChainAgent，由 create_agent 组装模型与工具。
"""

from app.agents.base import AgentContext, AgentResponse, BaseAgent
from app.agents.langchain_agent import GeneralAgent, LangChainAgent, PreparedRun
from app.agents.knowledge import RetrievalAgent
from app.agents.summary import SummarizationAgent
from app.agents.safety import FallbackAgent

__all__ = [
    "AgentContext",
    "AgentResponse",
    "BaseAgent",
    "LangChainAgent",
    "GeneralAgent",
    "PreparedRun",
    "RetrievalAgent",
    "SummarizationAgent",
    "FallbackAgent",
]
