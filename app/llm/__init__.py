"""LLM 模块。

- config：运行时配置（运行时覆盖 > 环境变量），供模型工厂读取
- gateway：统一调用入口（chat / embed），全部委托给 LangChain 实现
- factory：按配置创建并缓存 ChatModel / Embeddings
- messages：dict 消息与 LangChain 消息互转
"""

from app.llm.config import LLMRuntimeConfig, get_llm_config
from app.llm.gateway import LLMClient, LLMError, get_llm_client

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMRuntimeConfig",
    "get_llm_client",
    "get_llm_config",
]
