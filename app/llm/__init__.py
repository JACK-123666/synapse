"""LLM 客户端模块。

统一封装 Claude / OpenAI API，支持通过配置切换。
所有调用均为异步（基于 httpx.AsyncClient）。
"""

from app.llm.gateway import LLMClient, get_llm_client

__all__ = ["LLMClient", "get_llm_client"]
