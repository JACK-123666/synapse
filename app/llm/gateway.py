"""统一 LLM 客户端封装。

对外只提供三类能力：
- chat(messages, **kwargs) -> str   对话补全
- embed(text) -> list[float]        文本向量化
- embed_batch(texts)                批量文本向量化

实现全部委托给 app.llm.factory 创建的 LangChain ChatModel / Embeddings，
因此 provider（OpenAI / DeepSeek / Claude）、超时、重试、连接复用只有一条路径，
不会再出现两套并行实现行为不一致的问题。

配置不在这里：运行时切换与读取由 app.llm.config.LLMRuntimeConfig 负责，
本类只是它 + 模型工厂的一个便捷调用入口。
调用失败统一抛 LLMError，由上层捕获并触发降级兜底。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.llm.config import LLMRuntimeConfig, get_llm_config

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """LLM 调用异常，用于触发上层降级。"""


class LLMClient:
    """统一 LLM 客户端（单一 LangChain 后端）。

    Usage:
        client = get_llm_client()
        reply = await client.chat([{"role": "user", "content": "你好"}])
        vector = await client.embed("你好")

        # 运行时切换（管理员接口）
        client.switch_model(provider="openai", api_key="sk-...")
    """

    def __init__(self, config: Optional[LLMRuntimeConfig] = None) -> None:
        self.config: LLMRuntimeConfig = config or get_llm_config()

    # ---- 运行时配置（透传给 config 对象）----

    def switch_model(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.config.switch(
            provider=provider, model=model, api_key=api_key, base_url=base_url
        )

    def get_config(self) -> Dict[str, Any]:
        """返回当前生效的 LLM 配置（含 runtime 覆盖）。"""
        return self.config.snapshot()

    def reset_runtime(self) -> None:
        """清空所有运行时覆盖，回退到 settings。"""
        self.config.reset()

    # ---- 对话补全 ----

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system: Optional[str] = None,
    ) -> str:
        """对话补全，返回 LLM 生成的文本。

        Args:
            messages: [{"role": "user", "content": "..."}]
            temperature: 采样温度
            max_tokens: 最大生成 token 数
            system: 系统提示词（可选）

        Raises:
            LLMError: 调用失败，由上层捕获后触发降级。
        """
        from app.llm.factory import get_chat_model
        from app.llm.messages import message_text, to_langchain_messages

        try:
            model = get_chat_model(temperature=temperature, max_tokens=max_tokens)
            result = await model.ainvoke(to_langchain_messages(messages, system))
            return message_text(result)
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM 对话请求失败: %s", exc)
            raise LLMError(f"LLM 调用失败: {exc}") from exc

    # ---- 文本向量化 ----

    async def embed(self, text: str) -> List[float]:
        """文本向量化。

        统一走 EMBEDDING_PROVIDER：openai 兼容接口，或 local（ChromaDB 内置模型）。
        """
        from app.llm.factory import get_embeddings

        try:
            return await get_embeddings().aembed_query(text)
        except Exception as exc:  # noqa: BLE001
            logger.error("Embedding 请求失败: %s", exc)
            raise LLMError(f"Embedding 失败: {exc}") from exc

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量文本向量化，返回顺序与输入一致。"""
        if not texts:
            return []
        from app.llm.factory import get_embeddings

        try:
            return await get_embeddings().aembed_documents(list(texts))
        except Exception as exc:  # noqa: BLE001
            logger.error("批量 Embedding 请求失败: %s", exc)
            raise LLMError(f"批量 Embedding 失败: {exc}") from exc


# 全局单例

_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """获取全局 LLM 客户端单例。"""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
