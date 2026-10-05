"""LangChain 模型工厂。

按 (provider, model, api_key, base_url, temperature, max_tokens) 创建并缓存
LangChain ChatModel；按 embedding 配置创建 Embeddings。

配置读取顺序与 LLMClient 一致：运行时覆盖（/models/switch）> .env / 环境变量。
请求级的模型覆盖（/chat 的 model 参数）从请求上下文读取，只影响当前请求，
不修改任何全局状态。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from app.core.context import get_request_context
from app.llm.config import get_llm_config

logger = logging.getLogger(__name__)

#: 未配置 API Key 时使用的占位符：让错误延迟到真正调用时（401），而不是在构造时抛出
_MISSING_KEY = "sk-missing"


@dataclass(frozen=True)
class ModelSpec:
    """一个 ChatModel 实例的完整配置（可哈希，用作缓存键）。"""

    provider: str
    model: str
    api_key: str
    base_url: str
    timeout: float
    temperature: float
    max_tokens: int


def resolve_model_spec(
    model_override: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> ModelSpec:
    """根据当前生效配置 + 请求级覆盖，解析出模型配置。"""
    config = get_llm_config()
    cfg = config.snapshot()
    override = model_override or get_request_context().model_override
    return ModelSpec(
        provider=cfg["provider"],
        model=override or cfg["model"],
        api_key=config.get("llm_api_key", "") or "",
        base_url=cfg["base_url"] or "",
        timeout=float(config.get("llm_timeout", 60)),
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )


@lru_cache(maxsize=64)
def _build_chat_model(spec: ModelSpec) -> BaseChatModel:
    """按配置构造 ChatModel（带缓存，相同配置复用同一个实例与连接池）。"""
    if spec.provider == "claude":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - 取决于可选依赖
            raise RuntimeError(
                "使用 Claude 需要可选依赖 langchain-anthropic，"
                "请执行: pip install -r requirements-extras.txt"
            ) from exc

        # Anthropic SDK 会自动拼接 /v1/messages，配置里的 /v1 需要去掉
        base_url = spec.base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[: -len("/v1")]
        return ChatAnthropic(
            model=spec.model,
            api_key=spec.api_key or _MISSING_KEY,
            base_url=base_url or None,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
            timeout=spec.timeout,
            max_retries=1,
        )

    # openai / deepseek 及其他 OpenAI 兼容服务
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=spec.model,
        api_key=spec.api_key or _MISSING_KEY,
        base_url=spec.base_url or None,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        timeout=spec.timeout,
        max_retries=1,
    )


def get_chat_model(
    temperature: float = 0.7,
    max_tokens: int = 2048,
    model_override: Optional[str] = None,
) -> BaseChatModel:
    """获取 ChatModel（自动应用运行时配置与请求级模型覆盖）。"""
    spec = resolve_model_spec(model_override, temperature, max_tokens)
    return _build_chat_model(spec)


def clear_model_cache() -> None:
    """清空模型缓存（切换配置后调用，释放旧连接）。"""
    _build_chat_model.cache_clear()
    _build_openai_embeddings.cache_clear()


# ---- Embeddings ----


class ChromaLocalEmbeddings(Embeddings):
    """把 ChromaDB 内置的本地 ONNX 模型（all-MiniLM-L6-v2）包装成 LangChain Embeddings。

    无需任何 API Key，适合 DeepSeek 等不提供 embedding 接口的场景；
    首次使用会下载约 80MB 模型文件。
    """

    def __init__(self) -> None:
        from chromadb.utils import embedding_functions

        self._fn = embedding_functions.DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(x) for x in vec] for vec in self._fn(list(texts))]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def _local_embeddings() -> ChromaLocalEmbeddings:
    return ChromaLocalEmbeddings()


@lru_cache(maxsize=8)
def _build_openai_embeddings(
    model: str, api_key: str, base_url: str, timeout: float
) -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=model,
        api_key=api_key or _MISSING_KEY,
        base_url=base_url or None,
        timeout=timeout,
        max_retries=1,
        # 非 OpenAI 的兼容服务不接受 token id 输入，关闭按 token 切分
        check_embedding_ctx_length=False,
    )


def get_embeddings() -> Embeddings:
    """获取 Embeddings。

    embedding_provider=local 时使用本地模型；否则走 OpenAI 兼容接口，
    密钥 / 地址优先用 EMBEDDING_*，留空回退 LLM_*（与原实现一致）。
    """
    config = get_llm_config()
    provider = str(config.get("embedding_provider", "openai") or "openai").lower()
    if provider == "local":
        return _local_embeddings()
    api_key = config.get("embedding_api_key") or config.get("llm_api_key", "") or ""
    base_url = config.get("embedding_base_url") or config.get("llm_base_url", "") or ""
    return _build_openai_embeddings(
        config.get("embedding_model"),
        api_key,
        base_url,
        float(config.get("llm_timeout", 60)),
    )
