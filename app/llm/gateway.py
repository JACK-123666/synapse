"""统一 LLM 客户端封装。

支持两种后端：
- OpenAI（兼容 OpenAI 协议的自建服务 / 代理）
- Claude（Anthropic 官方 API）

统一接口：
- chat(messages, **kwargs) -> str：对话补全
- embed(text) -> list[float]：文本向量化（Embedding）

所有调用基于 httpx.AsyncClient，完全异步，支持超时控制。
当 LLM 不可用时抛出 LLMError，由上层捕获并触发降级兜底。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """LLM 调用异常，用于触发上层降级。"""


class LLMClient:
    """统一 LLM 客户端。

    通过 settings.llm_provider 切换 OpenAI / Claude / DeepSeek 后端。
    支持运行时通过 switch_model() 切换模型，无需重启。

    Usage:
        client = LLMClient(settings)
        reply = await client.chat([{"role": "user", "content": "你好"}])
        vector = await client.embed("你好")

        # 运行时切换
        client.switch_model(provider="openai", api_key="sk-...")
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings: Settings = settings or get_settings()
        self._http: Optional[httpx.AsyncClient] = None
        # 运行时覆盖（优先级高于 settings），由 API /models/switch 修改
        self._runtime: Dict[str, Any] = {}

    # 运行时配置（API /models/switch 调用）

    def _rc(self, key: str, default: Any = None) -> Any:
        """读取配置：runtime > settings > default。

        deepseek_model / claude_model 在 runtime 中缺省时回退到 llm_model，
        这样 switch_model(model=...) 对所有 provider 都生效。
        """
        if key in self._runtime:
            return self._runtime[key]
        if key in ("deepseek_model", "claude_model") and "llm_model" in self._runtime:
            return self._runtime["llm_model"]
        return getattr(self._settings, key, default)

    def switch_model(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """运行时切换 LLM 提供商 / 模型 / 密钥 / base_url。

        只更新传入的字段，未传的保留当前值（runtime 或 settings）。
        传空字符串 "" 表示清空 runtime 覆盖，回退到 settings。
        """
        overrides = {
            "llm_provider": provider,
            "llm_model": model,
            "llm_api_key": api_key,
            "llm_base_url": base_url,
        }
        for k, v in overrides.items():
            if v is not None:
                if v == "":
                    self._runtime.pop(k, None)
                else:
                    self._runtime[k] = v
        logger.info("LLM 运行时切换: %s", self.get_config())
        return self.get_config()

    def get_config(self) -> Dict[str, Any]:
        """返回当前生效的 LLM 配置（含 runtime 覆盖）。"""
        provider = self._rc("llm_provider", "openai").lower()
        base = {
            "provider": provider,
            "api_key_prefix": self._rc("llm_api_key", "")[:8] + "..." if self._rc("llm_api_key") else "(empty)",
            "timeout_s": self._rc("llm_timeout", 60),
            "embedding_api_key_set": bool(self._rc("embedding_api_key")),
        }
        if provider == "deepseek":
            base["model"] = self._rc("deepseek_model")
            base["base_url"] = self._rc("deepseek_base_url")
        elif provider == "claude":
            base["model"] = self._rc("claude_model")
            base["base_url"] = self._rc("anthropic_base_url")
        else:
            base["model"] = self._rc("llm_model")
            base["base_url"] = self._rc("llm_base_url")
        return base

    def reset_runtime(self) -> None:
        """清空所有运行时覆盖，回退到 settings。"""
        self._runtime.clear()
        logger.info("LLM 运行时配置已重置")

    # 生命周期管理

    async def connect(self) -> None:
        """初始化 HTTP 连接池。"""
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(self._settings.llm_timeout),
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            )
            logger.info(
                "LLM 客户端已初始化，provider=%s model=%s",
                self._rc("llm_provider"), self.get_config().get("model"),
            )

    async def close(self) -> None:
        """关闭 HTTP 连接池。"""
        if self._http is not None:
            await self._http.aclose()
            self._http = None
            logger.info("LLM 客户端已关闭")

    @property
    def http(self) -> httpx.AsyncClient:
        """获取已初始化的 HTTP 客户端，未初始化时抛出异常。"""
        if self._http is None:
            raise LLMError("LLM 客户端未初始化，请先调用 connect()")
        return self._http

    # 对话补全

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system: Optional[str] = None,
    ) -> str:
        """对话补全，返回 LLM 生成的文本。

        Args:
            messages: 对话消息列表，格式为 [{"role": "user", "content": "..."}]
            temperature: 采样温度，越高越发散
            max_tokens: 最大生成 token 数
            system: 系统提示词（可选）

        Returns:
            LLM 生成的回复文本

        Raises:
            LLMError: 调用失败时抛出，上层捕获后触发降级
        """
        provider = self._rc("llm_provider", "openai").lower()
        try:
            if provider == "claude":
                return await self._chat_claude(
                    messages, temperature, max_tokens, system
                )
            elif provider == "deepseek":
                return await self._chat_openai(
                    messages, temperature, max_tokens, system,
                    base_url=self._rc("deepseek_base_url"),
                    model=self._rc("deepseek_model"),
                )
            else:
                return await self._chat_openai(
                    messages, temperature, max_tokens, system
                )
        except httpx.HTTPError as exc:
            logger.error("LLM 对话请求网络错误: %s", exc)
            raise LLMError(f"LLM 网络错误: {exc}") from exc
        except Exception as exc:
            logger.error("LLM 对话请求失败: %s", exc)
            raise LLMError(f"LLM 调用失败: {exc}") from exc

    async def _chat_openai(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        system: Optional[str],
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """OpenAI 兼容协议的对话补全（也用于 DeepSeek 等兼容服务）。"""
        full_messages: List[Dict[str, str]] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        payload: Dict[str, Any] = {
            "model": model or self._rc("llm_model"),
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = self._openai_headers()
        url_base = base_url or self._rc("llm_base_url")
        url = f"{url_base.rstrip('/')}/chat/completions"

        resp = await self.http.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def _chat_claude(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        system: Optional[str],
    ) -> str:
        """Anthropic Claude 协议的对话补全。"""
        # Claude 的 system 消息是顶层参数，不在 messages 中
        payload: Dict[str, Any] = {
            "model": self._rc("claude_model"),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            payload["system"] = system

        headers = self._claude_headers()
        url = f"{self._rc('anthropic_base_url').rstrip('/')}/messages"

        resp = await self.http.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        # Claude 返回 content 是列表，取第一段文本
        return data["content"][0]["text"]

    # 文本向量化

    async def embed(self, text: str) -> List[float]:
        """文本向量化，返回 embedding 浮点向量。

        统一使用 OpenAI 兼容的 embeddings 接口。
        当 provider 为 deepseek/claude 且未配置 EMBEDDING_API_KEY 时，
        会在日志警告后尝试回退（可能 401），建议配置独立的 embedding 服务。
        """
        try:
            provider = self._rc("llm_provider", "openai").lower()
            embed_key = self._rc("embedding_api_key", "")
            if not embed_key and provider in ("deepseek", "claude"):
                logger.warning(
                    "Embedding: provider=%s 不支持 embedding 且未配置 EMBEDDING_API_KEY。"
                    "将用 llm_api_key + llm_base_url 尝试（可能因 key 不匹配而 401）。"
                    "建议在 .env 设置 EMBEDDING_API_KEY / EMBEDDING_BASE_URL。",
                    provider,
                )
            payload = {
                "model": self._rc("embedding_model"),
                "input": text,
            }
            headers = self._embed_headers()
            url = f"{self._embed_base_url().rstrip('/')}/embeddings"

            resp = await self.http.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]
        except Exception as exc:
            logger.error("Embedding 请求失败: %s", exc)
            raise LLMError(f"Embedding 失败: {exc}") from exc

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量文本向量化。

        Args:
            texts: 待向量化的文本列表

        Returns:
            embedding 向量列表，与输入顺序一致
        """
        try:
            payload = {
                "model": self._rc("embedding_model"),
                "input": texts,
            }
            headers = self._embed_headers()
            url = f"{self._embed_base_url().rstrip('/')}/embeddings"

            resp = await self.http.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            # 按 index 排序确保顺序一致
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_data]
        except Exception as exc:
            logger.error("批量 Embedding 请求失败: %s", exc)
            raise LLMError(f"批量 Embedding 失败: {exc}") from exc

    # 内部辅助

    def _openai_headers(self) -> Dict[str, str]:
        """构建 OpenAI 请求头。"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._rc('llm_api_key', '')}",
        }

    def _embed_base_url(self) -> str:
        """Embedding 专用 base_url，留空回退 llm_base_url。"""
        return self._rc("embedding_base_url") or self._rc("llm_base_url", "")

    def _embed_headers(self) -> Dict[str, str]:
        """Embedding 专用请求头，api_key 留空回退 llm_api_key。

        DeepSeek 不支持 embedding：LLM_PROVIDER=deepseek 时应配置
        EMBEDDING_API_KEY / EMBEDDING_BASE_URL 指向支持 embedding 的服务。
        """
        key = self._rc("embedding_api_key") or self._rc("llm_api_key", "")
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }

    def _claude_headers(self) -> Dict[str, str]:
        """构建 Anthropic Claude 请求头。"""
        return {
            "Content-Type": "application/json",
            "x-api-key": self._rc("llm_api_key", ""),
            "anthropic-version": "2023-06-01",
        }


# 全局单例

_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """获取全局 LLM 客户端单例。"""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
