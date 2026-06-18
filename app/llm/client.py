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

    通过 settings.llm_provider 切换 OpenAI / Claude 后端。
    内部维护一个 httpx.AsyncClient 连接池，复用连接提升性能。

    Usage:
        client = LLMClient(settings)
        reply = await client.chat([{"role": "user", "content": "你好"}])
        vector = await client.embed("你好")
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings: Settings = settings or get_settings()
        self._http: Optional[httpx.AsyncClient] = None

    # ----------------------------------------------------------
    # 生命周期管理
    # ----------------------------------------------------------

    async def connect(self) -> None:
        """初始化 HTTP 连接池。"""
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(self._settings.llm_timeout),
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            )
            logger.info(
                "LLM 客户端已初始化，provider=%s", self._settings.llm_provider
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

    # ----------------------------------------------------------
    # 对话补全
    # ----------------------------------------------------------

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
        provider = self._settings.llm_provider.lower()
        try:
            if provider == "claude":
                return await self._chat_claude(
                    messages, temperature, max_tokens, system
                )
            elif provider == "deepseek":
                return await self._chat_openai(
                    messages, temperature, max_tokens, system,
                    base_url=self._settings.deepseek_base_url,
                    model=self._settings.deepseek_model,
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
            "model": model or self._settings.llm_model,
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = self._openai_headers()
        url_base = base_url or self._settings.llm_base_url
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
            "model": self._settings.claude_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            payload["system"] = system

        headers = self._claude_headers()
        url = f"{self._settings.anthropic_base_url.rstrip('/')}/messages"

        resp = await self.http.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        # Claude 返回 content 是列表，取第一段文本
        return data["content"][0]["text"]

    # ----------------------------------------------------------
    # 文本向量化
    # ----------------------------------------------------------

    async def embed(self, text: str) -> List[float]:
        """文本向量化，返回 embedding 浮点向量。

        统一使用 OpenAI 兼容的 embeddings 接口（Claude 暂无官方 embedding 接口，
        实际部署时可配置 OpenAI embedding 或第三方服务）。

        Args:
            text: 待向量化的文本

        Returns:
            embedding 向量（list[float]）

        Raises:
            LLMError: 向量化失败时抛出
        """
        try:
            payload = {
                "model": self._settings.embedding_model,
                "input": text,
            }
            headers = self._openai_headers()
            url = f"{self._settings.llm_base_url.rstrip('/')}/embeddings"

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
                "model": self._settings.embedding_model,
                "input": texts,
            }
            headers = self._openai_headers()
            url = f"{self._settings.llm_base_url.rstrip('/')}/embeddings"

            resp = await self.http.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            # 按 index 排序确保顺序一致
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_data]
        except Exception as exc:
            logger.error("批量 Embedding 请求失败: %s", exc)
            raise LLMError(f"批量 Embedding 失败: {exc}") from exc

    # ----------------------------------------------------------
    # 内部辅助
    # ----------------------------------------------------------

    def _openai_headers(self) -> Dict[str, str]:
        """构建 OpenAI 请求头。"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._settings.llm_api_key}",
        }

    def _claude_headers(self) -> Dict[str, str]:
        """构建 Anthropic Claude 请求头。"""
        return {
            "Content-Type": "application/json",
            "x-api-key": self._settings.llm_api_key,
            "anthropic-version": "2023-06-01",
        }


# ============================================================
# 全局单例
# ============================================================

_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """获取全局 LLM 客户端单例。"""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
