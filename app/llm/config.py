"""LLM 运行时配置。

把「当前生效的 LLM 配置」从调用客户端里拆出来：模型工厂只需要读配置，
不需要一个能发请求的客户端，两边的依赖方向因此变成单向。

读取优先级：运行时覆盖（POST /models/switch 写入）> .env / 环境变量。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMRuntimeConfig:
    """当前生效的 LLM 配置。"""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings: Settings = settings or get_settings()
        self._runtime: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        """读取配置：runtime > settings > default。

        deepseek_model / claude_model 在 runtime 中缺省时回退到 llm_model，
        这样 switch_model(model=...) 对所有 provider 都生效。
        """
        if key in self._runtime:
            return self._runtime[key]
        if key in ("deepseek_model", "claude_model") and "llm_model" in self._runtime:
            return self._runtime["llm_model"]
        return getattr(self._settings, key, default)

    def switch(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """运行时切换提供商 / 模型 / 密钥 / base_url。

        只更新传入的字段；传空字符串 "" 表示清空该覆盖，回退到 settings。
        """
        overrides = {
            "llm_provider": provider,
            "llm_model": model,
            "llm_api_key": api_key,
            "llm_base_url": base_url,
        }
        for key, value in overrides.items():
            if value is None:
                continue
            if value == "":
                self._runtime.pop(key, None)
            else:
                self._runtime[key] = value
        logger.info("LLM 运行时切换: %s", self.snapshot())
        return self.snapshot()

    def reset(self) -> None:
        """清空所有运行时覆盖，回退到 settings。"""
        self._runtime.clear()
        logger.info("LLM 运行时配置已重置")

    def snapshot(self) -> Dict[str, Any]:
        """当前生效配置的摘要（密钥只暴露前缀）。"""
        provider = str(self.get("llm_provider", "openai") or "openai").lower()
        api_key = self.get("llm_api_key", "") or ""
        data: Dict[str, Any] = {
            "provider": provider,
            "api_key_prefix": api_key[:8] + "..." if api_key else "(empty)",
            "timeout_s": self.get("llm_timeout", 60),
            "embedding_api_key_set": bool(self.get("embedding_api_key")),
        }
        if provider == "deepseek":
            data["model"] = self.get("deepseek_model")
            data["base_url"] = self.get("deepseek_base_url")
        elif provider == "claude":
            data["model"] = self.get("claude_model")
            data["base_url"] = self.get("anthropic_base_url")
        else:
            data["model"] = self.get("llm_model")
            data["base_url"] = self.get("llm_base_url")
        return data


_config: Optional[LLMRuntimeConfig] = None


def get_llm_config() -> LLMRuntimeConfig:
    """获取全局 LLM 配置单例。"""
    global _config
    if _config is None:
        _config = LLMRuntimeConfig()
    return _config
