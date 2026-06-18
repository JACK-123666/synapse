"""LLM 语义意图识别。

使用 few-shot prompt 调用 LLM 将用户消息分类到预定义意图列表。
返回意图标签与置信度（0.0~1.0）。

LLM 不可用时返回 None，由上层融合器降级为默认意图。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, Optional

from app.config import Settings, get_settings
from app.llm.client import LLMError, get_llm_client

logger = logging.getLogger(__name__)


class LLMIntentRecognizer:
    """基于 LLM 的语义意图识别器。

    使用结构化 few-shot prompt，要求 LLM 将用户输入分类到预定义意图，
    并以 JSON 格式返回意图标签与置信度。
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings: Settings = settings or get_settings()

    async def recognize(self, message: str) -> Optional[Dict[str, float]]:
        """识别用户消息的意图。

        Args:
            message: 用户输入消息

        Returns:
            {意图标签: 置信度} 字典，LLM 不可用时返回 None
        """
        if not message.strip():
            return None

        # 构建 few-shot prompt
        system_prompt = self._build_system_prompt()
        user_prompt = f'用户消息: "{message}"\n\n请识别意图（仅输出 JSON）：'

        try:
            llm = get_llm_client()
            raw = await llm.chat(
                messages=[{"role": "user", "content": user_prompt}],
                system=system_prompt,
                temperature=0.1,
                max_tokens=256,
            )
        except LLMError as exc:
            logger.warning("LLM 意图识别调用失败: %s", exc)
            return None

        return self._parse_response(raw, message)

    def _build_system_prompt(self) -> str:
        """构建 few-shot 意图识别的系统提示词。"""
        intents = self._settings.known_intents
        descriptions = self._settings.intent_descriptions

        intent_lines = "\n".join(
            f"- {intent}: {descriptions.get(intent, '其他')}"
            for intent in intents
        )

        return (
            f"你是一个意图识别分类器。将用户的输入分类到以下意图之一：\n"
            f"{intent_lines}\n\n"
            f"要求：\n"
            f"1. 以 JSON 格式输出，包含每个意图的置信度（0.0~1.0），置信度之和应为 1.0\n"
            f"2. 仅输出 JSON，不要添加任何额外文字\n\n"
            f"示例输出格式：\n"
            f'{{"knowledge_retrieval": 0.8, "summarize": 0.1, "small_talk": 0.1}}\n\n'
            f"示例 1：\n"
            f'用户: "你好"\n'
            f'{{"knowledge_retrieval": 0.05, "summarize": 0.05, "small_talk": 0.90}}\n\n'
            f"示例 2：\n"
            f'用户: "什么是向量数据库？"\n'
            f'{{"knowledge_retrieval": 0.85, "summarize": 0.10, "small_talk": 0.05}}\n\n'
            f"示例 3：\n"
            f'用户: "帮我总结一下"\n'
            f'{{"knowledge_retrieval": 0.10, "summarize": 0.85, "small_talk": 0.05}}'
        )

    def _parse_response(
        self, raw: str, message: str
    ) -> Optional[Dict[str, float]]:
        """解析 LLM 返回的 JSON 响应。

        Args:
            raw: LLM 原始响应文本
            message: 原始用户消息（用于日志）

        Returns:
            解析后的意图置信度字典，解析失败返回 None
        """
        # 尝试提取 JSON 块
        json_match = re.search(r'\{[^}]+\}', raw)
        if json_match:
            raw = json_match.group(0)

        try:
            result = json.loads(raw)
            # 过滤掉非预定义意图的键
            valid: Dict[str, float] = {}
            for intent in self._settings.known_intents:
                if intent in result:
                    confidence = float(result[intent])
                    valid[intent] = max(0.0, min(1.0, confidence))
            if valid:
                # 归一化
                total = sum(valid.values())
                if total > 0:
                    valid = {k: v / total for k, v in valid.items()}
                logger.debug(
                    "LLM 意图识别: '%s' -> %s", message[:50], valid
                )
                return valid
            logger.warning("LLM 意图识别: 响应中无有效意图键: %s", raw)
            return None
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("LLM 意图识别: JSON 解析失败: %s, 原始: %s", exc, raw)
            return None
