"""dict 消息与 LangChain 消息之间的转换工具。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)


def to_langchain_messages(
    messages: List[Dict[str, Any]],
    system: Optional[str] = None,
) -> List[BaseMessage]:
    """把 [{"role": ..., "content": ...}] 转为 LangChain 消息列表。"""
    result: List[BaseMessage] = []
    if system:
        result.append(SystemMessage(content=system))
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "") or ""
        if role == "assistant":
            result.append(AIMessage(content=content))
        elif role == "system":
            result.append(SystemMessage(content=content))
        else:
            result.append(HumanMessage(content=content))
    return result


def message_text(message: Any) -> str:
    """提取消息中的纯文本（兼容 str 与 Anthropic 风格的内容块列表）。"""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content or "")
