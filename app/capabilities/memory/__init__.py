"""记忆检索能力：查询长期记忆摘要与当前会话记录。"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

from langchain_core.tools import tool

from app.agents.base import AgentContext, BaseAgent
from app.agents.langchain_agent import LangChainAgent, format_recall
from app.capabilities.base import Capability, CapabilityTool, ToolDecl
from app.core.context import get_request_context
from app.intent.catalog import IntentSpec
from app.memory.archive import get_long_term_memory
from app.memory.recent import get_short_term_memory


def _fmt_time(ts: Any) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(ts)))
    except (TypeError, ValueError):
        return "未知时间"


@tool("memory_search", response_format="content_and_artifact")
async def memory_search_tool(query: str, top_k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
    """检索与问题相关的历史对话摘要（长期记忆），用于回答“我之前问过什么”“我们上次聊到哪”等问题。

    Args:
        query: 要回忆的内容或关键词
        top_k: 返回条数，默认 5
    """
    ctx = get_request_context()
    results = await get_long_term_memory().recall(
        query_text=query, top_k=max(1, min(int(top_k or 5), 20)), user_id=ctx.memory_user_id
    )
    if not results:
        return "没有找到相关的历史记忆。", []
    lines = [
        f"[{i}] ({_fmt_time(r['metadata'].get('timestamp'))}, 相似度 {r['score']:.2f}) {r['text']}"
        for i, r in enumerate(results, 1)
    ]
    return "\n".join(lines), results


@tool("memory_recent_messages")
async def memory_recent_tool(limit: int = 10) -> str:
    """查看当前会话最近的对话记录（短期记忆，压缩前的原始消息）。

    Args:
        limit: 返回最近多少条消息，默认 10
    """
    ctx = get_request_context()
    messages = await get_short_term_memory().get_messages(ctx.session_id)
    if not messages:
        return "当前会话还没有对话记录（或已压缩为长期记忆摘要）。"
    lines = [
        f"{'用户' if m.get('role') == 'user' else '助手'}（{_fmt_time(m.get('timestamp'))}）: {m.get('content', '')}"
        for m in messages[-max(1, min(int(limit or 10), 50)):]
    ]
    return "\n".join(lines)


@tool("memory_list_summaries")
async def memory_list_tool(limit: int = 10) -> str:
    """按时间倒序列出最近的历史对话摘要。

    Args:
        limit: 返回条数，默认 10
    """
    ctx = get_request_context()
    items = await get_long_term_memory().list_summaries(
        user_id=ctx.memory_user_id, limit=max(1, min(int(limit or 10), 50))
    )
    if not items:
        return "还没有历史对话摘要。"
    return "\n".join(
        f"- ({_fmt_time(it['metadata'].get('timestamp'))}) {it['text']}" for it in items
    )


class MemoryAgent(LangChainAgent):
    """记忆助手：回忆历史对话。"""

    agent_id = "memory_agent"
    description = "记忆检索助手"
    tool_tags = ("memory",)

    def build_system_prompt(self, context: AgentContext) -> str:
        parts: List[str] = [
            "你负责帮用户回忆历史对话。先调用 memory_search 检索相关摘要，"
            "需要原始对话时调用 memory_recent_messages；按时间线整理后回答。",
            "记忆里没有的内容要明确说不记得，不要编造。",
        ]
        recall_text = format_recall(context.long_term_recall)
        if recall_text:
            parts.append(f"\n【已召回的相关摘要】\n{recall_text}")
        if context.user_profile_context:
            parts.append(f"\n【用户画像】\n{context.user_profile_context}")
        return "\n".join(parts)


class MemoryCapability(Capability):
    name = "memory"
    description = "记忆检索"

    def tools(self) -> List[ToolDecl]:
        return [
            CapabilityTool(memory_search_tool, tags=("memory",)),
            CapabilityTool(memory_recent_tool, tags=("memory",)),
            CapabilityTool(memory_list_tool, tags=("memory",)),
        ]

    def intents(self) -> List[IntentSpec]:
        return [
            IntentSpec(
                name="memory_query",
                description="回忆或查询之前的对话与记忆，例如“我上次问过什么”“你还记得我们聊过的……吗”",
                keywords=["记得", "上次", "之前说", "之前聊", "回忆", "历史对话", "我问过", "remember", "last time"],
                examples=[
                    "我上次问你的那个问题是什么？",
                    "你还记得我们之前聊过的向量数据库吗",
                    "帮我回忆一下之前讨论的方案",
                    "What did I ask you last time?",
                ],
            )
        ]

    def agents(self) -> List[BaseAgent]:
        return [MemoryAgent()]

    def routes(self) -> Dict[str, List[str]]:
        return {"memory_query": ["memory_agent", "general_agent", "fallback_agent"]}
