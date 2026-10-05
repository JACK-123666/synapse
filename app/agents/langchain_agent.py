"""基于 LangChain 1.x 的 Agent 基类。

在 BaseAgent 的统一接口（execute / stream）之下，用 langchain.agents.create_agent
组装 "模型 + 工具 + 系统提示词"，由模型自主决定是否调用工具（Function Calling）。

子类通常只需要覆写：
- tool_tags：从工具注册表挑选哪些标签的工具（() 表示不用工具，None 表示全部可用工具）
- build_system_prompt()：系统提示词
- 或者覆写 prepare() 做更复杂的预处理（如先检索知识库再拼 prompt）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple

from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from app.agents.base import AgentContext, AgentResponse, BaseAgent
from app.llm.factory import get_chat_model
from app.llm.gateway import LLMError
from app.llm.messages import message_text
from app.tools.registry import get_tool_registry

logger = logging.getLogger(__name__)


@dataclass
class PreparedRun:
    """一次 Agent 执行前准备好的输入。"""

    system_prompt: str
    tools: List[BaseTool]
    metadata: Dict[str, Any] = field(default_factory=dict)


def format_recall(recall: List[Dict[str, Any]]) -> str:
    """格式化长期记忆召回摘要。"""
    if not recall:
        return ""
    parts = []
    for i, item in enumerate(recall, 1):
        text = item.get("text", "")
        score = item.get("score", 0)
        if text:
            parts.append(f"[{i}] (相似度: {score:.2f}) {text}")
    return "\n".join(parts)


def collect_tool_info(messages: Sequence[BaseMessage]) -> Dict[str, Any]:
    """从 Agent 产生的消息中提取工具调用记录与工具产物（artifact）。"""
    tool_calls: List[Dict[str, Any]] = []
    artifacts: Dict[str, List[Any]] = {}
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({"name": tc.get("name"), "args": tc.get("args", {})})
        elif isinstance(msg, ToolMessage):
            artifact = getattr(msg, "artifact", None)
            if artifact is not None and msg.name:
                artifacts.setdefault(msg.name, []).append(artifact)
    return {"tool_calls": tool_calls, "artifacts": artifacts}


def _final_reply(messages: Sequence[BaseMessage]) -> str:
    """取最后一条有文本内容的 AI 消息作为最终回复。"""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            text = message_text(msg)
            if text.strip():
                return text
    return ""


class LangChainAgent(BaseAgent):
    """LangChain Agent 基类。"""

    agent_id: str = "langchain_agent"
    description: str = "LangChain Agent"

    #: 工具标签；() 不使用工具，None 使用当前用户可用的全部工具
    tool_tags: Optional[Tuple[str, ...]] = ()
    #: 是否允许写操作工具
    include_write_tools: bool = True
    temperature: float = 0.5
    max_tokens: int = 2048
    #: create_agent 图的最大步数（防止工具调用死循环）
    recursion_limit: int = 16

    # ---- 可覆写的钩子 ----

    def build_system_prompt(self, context: AgentContext) -> str:
        parts: List[str] = [f"你是 Synapse 智能助手中的「{self.description}」。"]
        recall_text = format_recall(context.long_term_recall)
        if recall_text:
            parts.append(f"\n【历史相关摘要】\n{recall_text}")
        if context.user_profile_context:
            parts.append(f"\n【用户画像】\n{context.user_profile_context}")
        return "\n".join(parts)

    def select_tools(self, context: AgentContext) -> List[BaseTool]:
        if self.tool_tags is not None and len(self.tool_tags) == 0:
            return []
        return get_tool_registry().tools_for(
            role=context.user_role,
            tags=self.tool_tags,
            names=context.allowed_tools,
            include_write=self.include_write_tools,
        )

    def build_messages(self, context: AgentContext) -> List[BaseMessage]:
        """短期记忆 + 当前消息。"""
        messages: List[BaseMessage] = []
        for msg in context.short_term_memory:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "assistant":
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))
        messages.append(HumanMessage(content=context.message))
        return messages

    async def prepare(self, context: AgentContext) -> PreparedRun:
        return PreparedRun(
            system_prompt=self.build_system_prompt(context),
            tools=self.select_tools(context),
        )

    def build_metadata(
        self,
        context: AgentContext,
        prepared: PreparedRun,
        new_messages: Sequence[BaseMessage],
    ) -> Dict[str, Any]:
        info = collect_tool_info(new_messages)
        metadata: Dict[str, Any] = {"mode": self.agent_id, **prepared.metadata}
        metadata["tool_calls"] = info["tool_calls"]
        metadata["function_calling"] = bool(info["tool_calls"])
        metadata["_artifacts"] = info["artifacts"]
        return metadata

    # ---- 执行 ----

    def _model(self, context: AgentContext):
        return get_chat_model(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            model_override=context.model_override,
        )

    async def execute(self, context: AgentContext) -> AgentResponse:
        prepared = await self.prepare(context)
        inputs = self.build_messages(context)
        model = self._model(context)
        try:
            if prepared.tools:
                agent = create_agent(
                    model, tools=prepared.tools, system_prompt=prepared.system_prompt
                )
                result = await agent.ainvoke(
                    {"messages": inputs},
                    config={"recursion_limit": self.recursion_limit},
                )
                all_messages: List[BaseMessage] = list(result.get("messages", []))
                new_messages = all_messages[len(inputs):]
                reply = _final_reply(new_messages)
            else:
                ai = await model.ainvoke(
                    [SystemMessage(content=prepared.system_prompt), *inputs]
                )
                new_messages = [ai]
                reply = message_text(ai)
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("%s: LLM 调用失败: %s", self.agent_id, exc)
            raise LLMError(f"{self.agent_id} 执行失败: {exc}") from exc

        metadata = self.build_metadata(context, prepared, new_messages)
        return AgentResponse(reply=reply.strip(), metadata=metadata)

    async def stream(self, context: AgentContext) -> AsyncIterator[Dict[str, Any]]:
        prepared = await self.prepare(context)
        inputs = self.build_messages(context)
        model = self._model(context)
        new_messages: List[BaseMessage] = []
        streamed: List[str] = []
        # 已输出过文本的消息 ID：模型不支持增量输出时，messages 模式会推送完整 AIMessage
        streamed_ids: set = set()

        try:
            if prepared.tools:
                agent = create_agent(
                    model, tools=prepared.tools, system_prompt=prepared.system_prompt
                )
                async for mode, data in agent.astream(
                    {"messages": inputs},
                    config={"recursion_limit": self.recursion_limit},
                    stream_mode=["messages", "updates"],
                ):
                    if mode == "messages":
                        chunk = data[0] if isinstance(data, tuple) else data
                        if isinstance(chunk, AIMessageChunk):
                            text = message_text(chunk)
                            if text:
                                streamed_ids.add(chunk.id)
                                streamed.append(text)
                                yield {"type": "token", "content": text}
                        elif isinstance(chunk, AIMessage) and chunk.id not in streamed_ids:
                            text = message_text(chunk)
                            if text:
                                streamed_ids.add(chunk.id)
                                streamed.append(text)
                                yield {"type": "token", "content": text}
                    elif mode == "updates" and isinstance(data, dict):
                        for update in data.values():
                            if not isinstance(update, dict):
                                continue
                            for msg in update.get("messages", []) or []:
                                new_messages.append(msg)
                                if isinstance(msg, AIMessage) and msg.tool_calls:
                                    for tc in msg.tool_calls:
                                        yield {
                                            "type": "tool_start",
                                            "name": tc.get("name"),
                                            "args": tc.get("args", {}),
                                        }
                                elif isinstance(msg, ToolMessage):
                                    yield {
                                        "type": "tool_end",
                                        "name": msg.name,
                                        "output": message_text(msg)[:500],
                                    }
                reply = _final_reply(new_messages) or "".join(streamed)
            else:
                async for chunk in model.astream(
                    [SystemMessage(content=prepared.system_prompt), *inputs]
                ):
                    text = message_text(chunk)
                    if text:
                        streamed.append(text)
                        yield {"type": "token", "content": text}
                reply = "".join(streamed)
                new_messages = [AIMessage(content=reply)]
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("%s: 流式调用失败: %s", self.agent_id, exc)
            raise LLMError(f"{self.agent_id} 执行失败: {exc}") from exc

        metadata = self.build_metadata(context, prepared, new_messages)
        yield {
            "type": "final",
            "response": AgentResponse(reply=reply.strip(), metadata=metadata),
        }


class GeneralAgent(LangChainAgent):
    """通用助手 Agent：可使用当前用户有权限的全部工具（内置、插件、MCP）。

    作为专属 Agent 失败时的降级对象，也处理 general_task 意图。
    """

    agent_id: str = "general_agent"
    description: str = "通用全能助手"
    tool_tags = None

    def build_system_prompt(self, context: AgentContext) -> str:
        parts: List[str] = [
            "你是 Synapse 全能智能助手，可以调用各种工具完成任务：知识库问答、"
            "记忆检索、网页抓取与搜索、代码仓库查询、定时任务管理以及插件提供的扩展能力。",
            "需要实时信息或具体数据时优先调用工具，不要编造；工具返回错误时如实告知用户。",
            "回答简洁清晰，必要时分点说明。",
        ]
        recall_text = format_recall(context.long_term_recall)
        if recall_text:
            parts.append(f"\n【历史相关摘要】\n{recall_text}")
        if context.user_profile_context:
            parts.append(f"\n【用户画像】\n{context.user_profile_context}")
        return "\n".join(parts)
