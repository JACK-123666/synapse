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
from functools import lru_cache
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple

from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from app.agents.base import AgentContext, AgentResponse, BaseAgent
from app.llm.factory import ModelSpec, get_chat_model, resolve_model_spec
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
    #: 每请求变化的上下文（检索到的知识等）。刻意不放进 system_prompt，
    #: 而是拼到消息末尾，见 build_context_block 的说明。
    context_blocks: List[str] = field(default_factory=list)


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


def build_context_block(
    context: AgentContext,
    extra_blocks: Optional[Sequence[str]] = None,
) -> str:
    """拼装每请求变化的上下文块：检索到的知识 + 长期记忆召回 + 用户画像。

    这些内容刻意不写进 system_prompt，因为 system_prompt 一旦逐字节稳定：
    - 编译后的 Agent 图才能按 (模型配置, 工具集, system_prompt) 缓存复用；
    - 上游 LLM 的 prefix cache 才能命中，省掉每轮重新 prefill 的开销。
    动态内容统一落在消息序列末尾，稳定前缀不被破坏。
    """
    blocks: List[str] = [b for b in (extra_blocks or []) if b]
    recall_text = format_recall(context.long_term_recall)
    if recall_text:
        blocks.append(f"【历史相关摘要】\n{recall_text}")
    if context.user_profile_context:
        blocks.append(f"【用户画像】\n{context.user_profile_context}")
    return "\n\n".join(blocks)


#: 编译后的 Agent 图缓存，key = (模型配置, 工具名集合, system_prompt)。
#: create_agent 会编译一张 LangGraph 状态图，每请求重建纯属浪费；
#: 编译产物本身无状态，可以安全地跨请求、跨并发复用。
@lru_cache(maxsize=128)
def _compiled_agent(spec: ModelSpec, tool_names: Tuple[str, ...], system_prompt: str):
    model = get_chat_model(
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        model_override=spec.model,
    )
    registry = get_tool_registry()
    tools = [t for t in (registry.get(n) for n in tool_names) if t is not None]
    return create_agent(model, tools=tools, system_prompt=system_prompt)


def clear_agent_cache() -> None:
    """清空 Agent 图缓存。

    能力 / 插件注册或注销后必须调用：缓存里的图持有的是当时的工具对象，
    热重载若只换了实现而工具名不变，不清缓存就会继续用旧的。
    """
    _compiled_agent.cache_clear()


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
    #: 系统提示词。必须是静态字符串：一旦掺入每请求都变的内容，
    #: Agent 图缓存会持续失效并把其他 Agent 的缓存挤出去。
    system_prompt: str = ""
    #: 是否允许写操作工具
    include_write_tools: bool = True
    temperature: float = 0.5
    max_tokens: int = 2048
    #: create_agent 图的最大步数（防止工具调用死循环）—— execute() 的非流式路径使用
    recursion_limit: int = 16
    #: 流式路径下最多跑几轮工具调用
    max_tool_rounds: int = 8

    # ---- 可覆写的钩子 ----

    def build_system_prompt(self, context: AgentContext) -> str:
        """系统提示词。

        必须只依赖 Agent 自身的静态信息 —— 子类通过 system_prompt 类属性提供。
        动态内容（记忆召回、用户画像、检索结果）一律交给 build_context_block
        拼进消息序列，这样 system_prompt 逐字节稳定，Agent 图缓存与上游
        prefix cache 才能命中。context 参数保留是为了子类签名兼容。
        """
        return self.system_prompt or f"你是 Synapse 智能助手中的「{self.description}」。"

    def select_tools(self, context: AgentContext) -> List[BaseTool]:
        """按当前用户的角色与标签，从工具注册表挑出本 Agent 可用的工具。"""
        if self.tool_tags is not None and len(self.tool_tags) == 0:
            return []
        return get_tool_registry().tools_for(
            role=context.user_role,
            tags=self.tool_tags,
            names=context.allowed_tools,
            include_write=self.include_write_tools,
        )

    def build_messages(
        self,
        context: AgentContext,
        prepared: Optional[PreparedRun] = None,
    ) -> List[BaseMessage]:
        """短期记忆 + 上下文块 + 当前消息。

        上下文块（检索知识 / 长期记忆 / 用户画像）刻意放在历史之后、当前消息之前：
        system_prompt 与历史构成稳定前缀，动态内容只落在末尾，每轮仅末段变化，
        Agent 图缓存与上游 prefix cache 都不受影响。
        """
        messages: List[BaseMessage] = []
        for msg in context.short_term_memory:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "assistant":
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))

        ctx_block = build_context_block(
            context, prepared.context_blocks if prepared else None
        )
        if ctx_block:
            messages.append(HumanMessage(content=ctx_block))
            # 给上下文一个明确的回合边界，避免模型把资料当成用户指令
            messages.append(AIMessage(content="好的，我已了解这些背景资料。"))

        messages.append(HumanMessage(content=context.message))
        return messages

    async def prepare(self, context: AgentContext) -> PreparedRun:
        """执行前的准备：拼系统提示词、选出工具。子类可覆写做更复杂的预处理。"""
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
        """从本轮消息中提取工具调用记录与产物，作为响应元数据。"""
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

    def _agent_for(self, context: AgentContext, prepared: PreparedRun):
        """取（必要时编译并缓存）本请求对应的 Agent 图。"""
        spec = resolve_model_spec(
            model_override=context.model_override,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        tool_names = tuple(sorted(t.name for t in prepared.tools))
        return _compiled_agent(spec, tool_names, prepared.system_prompt)

    async def execute(self, context: AgentContext) -> AgentResponse:
        """执行一次完整调用并返回回复。"""
        prepared = await self.prepare(context)
        inputs = self.build_messages(context, prepared)
        try:
            if prepared.tools:
                # 复用缓存的 Agent 图，不再每请求重新编译 LangGraph
                agent = self._agent_for(context, prepared)
                result = await agent.ainvoke(
                    {"messages": inputs},
                    config={"recursion_limit": self.recursion_limit},
                )
                all_messages: List[BaseMessage] = list(result.get("messages", []))
                new_messages = all_messages[len(inputs):]
                reply = _final_reply(new_messages)
            else:
                ai = await self._model(context).ainvoke(
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
        """流式执行，逐 token 产出事件。带工具时自己跑 ReAct 循环，确保真正逐字输出。"""
        prepared = await self.prepare(context)
        inputs = self.build_messages(context, prepared)
        new_messages: List[BaseMessage] = []
        streamed: List[str] = []

        try:
            if prepared.tools:
                # 这里刻意不用 create_agent：它的模型节点调用的是 model_.ainvoke()（非流式），
                # LangGraph 的 messages 模式因此收不到 token —— 整段回复会被攒成一条
                # AIMessage 再吐出来，前端看到的就是"一次性输出"。
                # 自己跑一遍 ReAct 循环，用 model.astream() 才能真正逐 token 推送。
                reply = ""
                tool_map = {t.name: t for t in prepared.tools}
                bound = self._model(context).bind_tools(prepared.tools)
                convo: List[BaseMessage] = [
                    SystemMessage(content=prepared.system_prompt), *inputs
                ]

                for _ in range(self.max_tool_rounds):
                    gathered = None
                    round_text: List[str] = []
                    async for chunk in bound.astream(convo):
                        # AIMessageChunk 支持相加，逐块拼成完整的 AIMessage
                        gathered = chunk if gathered is None else gathered + chunk
                        text = message_text(chunk)
                        if text:
                            round_text.append(text)
                            streamed.append(text)
                            yield {"type": "token", "content": text}

                    if gathered is None:
                        # 个别 provider 不支持流式工具调用，一个 chunk 都不给。
                        # 退回非流式取这一轮结果，避免整段回复丢失。
                        ai_msg = await bound.ainvoke(convo)
                        text = message_text(ai_msg)
                        if text:
                            round_text.append(text)
                            streamed.append(text)
                            yield {"type": "token", "content": text}
                    else:
                        ai_msg = AIMessage(
                            content=getattr(gathered, "content", "") or "",
                            tool_calls=list(getattr(gathered, "tool_calls", None) or []),
                        )

                    new_messages.append(ai_msg)
                    convo.append(ai_msg)

                    tool_calls = list(getattr(ai_msg, "tool_calls", None) or [])
                    if not tool_calls:
                        reply = "".join(round_text)
                        break

                    # 先把本轮所有 tool_start 推给前端，再逐个执行
                    for call in tool_calls:
                        yield {
                            "type": "tool_start",
                            "name": call.get("name"),
                            "args": call.get("args", {}),
                        }
                    for call in tool_calls:
                        name = call.get("name") or ""
                        tool = tool_map.get(name)
                        try:
                            if tool is None:
                                output = f"未知工具: {name}"
                            else:
                                raw = await tool.ainvoke(call.get("args") or {})
                                output = raw if isinstance(raw, str) else str(raw)
                        except Exception as exc:  # noqa: BLE001
                            output = f"工具执行失败: {exc}"
                        tool_msg = ToolMessage(
                            content=output, tool_call_id=call.get("id"), name=name
                        )
                        new_messages.append(tool_msg)
                        convo.append(tool_msg)
                        yield {"type": "tool_end", "name": name, "output": output[:500]}

                if not reply:
                    reply = _final_reply(new_messages) or "".join(streamed)
            else:
                async for chunk in self._model(context).astream(
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

    system_prompt = (
        "你是 Synapse 全能智能助手，可以调用各种工具完成任务：知识库问答、"
        "记忆检索、网页抓取与搜索、代码仓库查询、定时任务管理以及插件提供的扩展能力。\n"
        "需要实时信息或具体数据时优先调用工具，不要编造；工具返回错误时如实告知用户。\n"
        "回答简洁清晰，必要时分点说明。"
    )
