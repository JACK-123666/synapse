"""对话编排服务。

从原 api/chat.py 的 /chat 处理函数中抽出，处理链路保持不变：
    意图识别 → 记忆召回 → 路由调度 → Agent 执行 → 记忆更新 → 画像更新 → 后台压缩

变化：
- 请求级模型覆盖写入请求上下文，只影响当前请求（原实现会临时修改全局 LLM 客户端）
- 开启鉴权时，会话 ID 与记忆按登录用户隔离
- 提供 chat()（一次性返回）与 stream()（流式事件）两个入口，共用前后处理
"""

from __future__ import annotations

import logging
import re
from contextvars import Token
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from app.agents.base import AgentContext, AgentResponse
from app.config import get_settings
from app.core.context import (
    RequestContext,
    request_context,
    reset_request_context,
    set_request_context,
)
from app.core.deps import CurrentUser
from app.core.tasks import spawn
from app.intent.blend import get_intent_fusion
from app.memory.archive import get_long_term_memory
from app.memory.compress import get_memory_compressor
from app.memory.profile import get_user_profile_manager
from app.memory.recent import get_short_term_memory
from app.router.route import get_task_dispatcher

logger = logging.getLogger(__name__)


@dataclass
class ChatInput:
    """一次对话请求的输入。"""

    session_id: str
    message: str
    user_id: Optional[str] = None
    model: Optional[str] = None
    web_search: bool = True


@dataclass
class ChatResult:
    """一次对话请求的结果。"""

    reply: str
    intent: str
    agent_used: str
    confidence: float
    sources: List[Dict[str, Any]] = field(default_factory=list)
    web_sources: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转成 API 响应字段。"""
        return {
            "reply": self.reply,
            "intent": self.intent,
            "agent_used": self.agent_used,
            "confidence": self.confidence,
            "sources": self.sources,
            "web_sources": self.web_sources,
            "tool_calls": self.tool_calls,
        }


def session_key_for(user_id: str, session_id: str) -> str:
    """开启鉴权时按用户隔离会话；关闭鉴权时沿用原始会话 ID（与旧版本兼容）。"""
    if get_settings().auth_enabled:
        return f"{user_id}:{session_id}"
    return session_id


class ChatService:
    """对话编排。"""

    def build_request_context(self, user: CurrentUser, inp: ChatInput) -> RequestContext:
        """根据登录用户与请求参数组装请求上下文。"""
        settings = get_settings()
        # 关闭鉴权时沿用请求体里的 user_id（旧行为）；开启鉴权时强制使用登录用户，防止冒用
        memory_user_id = user.id if settings.auth_enabled else inp.user_id
        return RequestContext(
            user_id=user.id,
            username=user.username,
            role=user.role,
            session_id=session_key_for(user.id, inp.session_id),
            memory_user_id=memory_user_id,
            model_override=inp.model or None,
            web_search=inp.web_search,
        )

    async def _prepare(self, ctx: RequestContext, inp: ChatInput) -> AgentContext:
        """意图识别 + 组装 AgentContext。"""
        message = inp.message
        session_id = ctx.session_id
        user_id = ctx.memory_user_id

        if ctx.model_override:
            logger.info("[Chat] session=%s 本次请求使用模型: %s", session_id, ctx.model_override)

        # 意图识别
        fusion = get_intent_fusion()
        intent, confidence = await fusion.recognize(message)
        logger.info(
            "[Chat] session=%s message='%s' intent=%s confidence=%.2f",
            session_id, message[:100], intent, confidence,
        )

        short_mem = get_short_term_memory()
        long_mem = get_long_term_memory()
        profile_mgr = get_user_profile_manager()

        # 短期记忆（Redis 不可用时降级为空，保证对话可用）
        try:
            short_term_msgs = await short_mem.get_messages(session_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Chat] 短期记忆读取失败（降级空列表）: %s", exc)
            short_term_msgs = []

        # 长期记忆召回（best-effort，ChromaDB 不可用时返回空）
        try:
            recall = await long_mem.recall(query_text=message, user_id=user_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Chat] 长期记忆召回失败（降级空列表）: %s", exc)
            recall = []

        # 用户画像上下文
        try:
            profile_context = await profile_mgr.build_prompt_context(user_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Chat] 用户画像读取失败: %s", exc)
            profile_context = ""

        return AgentContext(
            session_id=session_id,
            user_id=user_id,
            message=message,
            intent=intent,
            short_term_memory=short_term_msgs,
            long_term_recall=recall,
            user_profile_context=profile_context,
            web_search=ctx.web_search,
            model_override=ctx.model_override,
            user_role=ctx.role,
            allowed_tools=ctx.allowed_tools,
            intent_confidence=confidence,
        )

    async def _after(self, ctx: RequestContext, message: str, reply: str) -> None:
        """记忆更新、画像更新、后台压缩。"""
        session_id = ctx.session_id
        user_id = ctx.memory_user_id
        short_mem = get_short_term_memory()
        profile_mgr = get_user_profile_manager()

        # 记忆更新 — 追加用户消息和助手回复到短期记忆
        try:
            await short_mem.append(session_id, "user", message)
            await short_mem.append(session_id, "assistant", reply)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Chat] 短期记忆追加失败: %s", exc)

        # 更新用户画像（交互次数 + 从消息中提取关键词作为常用术语）
        try:
            await profile_mgr.increment_interaction(user_id)
            # 简单提取：按空格和中文字符分割提取大于 2 字符的词
            terms = re.findall(r"[\w\u4e00-\u9fff]{2,}", message)
            if terms:
                await profile_mgr.add_frequent_terms(user_id, terms[:10])
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Chat] 用户画像更新失败: %s", exc)

        # 检查是否需要摘要压缩 — 后台异步，不阻塞用户响应
        try:
            compressor = get_memory_compressor()
            if await compressor.should_compress(session_id):
                # 经 spawn() 登记强引用：裸 create_task 的返回值无人引用时，
                # 事件循环可能中途 GC 掉该任务，异常也无从取出
                spawn(
                    compressor.compress(session_id, user_id),
                    name=f"compress-{session_id}",
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Chat] 记忆压缩检查失败: %s", exc)

    @staticmethod
    def _result(agent_ctx: AgentContext, response: AgentResponse) -> ChatResult:
        metadata = response.metadata
        metadata.pop("_artifacts", None)
        return ChatResult(
            reply=response.reply,
            intent=agent_ctx.intent,
            agent_used=metadata.get("agent_id", "unknown"),
            confidence=round(agent_ctx.intent_confidence, 4),
            sources=list(metadata.get("sources", []) or []),
            web_sources=list(metadata.get("web_sources", []) or []),
            tool_calls=list(metadata.get("tool_calls", []) or []),
        )

    async def chat(self, user: CurrentUser, inp: ChatInput) -> ChatResult:
        """一次性返回完整回复。"""
        ctx = self.build_request_context(user, inp)
        with request_context(ctx):
            agent_ctx = await self._prepare(ctx, inp)
            response = await get_task_dispatcher().dispatch(agent_ctx)
            await self._after(ctx, inp.message, response.reply)
            return self._result(agent_ctx, response)

    async def stream(self, user: CurrentUser, inp: ChatInput) -> AsyncIterator[Dict[str, Any]]:
        """流式返回。

        事件顺序：meta → (token | tool_start | tool_end)* → done；出错时有 error 事件。
        """
        ctx = self.build_request_context(user, inp)
        token: Token = set_request_context(ctx)
        try:
            agent_ctx = await self._prepare(ctx, inp)
            yield {
                "type": "meta",
                "intent": agent_ctx.intent,
                "confidence": round(agent_ctx.intent_confidence, 4),
            }
            final: Optional[Dict[str, Any]] = None
            async for event in get_task_dispatcher().dispatch_stream(agent_ctx):
                if event.get("type") == "final":
                    final = event
                    continue
                yield event
            if final is None:
                return
            response: AgentResponse = final["response"]
            await self._after(ctx, inp.message, response.reply)
            yield {"type": "done", **self._result(agent_ctx, response).to_dict()}
        finally:
            try:
                reset_request_context(token)
            except ValueError:
                # 生成器在其他上下文中被关闭（如客户端断开），忽略
                pass


_service: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    """获取对话编排服务单例。"""
    global _service
    if _service is None:
        _service = ChatService()
    return _service
