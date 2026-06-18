"""核心 API 端点。

POST /chat
- 接收用户消息，执行完整处理链路：
  意图识别 → 路由调度 → Agent 执行 → 记忆管理 → 返回回复

GET /health
- 返回各模块连通性状态（Redis、ChromaDB、LLM）

GET /metrics
- Prometheus 指标拉取端点
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from prometheus_client import generate_latest
from pydantic import BaseModel, Field

from app.agents.base import AgentContext
from app.intent.fusion import get_intent_fusion
from app.memory.compressor import get_memory_compressor
from app.memory.long_term import get_long_term_memory
from app.memory.short_term import get_short_term_memory
from app.memory.user_profile import get_user_profile_manager
from app.router.dispatcher import get_task_dispatcher
from app.router.registry import get_agent_registry
from app.storage import get_chroma, get_redis

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================
# 请求 / 响应模型
# ============================================================

class ChatRequest(BaseModel):
    """对话请求。

    填 session_id 和 message 就能用，user_id 选填。
    同一个 session_id 会共享短期记忆，实现多轮对话。
    """

    session_id: str = Field(
        ..., min_length=1,
        description="会话标识，随便起个名字就行（如 test-001）。同一会话多轮对话用同一个 ID",
    )
    message: str = Field(
        ..., min_length=1,
        description="你想说的话，支持模糊短句（如「那个怎么弄」「帮我查一下」）",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="可选。填了会记录你的偏好，下次回答更懂你",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": "demo-001",
                    "message": "什么是向量数据库？",
                    "user_id": "u1",
                },
                {
                    "session_id": "demo-001",
                    "message": "它和 MySQL 有什么区别",
                },
            ]
        }
    }


class ChatResponse(BaseModel):
    """对话响应。"""

    reply: str = Field(description="Agent 生成的回复内容")
    intent: str = Field(description="识别出的意图：knowledge_retrieval 查知识 / summarize 做摘要 / small_talk 闲聊")
    agent_used: str = Field(description="实际干活的是哪个 Agent")
    confidence: float = Field(description="意图识别的把握有多大（0~1）")


class HealthResponse(BaseModel):
    """健康检查响应体。"""

    status: str
    modules: Dict[str, Any]


# ============================================================
# POST /chat
# ============================================================

@router.post("/chat", response_model=ChatResponse, tags=["对话"], summary="发送消息")
async def chat(request: ChatRequest) -> ChatResponse:
    """发一条消息给 Synapse，拿到回复。

    背后自动完成：意图识别 → 记忆召回 → Agent 执行 → 记忆更新。

    同一个 session_id 连续调用即可多轮对话，系统会记住上下文。
    对话超过 8 轮会自动压缩历史，不额外消耗 Token。
    """
    session_id = request.session_id
    message = request.message
    user_id = request.user_id

    # ---- 1. 意图识别 ----
    fusion = get_intent_fusion()
    intent, confidence = await fusion.recognize(message)
    logger.info(
        "[API] session=%s message='%s' intent=%s confidence=%.2f",
        session_id, message[:100], intent, confidence,
    )

    # ---- 2. 组装 AgentContext ----
    short_mem = get_short_term_memory()
    long_mem = get_long_term_memory()
    profile_mgr = get_user_profile_manager()

    # 短期记忆
    short_term_msgs = await short_mem.get_messages(session_id)

    # 长期记忆召回（best-effort，ChromaDB 不可用时返回空）
    try:
        recall = await long_mem.recall(query_text=message, user_id=user_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[API] 长期记忆召回失败（降级空列表）: %s", exc)
        recall = []

    # 用户画像上下文
    profile_context = await profile_mgr.build_prompt_context(user_id)

    context = AgentContext(
        session_id=session_id,
        user_id=user_id,
        message=message,
        intent=intent,
        short_term_memory=short_term_msgs,
        long_term_recall=recall,
        user_profile_context=profile_context,
    )

    # ---- 3. 路由调度 ----
    dispatcher = get_task_dispatcher()
    response = await dispatcher.dispatch(context)

    # ---- 4. 记忆更新 ----
    # 追加用户消息和助手回复到短期记忆
    try:
        await short_mem.append(session_id, "user", message)
        await short_mem.append(session_id, "assistant", response.reply)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[API] 短期记忆追加失败: %s", exc)

    # 更新用户画像（交互次数 + 从消息中提取关键词作为常用术语）
    try:
        await profile_mgr.increment_interaction(user_id)
        # 简单提取：按空格和中文字符分割提取大于 2 字符的词
        import re
        terms = re.findall(r'[\w\u4e00-\u9fff]{2,}', message)
        if terms:
            await profile_mgr.add_frequent_terms(user_id, terms[:10])
    except Exception as exc:  # noqa: BLE001
        logger.warning("[API] 用户画像更新失败: %s", exc)

    # 检查是否需要摘要压缩 — 改为后台异步，不阻塞用户响应
    try:
        compressor = get_memory_compressor()
        if await compressor.should_compress(session_id):
            asyncio.create_task(
                compressor.compress(session_id, user_id),
                name=f"compress-{session_id}"
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[API] 记忆压缩检查失败: %s", exc)

    # ---- 5. 返回 ----
    agent_used = response.metadata.get("agent_id", "unknown")
    return ChatResponse(
        reply=response.reply,
        intent=intent,
        agent_used=agent_used,
        confidence=round(confidence, 4),
    )


# ============================================================
# GET /health
# ============================================================

@router.get("/health", response_model=HealthResponse, tags=["系统"], summary="健康检查")
async def health() -> HealthResponse:
    """看一眼 Redis、ChromaDB、LLM 还活着没，Agent 状态如何。

    返回 healthy 表示一切正常，degraded 表示有模块挂了。
    """
    modules: Dict[str, Any] = {}

    # Redis
    try:
        redis = await get_redis()
        await redis.ping()
        modules["redis"] = "connected"
    except Exception as exc:  # noqa: BLE001
        modules["redis"] = f"error: {exc}"

    # ChromaDB
    try:
        chroma = get_chroma()
        chroma.heartbeat()
        modules["chromadb"] = "connected"
    except Exception as exc:  # noqa: BLE001
        modules["chromadb"] = f"error: {exc}"

    # LLM（轻量检查：仅验证客户端存在）
    try:
        from app.llm.client import get_llm_client
        llm = get_llm_client()
        if llm.http is not None:
            modules["llm"] = "connected"
        else:
            modules["llm"] = "not initialized"
    except Exception as exc:  # noqa: BLE001
        modules["llm"] = f"error: {exc}"

    # Agent 健康状态
    registry = get_agent_registry()
    modules["agents"] = registry.get_all_health_status()

    all_ok = all(
        isinstance(v, str) and v == "connected"
        for v in [modules.get("redis"), modules.get("chromadb")]
    )
    status = "healthy" if all_ok else "degraded"

    return HealthResponse(status=status, modules=modules)


# ============================================================
# GET /metrics
# ============================================================

@router.get("/metrics", tags=["系统"], summary="监控指标")
async def metrics():
    """Prometheus 吃的指标数据：请求量、延迟分布、Agent 健康分。

    配好 prometheus.yml 指向这个端点就能采集。
    """
    from fastapi.responses import Response
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4",
    )
