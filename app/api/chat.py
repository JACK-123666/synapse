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
    """聊天请求体。

    Attributes:
        session_id: 会话唯一标识
        message: 用户输入消息
        user_id: 用户 ID（可选，用于用户画像）
    """

    session_id: str = Field(..., min_length=1, description="会话 ID")
    message: str = Field(..., min_length=1, description="用户消息")
    user_id: Optional[str] = Field(default=None, description="用户 ID")


class ChatResponse(BaseModel):
    """聊天响应体。

    Attributes:
        reply: Agent 回复文本
        intent: 识别出的意图标签
        agent_used: 实际执行的 Agent ID
        confidence: 意图识别置信度
    """

    reply: str
    intent: str
    agent_used: str
    confidence: float


class HealthResponse(BaseModel):
    """健康检查响应体。"""

    status: str
    modules: Dict[str, Any]


# ============================================================
# POST /chat
# ============================================================

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """核心对话接口。

    处理流程：
    1. 意图识别（三路融合）
    2. 构建 AgentContext（短期记忆、长期记忆召回、用户画像）
    3. 路由调度执行 Agent
    4. 更新记忆（短期记忆追加 + 压缩检查）
    5. 返回回复

    Args:
        request: 聊天请求

    Returns:
        包含回复、意图、Agent 信息的响应
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

@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """系统健康检查。

    检查 Redis、ChromaDB、LLM 的连通性。
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

@router.get("/metrics")
async def metrics():
    """Prometheus 指标拉取端点。

    返回 Prometheus 标准格式的指标数据，
    供 prometheus.yml 中配置的 scrape_config 抓取。
    """
    from fastapi.responses import Response
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4",
    )
