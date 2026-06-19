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

from fastapi import APIRouter, Header, HTTPException
from prometheus_client import generate_latest
from pydantic import BaseModel, Field

from app.agents.base import AgentContext
from app.intent.blend import get_intent_fusion
from app.memory.compress import get_memory_compressor
from app.memory.archive import get_long_term_memory
from app.memory.recent import get_short_term_memory
from app.memory.profile import get_user_profile_manager
from app.router.route import get_task_dispatcher
from app.router.pool import get_agent_registry
from app.store import get_chroma, get_redis

logger = logging.getLogger(__name__)

router = APIRouter()


# 请求 / 响应模型

class ChatRequest(BaseModel):
    """对话请求。

    填 session_id 和 message 就能用，user_id 选填。
    同一个 session_id 会共享短期记忆，实现多轮对话。
    model 可选：临时指定本次请求使用的模型（不影响全局配置）。
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
    model: Optional[str] = Field(
        default=None,
        description="可选。本次请求使用的模型名，覆盖全局配置（如 gpt-4o、deepseek-chat）",
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


# GET /chat — 返回聊天网页

@router.get("/chat", include_in_schema=False)
async def chat_page():
    """聊天页面。"""
    from fastapi.responses import FileResponse
    return FileResponse("app/static/index.html")


# POST /chat

@router.post("/chat", response_model=ChatResponse, tags=["对话"], summary="发送消息")
async def chat(
    request: ChatRequest,
    x_web_search: str = Header(default="1", alias="X-Web-Search"),
) -> ChatResponse:
    """发一条消息给 Synapse，拿到回复。

    背后自动完成：意图识别 → 记忆召回 → Agent 执行 → 记忆更新。

    同一个 session_id 连续调用即可多轮对话，系统会记住上下文。
    对话超过 8 轮会自动压缩历史，不额外消耗 Token。

    Header X-Web-Search: 1/0 控制是否启用联网搜索。
    """
    web_search_enabled = x_web_search == "1"
    session_id = request.session_id
    message = request.message
    user_id = request.user_id

    # 模型临时覆盖
    saved_model: Optional[str] = None
    if request.model:
        from app.llm.gateway import get_llm_client
        llm_cl = get_llm_client()
        saved_config = llm_cl.get_config()
        saved_model = saved_config.get("model")
        llm_cl.switch_model(model=request.model)
        logger.info(
            "[API] session=%s 临时切换模型: %s -> %s",
            session_id, saved_model, request.model,
        )

    # 意图识别
    fusion = get_intent_fusion()
    intent, confidence = await fusion.recognize(message)
    logger.info(
        "[API] session=%s message='%s' intent=%s confidence=%.2f",
        session_id, message[:100], intent, confidence,
    )

    # 组装 AgentContext
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
        web_search=web_search_enabled,
    )

    # 路由调度
    dispatcher = get_task_dispatcher()
    response = await dispatcher.dispatch(context)

    # 记忆更新 — 追加用户消息和助手回复到短期记忆
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

    # 恢复模型
    if saved_model is not None:
        from app.llm.gateway import get_llm_client
        get_llm_client().switch_model(model=saved_model)

    # 返回
    agent_used = response.metadata.get("agent_id", "unknown")
    return ChatResponse(
        reply=response.reply,
        intent=intent,
        agent_used=agent_used,
        confidence=round(confidence, 4),
    )


# GET /health

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
        from app.llm.gateway import get_llm_client
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


# GET /models — 查看当前 LLM 配置

class ModelInfo(BaseModel):
    """LLM 模型信息。"""
    provider: str
    model: str
    base_url: str
    has_api_key: bool
    embedding_api_key_set: bool
    timeout_s: int
    runtime_overrides: List[str] = Field(default_factory=list)


@router.get("/models", response_model=ModelInfo, tags=["系统"], summary="查看 LLM 配置")
async def get_models() -> ModelInfo:
    """返回当前生效的 LLM 提供商、模型、base_url 等信息。"""
    from app.llm.gateway import get_llm_client
    llm = get_llm_client()
    cfg = llm.get_config()
    return ModelInfo(
        provider=cfg["provider"],
        model=cfg["model"],
        base_url=cfg["base_url"],
        has_api_key=cfg["api_key_prefix"] != "(empty)",
        embedding_api_key_set=cfg.get("embedding_api_key_set", False),
        timeout_s=cfg["timeout_s"],
        runtime_overrides=list(llm._runtime.keys()) if hasattr(llm, "_runtime") else [],
    )


# POST /models/switch — 运行时切换模型

class ModelSwitchRequest(BaseModel):
    """模型切换请求。只需填你要改的字段，未填的不变。"""
    provider: Optional[str] = Field(
        default=None, description="LLM 提供商: openai / deepseek / claude"
    )
    model: Optional[str] = Field(
        default=None, description="模型名（如 gpt-4o、deepseek-chat）"
    )
    api_key: Optional[str] = Field(
        default=None, description="新的 API 密钥"
    )
    base_url: Optional[str] = Field(
        default=None, description="新的 base_url"
    )


@router.post("/models/switch", response_model=ModelInfo, tags=["系统"],
             summary="运行时切换 LLM 模型")
async def switch_model(req: ModelSwitchRequest) -> ModelInfo:
    """运行时切换 LLM 提供商 / 模型 / 密钥，无需重启。

    只填你要改的字段；未填的保持当前值。
    传空字符串 "" 可清空某个运行时覆盖，回退到 .env 配置。
    切换后立即生效，下一次 /chat 使用新配置。
    """
    from app.llm.gateway import get_llm_client
    llm = get_llm_client()
    cfg = llm.switch_model(
        provider=req.provider,
        model=req.model,
        api_key=req.api_key,
        base_url=req.base_url,
    )
    return ModelInfo(
        provider=cfg["provider"],
        model=cfg["model"],
        base_url=cfg["base_url"],
        has_api_key=cfg["api_key_prefix"] != "(empty)",
        embedding_api_key_set=cfg.get("embedding_api_key_set", False),
        timeout_s=cfg["timeout_s"],
        runtime_overrides=list(llm._runtime.keys()) if hasattr(llm, "_runtime") else [],
    )


# POST /models/reset — 重置模型配置

@router.post("/models/reset", response_model=ModelInfo, tags=["系统"],
             summary="重置 LLM 配置")
async def reset_model() -> ModelInfo:
    """清空所有运行时覆盖，回退到 .env / 环境变量配置。"""
    from app.llm.gateway import get_llm_client
    llm = get_llm_client()
    llm.reset_runtime()
    return await get_models()


# GET /metrics

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
