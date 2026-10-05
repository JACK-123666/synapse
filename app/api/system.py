"""系统类 API。

GET /health
- 返回各模块连通性状态（Redis、ChromaDB、LLM、数据库、Agent）

GET /models、POST /models/switch、POST /models/reset
- 查看 / 运行时切换 / 重置 LLM 配置（切换与重置仅管理员）

GET /metrics
- Prometheus 指标拉取端点
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from prometheus_client import generate_latest
from pydantic import BaseModel, Field

from app.core.deps import CurrentUser, get_current_user, require_admin
from app.router.pool import get_agent_registry
from app.store import get_chroma, get_redis

logger = logging.getLogger(__name__)

router = APIRouter()


class HealthResponse(BaseModel):
    """健康检查响应体。"""

    status: str
    modules: Dict[str, Any]


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

    # ChromaDB（同步客户端，放到线程中执行，避免阻塞事件循环）
    try:
        chroma = await asyncio.to_thread(get_chroma)
        await asyncio.to_thread(chroma.heartbeat)
        modules["chromadb"] = "connected"
    except Exception as exc:  # noqa: BLE001
        modules["chromadb"] = f"error: {exc}"

    # 数据库
    try:
        from sqlalchemy import text

        from app.core.db import session_scope

        async with session_scope() as session:
            await session.execute(text("SELECT 1"))
        modules["database"] = "connected"
    except Exception as exc:  # noqa: BLE001
        modules["database"] = f"error: {exc}"

    # LLM（轻量检查：配置是否可用，不发起真实调用）
    try:
        from app.llm.config import get_llm_config

        cfg = get_llm_config().snapshot()
        modules["llm"] = f"configured: {cfg['provider']}/{cfg.get('model')}"
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


def _model_info() -> ModelInfo:
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


@router.get("/models", response_model=ModelInfo, tags=["系统"], summary="查看 LLM 配置")
async def get_models(_: CurrentUser = Depends(get_current_user)) -> ModelInfo:
    """返回当前生效的 LLM 提供商、模型、base_url 等信息。"""
    return _model_info()


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
             summary="运行时切换 LLM 模型（管理员）")
async def switch_model(
    req: ModelSwitchRequest,
    _: CurrentUser = Depends(require_admin),
) -> ModelInfo:
    """运行时切换 LLM 提供商 / 模型 / 密钥，无需重启。

    只填你要改的字段；未填的保持当前值。
    传空字符串 "" 可清空某个运行时覆盖，回退到 .env 配置。
    切换后立即生效，下一次 /chat 使用新配置。
    """
    from app.llm.gateway import get_llm_client
    llm = get_llm_client()
    llm.switch_model(
        provider=req.provider,
        model=req.model,
        api_key=req.api_key,
        base_url=req.base_url,
    )
    return _model_info()


# POST /models/reset — 重置模型配置

@router.post("/models/reset", response_model=ModelInfo, tags=["系统"],
             summary="重置 LLM 配置（管理员）")
async def reset_model(_: CurrentUser = Depends(require_admin)) -> ModelInfo:
    """清空所有运行时覆盖，回退到 .env / 环境变量配置。"""
    from app.llm.gateway import get_llm_client
    llm = get_llm_client()
    llm.reset_runtime()
    return _model_info()


# GET /metrics

@router.get("/metrics", tags=["系统"], summary="监控指标")
async def metrics():
    """Prometheus 吃的指标数据：请求量、延迟分布、Agent 健康分。

    配好 prometheus.yml 指向这个端点就能采集。
    """
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4",
    )


# GET /capabilities — 已注册的能力、意图、路由

@router.get("/capabilities", tags=["系统"], summary="已注册的能力与意图路由")
async def capabilities(_: CurrentUser = Depends(get_current_user)) -> Dict[str, Any]:
    """返回已注册的能力（内置 + 插件）、意图目录与路由表。"""
    from app.capabilities.manager import get_capability_manager
    from app.intent.catalog import get_intent_catalog

    catalog = get_intent_catalog()
    return {
        "capabilities": get_capability_manager().list(),
        "intents": [
            {"name": s.name, "description": s.description, "source": s.source}
            for s in catalog.specs()
        ],
        "routes": get_agent_registry().get_routes(),
    }
