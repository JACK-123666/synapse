"""Synapse 平台 FastAPI 入口。

负责：
- 配置日志系统
- 注册 API 路由
- 启动/关闭事件：初始化各模块连接、注册 Agent 和路由
- CORS 中间件

启动方式：
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.config import get_settings

# ============================================================
# 日志配置
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s [%(levelname)s] %(name)s "
        "%(filename)s:%(lineno)d - %(message)s"
    ),
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

# 降低第三方库的日志级别
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ============================================================
# FastAPI 应用实例
# ============================================================

app = FastAPI(
    title="Synapse - 多 Agent 知识检索平台",
    description=(
        "六模块标准化架构：API 网关、意图识别、路由调度、"
        "Agent 执行器、记忆管理、可观测性与故障恢复"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat_router, prefix="", tags=["Core"])


# ============================================================
# 启动事件
# ============================================================

@app.on_event("startup")
async def startup() -> None:
    """应用启动时初始化所有模块连接和配置。

    初始化顺序：
    1. 加载配置
    2. 连接 LLM 客户端
    3. 初始化向量意图索引
    4. 注册 Agent
    5. 注册路由表
    6. 启动异常检测后台任务
    """
    settings = get_settings()
    logger.info("=" * 60)
    logger.info("Synapse v1.0.0 正在启动...")
    logger.info("LLM Provider: %s, Model: %s",
                settings.llm_provider, settings.llm_model)
    logger.info("=" * 60)

    # 1. LLM 客户端
    from app.llm.client import get_llm_client
    llm = get_llm_client()
    try:
        await llm.connect()
        logger.info("[OK] LLM 客户端已初始化")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[SKIP] LLM 客户端初始化失败: %s", exc)

    # 2. 预检 Redis
    try:
        from app.storage import get_redis
        redis = await get_redis()
        await redis.ping()
        logger.info("[OK] Redis 连接正常")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[SKIP] Redis 不可用: %s", exc)

    # 3. 预检 ChromaDB
    try:
        from app.storage import get_chroma
        chroma = get_chroma()
        chroma.heartbeat()
        logger.info("[OK] ChromaDB 连接正常")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[SKIP] ChromaDB 不可用: %s", exc)

    # 4. 初始化向量意图索引
    try:
        from app.intent.fusion import get_intent_fusion
        fusion = get_intent_fusion()
        await fusion.initialize()
        logger.info("[OK] 意图向量索引已就绪")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[SKIP] 意图向量索引初始化失败: %s", exc)

    # 5. 注册 Agent
    from app.router.registry import get_agent_registry
    from app.agents.retrieval_agent import RetrievalAgent
    from app.agents.summarize_agent import SummarizationAgent
    from app.agents.fallback_agent import FallbackAgent

    registry = get_agent_registry()

    retrieval = RetrievalAgent()
    summarize = SummarizationAgent()
    fallback = FallbackAgent()

    registry.register_agent(retrieval)
    registry.register_agent(summarize)
    registry.register_agent(fallback)

    logger.info("[OK] 已注册 %d 个 Agent", len(registry.get_all_agents()))

    # 6. 注册路由表
    # knowledge_retrieval → RetrievalAgent（主），FallbackAgent（备）
    registry.register_route(
        "knowledge_retrieval",
        ["retrieval_agent", "fallback_agent"],
    )
    # summarize → SummarizationAgent（主），FallbackAgent（备）
    registry.register_route(
        "summarize",
        ["summarize_agent", "fallback_agent"],
    )
    # small_talk → RetrievalAgent 兜底处理简单闲聊，FallbackAgent 兜底
    registry.register_route(
        "small_talk",
        ["retrieval_agent", "fallback_agent"],
    )

    logger.info("[OK] 路由表已注册")

    # 7. 启动异常检测后台任务
    from app.observability.anomaly_detector import get_anomaly_detector
    detector = get_anomaly_detector()
    await detector.start_recovery_loop()
    logger.info("[OK] 异常检测后台任务已启动")

    logger.info("=" * 60)
    logger.info("Synapse 启动完成！")
    logger.info("API Docs: http://0.0.0.0:8000/docs")
    logger.info("Metrics:  http://0.0.0.0:8000/metrics")
    logger.info("=" * 60)


# ============================================================
# 关闭事件
# ============================================================

@app.on_event("shutdown")
async def shutdown() -> None:
    """应用关闭时清理资源。"""
    logger.info("Synapse 正在关闭...")

    # 停止异常检测后台任务
    try:
        from app.observability.anomaly_detector import get_anomaly_detector
        detector = get_anomaly_detector()
        await detector.stop_recovery_loop()
    except Exception as exc:  # noqa: BLE001
        logger.warning("停止异常检测失败: %s", exc)

    # 关闭 LLM 客户端
    try:
        from app.llm.client import get_llm_client
        llm = get_llm_client()
        await llm.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("关闭 LLM 客户端失败: %s", exc)

    # 关闭 Redis
    try:
        from app.storage import close_redis
        await close_redis()
    except Exception as exc:  # noqa: BLE001
        logger.warning("关闭 Redis 失败: %s", exc)

    # 关闭 ChromaDB
    try:
        from app.storage import close_chroma
        close_chroma()
    except Exception as exc:  # noqa: BLE001
        logger.warning("关闭 ChromaDB 失败: %s", exc)

    logger.info("Synapse 已关闭")


# ============================================================
# 根路径
# ============================================================

@app.get("/")
async def root():
    """根路径，返回欢迎信息。"""
    return {
        "name": "Synapse - 多 Agent 知识检索平台",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }
