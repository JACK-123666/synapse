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
import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.config import get_settings

# 日志配置

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

# FastAPI 应用实例

_app_settings = get_settings()

app = FastAPI(
    title="Synapse · 智能对话平台",
    description="""
## 👋 欢迎使用 Synapse

一个带**意图识别**、**记忆管理**和**故障自愈**的智能对话 API。

### 怎么用

1. 先调 `POST /chat` 发一条消息
2. 拿到的 `session_id` 原样传回，就能多轮对话
3. 随时调 `GET /health` 看各模块是否正常

### 背后做了什么

你说「那个怎么弄」→ 三路融合识别你要查文档还是闲聊 → 按 Agent 权重路由 →
检索知识库 / 摘要历史 / 兜底回复 → 自动管理短期和长期记忆。

全程有 Prometheus 盯着，哪个 Agent 慢了自动降权、摘除、恢复。
""",
    version="1.0.0",
    docs_url="/docs" if _app_settings.docs_enabled else None,
    redoc_url="/redoc" if _app_settings.docs_enabled else None,
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,
        "displayRequestDuration": True,
        "filter": True,
        "tryItOutEnabled": True,
    },
)

# CORS 中间件：allow_origins=* 时浏览器禁止 credentials，故按配置动态决定
_cors_origins = [o.strip() for o in _app_settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials="*" not in _cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由（不设置顶层 tags，交给各自端点自行标记）
app.include_router(chat_router, prefix="")

# 静态文件 — 聊天首页
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")


# 启动事件

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
    # 配置来源：docker 走 compose env_file 注入环境变量；本地开发走 .env 文件
    logger.info(
        "配置来源: %s",
        "环境变量(compose env_file 注入)" if os.environ.get("LLM_PROVIDER")
        else ".env 文件(本地开发)",
    )
    logger.info("LLM Provider: %s, Model: %s",
                settings.llm_provider, settings.llm_model)
    logger.info("LLM BaseURL: %s", settings.llm_base_url)
    logger.info("LLM API Key: %s...", settings.llm_api_key[:8] if settings.llm_api_key else "(empty)")
    logger.info("DeepSeek URL: %s, Model: %s",
                settings.deepseek_base_url, settings.deepseek_model)
    logger.info("=" * 60)

    # LLM 客户端
    from app.llm.gateway import get_llm_client
    llm = get_llm_client()
    try:
        await llm.connect()
        logger.info("[OK] LLM 客户端已初始化")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[SKIP] LLM 客户端初始化失败: %s", exc)

    # 预检 Redis
    try:
        from app.store import get_redis
        redis = await get_redis()
        await redis.ping()
        logger.info("[OK] Redis 连接正常")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[SKIP] Redis 不可用: %s", exc)

    # 预检 ChromaDB
    try:
        from app.store import get_chroma
        chroma = get_chroma()
        chroma.heartbeat()
        logger.info("[OK] ChromaDB 连接正常")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[SKIP] ChromaDB 不可用: %s", exc)

    # 初始化向量意图索引
    try:
        from app.intent.blend import get_intent_fusion
        fusion = get_intent_fusion()
        await fusion.initialize()
        logger.info("[OK] 意图向量索引已就绪")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[SKIP] 意图向量索引初始化失败: %s", exc)

    # 注册 Agent + 路由表
    try:
        from app.router.pool import get_agent_registry
        from app.agents.knowledge import RetrievalAgent
        from app.agents.summary import SummarizationAgent
        from app.agents.safety import FallbackAgent

        registry = get_agent_registry()

        retrieval = RetrievalAgent()
        summarize = SummarizationAgent()
        fallback = FallbackAgent()

        registry.register_agent(retrieval)
        registry.register_agent(summarize)
        registry.register_agent(fallback)

        logger.info("[OK] 已注册 %d 个 Agent", len(registry.get_all_agents()))

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
    except Exception as exc:  # noqa: BLE001
        logger.warning("[SKIP] Agent/路由注册失败: %s", exc)

    # 启动异常检测后台任务
    try:
        from app.observability.health import get_anomaly_detector
        detector = get_anomaly_detector()
        await detector.start_recovery_loop()
        logger.info("[OK] 异常检测后台任务已启动")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[SKIP] 异常检测后台任务启动失败: %s", exc)

    logger.info("=" * 60)
    logger.info("Synapse 启动完成！")
    logger.info("API Docs: http://0.0.0.0:8000/docs")
    logger.info("Metrics:  http://0.0.0.0:8000/metrics")
    logger.info("=" * 60)


# 关闭事件

@app.on_event("shutdown")
async def shutdown() -> None:
    """应用关闭时清理资源。"""
    logger.info("Synapse 正在关闭...")

    # 停止异常检测后台任务
    try:
        from app.observability.health import get_anomaly_detector
        detector = get_anomaly_detector()
        await detector.stop_recovery_loop()
    except Exception as exc:  # noqa: BLE001
        logger.warning("停止异常检测失败: %s", exc)

    # 关闭 LLM 客户端
    try:
        from app.llm.gateway import get_llm_client
        llm = get_llm_client()
        await llm.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("关闭 LLM 客户端失败: %s", exc)

    # 关闭 Redis
    try:
        from app.store import close_redis
        await close_redis()
    except Exception as exc:  # noqa: BLE001
        logger.warning("关闭 Redis 失败: %s", exc)

    # 关闭 ChromaDB
    try:
        from app.store import close_chroma
        close_chroma()
    except Exception as exc:  # noqa: BLE001
        logger.warning("关闭 ChromaDB 失败: %s", exc)

    logger.info("Synapse 已关闭")
