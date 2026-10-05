"""Synapse 平台 FastAPI 入口。

负责：
- 配置日志系统
- 注册 API 路由
- 生命周期（lifespan）：初始化各模块连接、数据库、注册能力 / 插件 / Agent / 路由、
  启动异常检测与定时任务；关闭时按相反顺序释放资源
- CORS 中间件

启动方式：
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.auth import router as auth_router
from app.api.auth import users_router
from app.api.chat import router as chat_router
from app.api.knowledge import router as knowledge_router
from app.api.memory import router as memory_router
from app.api.plugins import router as plugins_router
from app.api.repos import router as repos_router
from app.api.schedules import router as schedules_router
from app.api.system import router as system_router
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
logging.getLogger("apscheduler").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


async def startup() -> None:
    """应用启动时初始化所有模块连接和配置。

    初始化顺序：
    1. 加载配置
    2. 连接 LLM 客户端
    3. 预检 Redis / ChromaDB
    4. 初始化数据库、初始管理员、工具白名单
    5. 注册内置能力（Agent、工具、意图、路由）
    6. 加载本地插件与 MCP 服务
    7. 初始化向量意图索引（此时意图目录已完整）
    8. 启动异常检测后台任务
    """
    settings = get_settings()
    logger.info("=" * 60)
    logger.info("Synapse v2.0.0 正在启动...")
    # 配置来源：docker 走 compose env_file 注入环境变量；本地开发走 .env 文件
    logger.info(
        "配置来源: %s",
        "环境变量(compose env_file 注入)" if os.environ.get("LLM_PROVIDER")
        else ".env 文件(本地开发)",
    )
    logger.info("LLM Provider: %s, Model: %s",
                settings.llm_provider, settings.llm_model)
    logger.info("LLM BaseURL: %s", settings.llm_base_url)
    logger.info("LLM API Key: %s", "已配置" if settings.llm_api_key else "(empty)")
    logger.info("DeepSeek URL: %s, Model: %s",
                settings.deepseek_base_url, settings.deepseek_model)
    logger.info("鉴权: %s", "开启" if settings.auth_enabled else "关闭（本地管理员模式）")
    logger.info("=" * 60)

    # LLM 配置（无需建连接池：ChatModel 由工厂按需创建并缓存）
    from app.llm.config import get_llm_config
    logger.info("[OK] LLM 配置: %s", get_llm_config().snapshot())

    # 预检 Redis
    try:
        from app.store import get_redis
        redis = await get_redis()
        await asyncio.wait_for(redis.ping(), timeout=5)
        logger.info("[OK] Redis 连接正常")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[SKIP] Redis 不可用: %s", exc)

    # 预检 ChromaDB（同步客户端，放到线程中执行）
    try:
        from app.store import get_chroma
        chroma = await asyncio.to_thread(get_chroma)
        await asyncio.to_thread(chroma.heartbeat)
        logger.info("[OK] ChromaDB 连接正常")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[SKIP] ChromaDB 不可用: %s", exc)

    # 数据库 + 初始管理员 + 工具白名单
    try:
        from app.core.db import init_db
        from app.services.policies import load_tool_policies
        from app.services.users import bootstrap_admin

        await init_db()
        await bootstrap_admin()
        await load_tool_policies()
        logger.info("[OK] 数据库已就绪")
    except Exception as exc:  # noqa: BLE001
        logger.error("[FAIL] 数据库初始化失败: %s", exc)

    # 注册内置能力（Agent + 工具 + 意图 + 路由）
    try:
        from app.capabilities import builtin_capabilities
        from app.capabilities.manager import get_capability_manager
        from app.router.pool import get_agent_registry

        manager = get_capability_manager()
        for capability in builtin_capabilities():
            try:
                await manager.register(capability)
            except Exception as exc:  # noqa: BLE001
                logger.error("[FAIL] 内置能力 '%s' 注册失败: %s", capability.name, exc)

        registry = get_agent_registry()
        logger.info("[OK] 已注册 %d 个 Agent", len(registry.get_all_agents()))
        logger.info("[OK] 路由表已注册: %s", list(registry.get_routes()))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[SKIP] Agent/路由注册失败: %s", exc)

    # 本地插件 + MCP
    try:
        from app.plugins.manager import get_plugin_manager
        await get_plugin_manager().load_all()
        logger.info("[OK] 本地插件已加载")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[SKIP] 本地插件加载失败: %s", exc)

    if settings.mcp_enabled:
        try:
            from app.plugins.mcp import get_mcp_manager
            await get_mcp_manager().load_all()
            logger.info("[OK] MCP 服务已加载")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[SKIP] MCP 服务加载失败: %s", exc)

    # 初始化向量意图索引
    try:
        from app.intent.blend import get_intent_fusion
        fusion = get_intent_fusion()
        await fusion.initialize()
        logger.info("[OK] 意图向量索引已就绪")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[SKIP] 意图向量索引初始化失败: %s", exc)

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


async def shutdown() -> None:
    """应用关闭时清理资源。"""
    logger.info("Synapse 正在关闭...")

    # 先等后台任务（记忆压缩等）落盘，此时 Redis / 数据库仍然可用
    try:
        from app.core.tasks import drain
        await drain()
    except Exception as exc:  # noqa: BLE001
        logger.warning("等待后台任务失败: %s", exc)

    # 关闭能力（定时任务调度器等）
    try:
        from app.capabilities.manager import get_capability_manager
        await get_capability_manager().shutdown_all()
    except Exception as exc:  # noqa: BLE001
        logger.warning("关闭能力失败: %s", exc)

    # 关闭 MCP
    try:
        from app.plugins.mcp import get_mcp_manager
        await get_mcp_manager().close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("关闭 MCP 失败: %s", exc)

    # 停止异常检测后台任务
    try:
        from app.observability.health import get_anomaly_detector
        detector = get_anomaly_detector()
        await detector.stop_recovery_loop()
    except Exception as exc:  # noqa: BLE001
        logger.warning("停止异常检测失败: %s", exc)

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

    # 关闭数据库
    try:
        from app.core.db import close_db
        await close_db()
    except Exception as exc:  # noqa: BLE001
        logger.warning("关闭数据库失败: %s", exc)

    logger.info("Synapse 已关闭")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await startup()
    try:
        yield
    finally:
        await shutdown()


# FastAPI 应用实例

_app_settings = get_settings()

app = FastAPI(
    title="Synapse · 全能智能助手平台",
    description="""
## 👋 欢迎使用 Synapse

一个带**意图识别**、**记忆管理**、**故障自愈**的多 Agent 智能助手（基于 LangChain）。

### 能力

- **知识问答（RAG）**：创建知识库、上传文档、带引用回答
- **记忆检索**：短期对话 + 长期摘要 + 用户画像
- **网页抓取 / 联网搜索**
- **代码仓库助手**：GitHub / GitLab / 本地 Git
- **定时任务**：按 cron 让助手执行任务，结果可推送到 Webhook
- **插件**：本地 Python 插件 + MCP 服务

### 怎么用

1. 开启鉴权时先调 `POST /auth/login` 拿到 token（或用 API Key）
2. 调 `POST /chat`（或流式 `POST /chat/stream`）发消息
3. 拿到的 `session_id` 原样传回，就能多轮对话
4. 随时调 `GET /health` 看各模块是否正常
""",
    version="2.0.0",
    docs_url="/docs" if _app_settings.docs_enabled else None,
    redoc_url="/redoc" if _app_settings.docs_enabled else None,
    lifespan=lifespan,
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
app.include_router(system_router, prefix="")
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(knowledge_router)
app.include_router(memory_router)
app.include_router(repos_router)
app.include_router(schedules_router)
app.include_router(plugins_router)

# 静态文件 — 聊天首页（必须最后挂载，避免覆盖 API 路由）
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
