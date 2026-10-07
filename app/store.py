"""共享存储连接层。

统一管理 Redis 与 ChromaDB 的连接生命周期。两者都支持"自动退化"：

- Redis  ：连不上真实服务时退化为进程内内存实现（fakeredis），
           短期记忆与用户画像照常可用，但重启后丢失。
- Chroma ：连不上真实服务时退化为内嵌持久化客户端（本地文件），
           向量检索与长期记忆照常可用，数据落盘在 data/chroma。

这样单机开发不必先起 Docker 就能跑全部功能；部署到多 worker 时把
REDIS_MODE / CHROMA_MODE 显式设为 server 即可强制依赖外部服务。
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from redis.asyncio import Redis, from_url

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

#: 当前是否运行在退化后端上（供 /health 如实汇报）
redis_is_fallback: bool = False
chroma_is_fallback: bool = False


# Redis

#: 单例。启动阶段（main.py lifespan）就会调用一次，服务流量到来前必然已就绪，
#: 因此不需要加锁：asyncio.Lock 会在首次 acquire 时绑定事件循环，跨 loop 反而有风险。
_redis: Optional[Redis] = None


def _server_redis(settings: Settings) -> Redis:
    return from_url(
        settings.redis_url,
        decode_responses=True,
        encoding="utf-8",
        socket_connect_timeout=settings.redis_connect_timeout,
    )


def _memory_redis() -> Redis:
    """进程内内存实现（fakeredis），API 与真实 Redis 一致。"""
    import fakeredis

    return fakeredis.FakeAsyncRedis(decode_responses=True)


async def get_redis() -> Redis:
    """获取 Redis 客户端（异步）。

    REDIS_MODE=server 时只连真实服务；memory 时只用内存实现；
    auto（默认）先探测真实服务，失败则退化为内存实现。
    """
    global _redis, redis_is_fallback
    if _redis is not None:
        return _redis

    settings = get_settings()
    mode = str(settings.redis_mode or "auto").lower()

    if mode == "memory":
        _redis, redis_is_fallback = _memory_redis(), True
        logger.warning("Redis: REDIS_MODE=memory，使用进程内内存实现（重启即丢）")
        return _redis

    client = _server_redis(settings)
    if mode == "server":
        _redis = client
        logger.info("Redis 客户端已初始化: %s", settings.redis_url)
        return _redis

    # auto：先探测真实服务
    try:
        await asyncio.wait_for(client.ping(), timeout=settings.redis_connect_timeout)
    except Exception as exc:  # noqa: BLE001
        try:
            await client.aclose()
        except Exception:  # noqa: BLE001
            pass
        _redis, redis_is_fallback = _memory_redis(), True
        logger.warning(
            "Redis 不可用(%s)，已退化为进程内内存实现：短期记忆与用户画像"
            "重启后会丢失。多 worker 部署请启动 Redis 并设 REDIS_MODE=server",
            exc,
        )
        return _redis

    _redis = client
    logger.info("Redis 连接正常: %s", settings.redis_url)
    return _redis


async def close_redis() -> None:
    """关闭 Redis 连接。"""
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None
        logger.info("Redis 连接已关闭")


# ChromaDB

_chroma: Optional[chromadb.api.ClientAPI] = None
_chroma_lock = threading.Lock()
#: 上次连接失败的时间；冷却期内直接报错，避免每个请求都同步重连阻塞
_chroma_failed_at: float = 0.0
_CHROMA_RETRY_INTERVAL = 30.0


def _chroma_embedded(settings: Settings):
    """内嵌持久化客户端：向量数据存本地文件，无需 Chroma 服务端。"""
    path = settings.data_path("chroma")
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(path),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_chroma():
    """获取 ChromaDB 客户端（同步）。

    CHROMA_MODE=embedded 直接用内嵌持久化；server 只连真实服务；
    auto（默认）先试 HTTP 服务，失败则退化为内嵌持久化。

    注意：ChromaDB 客户端本身是同步的，异步上下文中需用 asyncio.to_thread 包装。
    """
    global _chroma, _chroma_failed_at, chroma_is_fallback
    if _chroma is not None:
        return _chroma

    with _chroma_lock:
        if _chroma is not None:
            return _chroma
        settings = get_settings()
        mode = str(settings.chroma_mode or "auto").lower()

        if mode == "embedded":
            _chroma, chroma_is_fallback = _chroma_embedded(settings), True
            logger.info("ChromaDB: 使用内嵌持久化模式 (%s)", settings.data_path("chroma"))
            return _chroma

        if _chroma_failed_at and time.monotonic() - _chroma_failed_at < _CHROMA_RETRY_INTERVAL:
            raise RuntimeError("ChromaDB 暂不可用（连接失败冷却中）")

        try:
            client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            client.heartbeat()
        except Exception as exc:  # noqa: BLE001
            if mode == "server":
                _chroma_failed_at = time.monotonic()
                raise
            _chroma, chroma_is_fallback = _chroma_embedded(settings), True
            logger.warning(
                "ChromaDB 服务不可用(%s)，已退化为内嵌持久化模式：向量数据保存在 %s。"
                "多 worker 部署请启动 Chroma 服务并设 CHROMA_MODE=server",
                exc, settings.data_path("chroma"),
            )
            return _chroma

        _chroma = client
        _chroma_failed_at = 0.0
        logger.info("ChromaDB 连接正常: %s:%s", settings.chroma_host, settings.chroma_port)
        return _chroma


def close_chroma() -> None:
    """关闭 ChromaDB 连接。"""
    global _chroma
    if _chroma is not None:
        _chroma = None
        logger.info("ChromaDB 连接已关闭")
