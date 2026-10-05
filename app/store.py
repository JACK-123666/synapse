"""共享存储连接层。

统一管理 Redis 异步客户端和 ChromaDB HTTP 客户端的连接生命周期。
所有模块通过此模块获取连接单例，避免重复创建。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from redis.asyncio import Redis, from_url

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Redis 异步客户端单例

_redis: Optional[Redis] = None


async def get_redis() -> Redis:
    """获取 Redis 异步客户端单例。

    使用 redis.asyncio 库，所有操作均为异步。
    首次调用时创建连接，后续复用。
    """
    global _redis
    if _redis is None:
        settings = get_settings()
        _redis = from_url(
            settings.redis_url,
            decode_responses=True,
            encoding="utf-8",
            socket_connect_timeout=5,
        )
        logger.info("Redis 客户端已初始化: %s", settings.redis_url)
    return _redis


async def close_redis() -> None:
    """关闭 Redis 连接。"""
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None
        logger.info("Redis 客户端已关闭")


# ChromaDB 客户端单例

_chroma: Optional[chromadb.api.ClientAPI] = None
_chroma_lock = threading.Lock()
#: 上次连接失败的时间；冷却期内直接报错，避免每个请求都同步重连阻塞
_chroma_failed_at: float = 0.0
_CHROMA_RETRY_INTERVAL = 30.0


def get_chroma() -> chromadb.api.ClientAPI:
    """获取 ChromaDB HTTP 客户端单例。

    ChromaDB 官方 Python 客户端以 HTTP 模式连接 chromadb 服务。
    注意：ChromaDB 客户端本身是同步的，异步上下文中需用
    asyncio.to_thread 包装调用。

    连接失败后 30 秒内不再重试，直接抛出异常，由调用方降级。

    Returns:
        ChromaDB 客户端实例
    """
    global _chroma, _chroma_failed_at
    if _chroma is not None:
        return _chroma
    with _chroma_lock:
        if _chroma is not None:
            return _chroma
        if _chroma_failed_at and time.monotonic() - _chroma_failed_at < _CHROMA_RETRY_INTERVAL:
            raise RuntimeError("ChromaDB 暂不可用（连接失败冷却中）")
        settings = get_settings()
        try:
            _chroma = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        except Exception:
            _chroma_failed_at = time.monotonic()
            raise
        _chroma_failed_at = 0.0
        logger.info(
            "ChromaDB 客户端已初始化: %s:%s",
            settings.chroma_host, settings.chroma_port,
        )
    return _chroma


def close_chroma() -> None:
    """关闭 ChromaDB 连接。"""
    global _chroma
    if _chroma is not None:
        _chroma = None
        logger.info("ChromaDB 客户端已关闭")
