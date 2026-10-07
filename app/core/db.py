"""关系型数据库（SQLAlchemy 2.0 异步）。

存储用户、API Key、知识库元数据、仓库连接、定时任务、插件状态等结构化数据。
默认使用 data_dir 下的 SQLite（个人部署零配置），配置 DATABASE_URL 后可切换 PostgreSQL。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """ORM 基类。"""


_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def _enable_sqlite_fk(dbapi_connection, _record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def get_engine() -> AsyncEngine:
    """获取数据库引擎单例。"""
    global _engine, _session_factory
    if _engine is None:
        url = get_settings().resolved_database_url()
        _engine = create_async_engine(url, pool_pre_ping=True)
        if url.startswith("sqlite"):
            event.listen(_engine.sync_engine, "connect", _enable_sqlite_fk)
        _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
        logger.info("数据库引擎已初始化: %s", url.split("@")[-1])
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """获取异步会话工厂；首次调用时创建引擎。"""
    get_engine()
    assert _session_factory is not None
    return _session_factory


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """数据库会话上下文：正常退出提交，异常回滚。"""
    session = get_session_factory()()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI 依赖：每个请求一个数据库会话。"""
    async with session_scope() as session:
        yield session


async def init_db() -> None:
    """创建全部数据表（幂等）。"""
    import app.models  # noqa: F401  确保模型已注册到 Base.metadata

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("数据库表结构已就绪")


async def close_db() -> None:
    """释放引擎与连接池（应用关闭时调用）。"""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("数据库引擎已关闭")


def reset_engine_for_tests() -> None:
    """测试用：丢弃引擎单例（不做 dispose）。"""
    global _engine, _session_factory
    _engine = None
    _session_factory = None
