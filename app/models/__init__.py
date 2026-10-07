"""ORM 数据模型。"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


def new_id() -> str:
    """生成 32 位十六进制的随机主键。"""
    return uuid.uuid4().hex


def utcnow() -> datetime:
    """当前 UTC 时间（带时区），作为所有表的时间戳默认值。"""
    return datetime.now(timezone.utc)


class User(Base):
    """用户。role: admin / user。"""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=new_id)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(128), default="")
    role: Mapped[str] = mapped_column(String(16), default="user")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ApiKey(Base):
    """API Key：只保存 SHA-256 哈希，明文仅在创建时返回一次。"""

    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(64), default="default")
    prefix: Mapped[str] = mapped_column(String(16))
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class KnowledgeBase(Base):
    """知识库。visibility: private（仅自己）/ shared（所有用户可检索）。"""

    __tablename__ = "knowledge_bases"
    __table_args__ = (UniqueConstraint("owner_id", "name", name="uq_kb_owner_name"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str] = mapped_column(Text, default="")
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    visibility: Mapped[str] = mapped_column(String(16), default="private")
    collection_name: Mapped[str] = mapped_column(String(128), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Document(Base):
    """知识库中的文档。source: upload / url / text / repo。"""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=new_id)
    kb_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True
    )
    filename: Mapped[str] = mapped_column(String(512))
    source: Mapped[str] = mapped_column(String(16), default="upload")
    source_uri: Mapped[str] = mapped_column(String(2048), default="")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default="ready")
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class RepoConnection(Base):
    """代码仓库连接。provider: github / gitlab / local。"""

    __tablename__ = "repo_connections"
    __table_args__ = (UniqueConstraint("owner_id", "name", name="uq_repo_owner_name"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=new_id)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(128))
    provider: Mapped[str] = mapped_column(String(16))
    # github/gitlab: "owner/repo"；local: 本地路径或 clone 地址
    repo: Mapped[str] = mapped_column(String(1024))
    base_url: Mapped[str] = mapped_column(String(512), default="")
    token_encrypted: Mapped[str] = mapped_column(Text, default="")
    default_branch: Mapped[str] = mapped_column(String(128), default="")
    allow_write: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class PendingAction(Base):
    """待用户确认的写操作。status: pending / done / rejected / failed。"""

    __tablename__ = "pending_actions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(64))
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    summary: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(16), default="pending")
    result: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class Schedule(Base):
    """定时任务。kind: prompt / web_watch / kb_sync。"""

    __tablename__ = "schedules"
    __table_args__ = (UniqueConstraint("owner_id", "name", name="uq_schedule_owner_name"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=new_id)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(128))
    kind: Mapped[str] = mapped_column(String(16), default="prompt")
    cron: Mapped[str] = mapped_column(String(64))
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    webhook_url: Mapped[str] = mapped_column(String(2048), default="")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    session_id: Mapped[str] = mapped_column(String(128), default="")
    last_run_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_status: Mapped[str] = mapped_column(String(16), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ScheduleRun(Base):
    """定时任务执行记录。status: running / success / failed / skipped。"""

    __tablename__ = "schedule_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=new_id)
    schedule_id: Mapped[str] = mapped_column(
        ForeignKey("schedules.id", ondelete="CASCADE"), index=True
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="running")
    output: Mapped[str] = mapped_column(Text, default="")


class PluginState(Base):
    """本地插件启停状态。"""

    __tablename__ = "plugin_states"

    name: Mapped[str] = mapped_column(String(128), primary_key=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class McpServer(Base):
    """MCP 服务配置。transport: stdio / streamable_http / sse。

    config 示例：
        stdio: {"command": "npx", "args": [...], "env": {...}}
        http:  {"url": "https://...", "headers": {...}}
    """

    __tablename__ = "mcp_servers"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    transport: Mapped[str] = mapped_column(String(32), default="streamable_http")
    config: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ToolPolicy(Base):
    """角色工具白名单（fnmatch 通配模式列表）。"""

    __tablename__ = "tool_policies"

    role: Mapped[str] = mapped_column(String(16), primary_key=True)
    patterns: Mapped[List[str]] = mapped_column(JSON, default=list)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


__all__ = [
    "new_id",
    "utcnow",
    "User",
    "ApiKey",
    "KnowledgeBase",
    "Document",
    "RepoConnection",
    "PendingAction",
    "Schedule",
    "ScheduleRun",
    "PluginState",
    "McpServer",
    "ToolPolicy",
]
