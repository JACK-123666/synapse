"""请求级上下文。

使用 contextvars 在一次请求（或一次定时任务执行）内传递当前用户、会话、
模型覆盖等信息。工具函数、LLM 工厂可以直接读取，无需层层传参，
也不会像修改全局单例那样在并发请求之间互相干扰。
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

#: 本地管理员（关闭鉴权时使用）的固定用户 ID
LOCAL_ADMIN_ID = "local-admin"


@dataclass
class RequestContext:
    """一次请求的上下文。

    Attributes:
        user_id: 当前登录用户 ID
        username: 用户名
        role: 角色 admin / user
        session_id: 会话 ID（已按用户做命名空间隔离）
        memory_user_id: 记忆与画像使用的用户标识（关闭鉴权时沿用请求里的 user_id）
        model_override: 本次请求临时指定的模型名
        web_search: 是否允许联网
        allowed_tools: 额外限定的工具名（None 表示不额外限定）
    """

    user_id: str = LOCAL_ADMIN_ID
    username: str = "admin"
    role: str = "admin"
    session_id: str = ""
    memory_user_id: Optional[str] = None
    model_override: Optional[str] = None
    web_search: bool = True
    allowed_tools: Optional[List[str]] = None
    extras: dict = field(default_factory=dict)

    @property
    def is_admin(self) -> bool:
        """当前用户是否管理员。"""
        return self.role == "admin"


_current: ContextVar[Optional[RequestContext]] = ContextVar(
    "synapse_request_context", default=None
)


def get_request_context() -> RequestContext:
    """获取当前请求上下文；不在请求内时返回默认（本地管理员）上下文。"""
    ctx = _current.get()
    return ctx if ctx is not None else RequestContext()


def has_request_context() -> bool:
    """当前是否处于某个请求上下文中。"""
    return _current.get() is not None


def set_request_context(ctx: RequestContext) -> Token:
    """设置请求上下文，返回用于还原的 Token。"""
    return _current.set(ctx)


def reset_request_context(token: Token) -> None:
    """还原到设置之前的上下文。"""
    _current.reset(token)


@contextmanager
def request_context(ctx: RequestContext) -> Iterator[RequestContext]:
    """在 with 块内设置请求上下文，退出时自动还原。"""
    token = _current.set(ctx)
    try:
        yield ctx
    finally:
        _current.reset(token)
