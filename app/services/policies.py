"""角色工具白名单：持久化到数据库，启动时加载到工具注册表。"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from sqlalchemy import select

from app.core.db import session_scope
from app.models import ToolPolicy, utcnow
from app.tools.registry import get_tool_registry

logger = logging.getLogger(__name__)


async def load_tool_policies() -> Dict[str, List[str]]:
    registry = get_tool_registry()
    async with session_scope() as session:
        rows = (await session.execute(select(ToolPolicy))).scalars().all()
        policies = {row.role: list(row.patterns or []) for row in rows}
    for role, patterns in policies.items():
        registry.set_policy(role, patterns)
    if policies:
        logger.info("已加载工具白名单: %s", policies)
    return policies


async def set_tool_policy(role: str, patterns: Optional[List[str]]) -> None:
    """设置角色白名单；patterns 为 None 表示取消限制（可使用全部工具）。"""
    async with session_scope() as session:
        row = await session.get(ToolPolicy, role)
        if patterns is None:
            if row is not None:
                await session.delete(row)
        elif row is None:
            session.add(ToolPolicy(role=role, patterns=list(patterns)))
        else:
            row.patterns = list(patterns)
            row.updated_at = utcnow()
    get_tool_registry().set_policy(role, patterns)
