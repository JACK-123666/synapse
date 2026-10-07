"""代码仓库服务：仓库连接管理、变更日志、索引入知识库、写操作二次确认。"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from app.capabilities.knowledge.loaders import TEXT_EXTENSIONS
from app.capabilities.knowledge.service import Principal, current_principal, get_knowledge_service
from app.capabilities.repo.providers import (
    GitHubProvider,
    GitLabProvider,
    LocalGitProvider,
    RepoError,
    RepoProvider,
)
from app.config import get_settings
from app.core.db import session_scope
from app.core.security import decrypt_secret, encrypt_secret
from app.models import PendingAction, RepoConnection, utcnow

logger = logging.getLogger(__name__)

PROVIDERS = ("github", "gitlab", "local")

_CONVENTIONAL_RE = re.compile(
    r"^(?P<type>feat|fix|perf|refactor|docs|test|build|ci|chore|style|revert|ui|debug)"
    r"(?:\((?P<scope>[^)]+)\))?!?:\s*(?P<desc>.+)$",
    re.I,
)
_SECTIONS = OrderedDict([
    ("feat", "新功能"),
    ("fix", "问题修复"),
    ("perf", "性能优化"),
    ("refactor", "重构"),
    ("docs", "文档"),
    ("other", "其他"),
])


def conn_to_dict(conn: RepoConnection) -> Dict[str, Any]:
    """把仓库连接转成对外字典，不包含令牌明文。"""
    return {
        "id": conn.id,
        "name": conn.name,
        "provider": conn.provider,
        "repo": conn.repo,
        "base_url": conn.base_url,
        "default_branch": conn.default_branch,
        "has_token": bool(conn.token_encrypted),
        "allow_write": conn.allow_write,
        "owner_id": conn.owner_id,
        "created_at": conn.created_at.isoformat() if conn.created_at else None,
    }


def action_to_dict(action: PendingAction) -> Dict[str, Any]:
    """把待确认的写操作转成对外字典。"""
    return {
        "id": action.id,
        "kind": action.kind,
        "summary": action.summary,
        "payload": action.payload,
        "status": action.status,
        "result": action.result,
        "created_at": action.created_at.isoformat() if action.created_at else None,
    }


def build_changelog(commits: List[Dict[str, Any]], title: str = "变更日志") -> str:
    """按约定式提交（feat/fix/...）分组生成 Markdown 变更日志。"""
    groups: Dict[str, List[str]] = {key: [] for key in _SECTIONS}
    for c in commits:
        subject = c.get("title", "").strip()
        short = c.get("sha", "")[:7]
        match = _CONVENTIONAL_RE.match(subject)
        if match:
            kind = match.group("type").lower()
            scope = match.group("scope")
            desc = match.group("desc")
            line = f"- {'**' + scope + '**: ' if scope else ''}{desc} ({short})"
            groups[kind if kind in groups else "other"].append(line)
        else:
            groups["other"].append(f"- {subject} ({short})")
    lines = [f"## {title}", ""]
    for key, label in _SECTIONS.items():
        if groups[key]:
            lines.append(f"### {label}")
            lines.extend(groups[key])
            lines.append("")
    if len(lines) == 2:
        lines.append("（没有提交记录）")
    return "\n".join(lines).strip()


class RepoService:
    """仓库服务。"""

    # ---- 连接管理 ----

    async def create_connection(
        self,
        who: Principal,
        *,
        name: str,
        provider: str,
        repo: str,
        token: str = "",
        base_url: str = "",
        default_branch: str = "",
        allow_write: bool = False,
    ) -> Dict[str, Any]:
        """新增仓库连接；访问令牌加密后才入库。"""
        name, provider, repo = name.strip(), provider.strip().lower(), repo.strip()
        if not name or not repo:
            raise RepoError("name 与 repo 不能为空")
        if provider not in PROVIDERS:
            raise RepoError(f"provider 只能是 {PROVIDERS}")
        if provider == "local" and not re.match(r"^(https?://|git@|ssh://)", repo) and not who.is_admin:
            raise PermissionError("只有管理员可以连接服务器本地路径的仓库")
        if allow_write and not who.is_admin:
            raise PermissionError("只有管理员可以开启仓库写操作")
        async with session_scope() as session:
            exists = (
                await session.execute(
                    select(RepoConnection.id).where(
                        RepoConnection.owner_id == who.user_id, RepoConnection.name == name
                    )
                )
            ).first()
            if exists:
                raise RepoError(f"仓库连接「{name}」已存在")
            conn = RepoConnection(
                owner_id=who.user_id,
                name=name,
                provider=provider,
                repo=repo,
                base_url=base_url.strip(),
                token_encrypted=encrypt_secret(token),
                default_branch=default_branch.strip(),
                allow_write=allow_write,
            )
            session.add(conn)
            await session.flush()
            return conn_to_dict(conn)

    async def list_connections(self, who: Principal) -> List[Dict[str, Any]]:
        """列出该用户的仓库连接。"""
        async with session_scope() as session:
            stmt = select(RepoConnection).order_by(RepoConnection.created_at)
            if not who.is_admin:
                stmt = stmt.where(RepoConnection.owner_id == who.user_id)
            return [conn_to_dict(c) for c in (await session.execute(stmt)).scalars().all()]

    async def get_connection(self, who: Principal, name_or_id: str = "") -> RepoConnection:
        """按名称或 ID 获取连接；留空且用户只有一个连接时直接返回它。"""
        key = (name_or_id or "").strip()
        async with session_scope() as session:
            if key:
                conn = await session.get(RepoConnection, key)
                if conn is not None and (conn.owner_id == who.user_id or who.is_admin):
                    return conn
                rows = (
                    await session.execute(select(RepoConnection).where(RepoConnection.name == key))
                ).scalars().all()
                rows = [c for c in rows if c.owner_id == who.user_id or who.is_admin]
                rows.sort(key=lambda c: c.owner_id != who.user_id)
                if rows:
                    return rows[0]
                raise RepoError(f"找不到仓库连接「{key}」，可先用 repo_list 查看已连接的仓库")
            rows = (
                await session.execute(
                    select(RepoConnection).where(RepoConnection.owner_id == who.user_id)
                )
            ).scalars().all()
            if len(rows) == 1:
                return rows[0]
            if not rows:
                raise RepoError("还没有连接任何代码仓库，请先通过 /repos 接口添加")
            raise RepoError("有多个仓库连接，请指定仓库名称：" + "、".join(c.name for c in rows))

    async def update_connection(
        self, who: Principal, conn_id: str, **fields: Any
    ) -> Dict[str, Any]:
        """修改仓库连接的地址、令牌或写权限。"""
        async with session_scope() as session:
            conn = await session.get(RepoConnection, conn_id)
            if conn is None or not (conn.owner_id == who.user_id or who.is_admin):
                raise RepoError("仓库连接不存在")
            if fields.get("allow_write") and not who.is_admin:
                raise PermissionError("只有管理员可以开启仓库写操作")
            for key in ("name", "repo", "base_url", "default_branch", "allow_write"):
                if fields.get(key) is not None:
                    setattr(conn, key, fields[key])
            if fields.get("token") is not None:
                conn.token_encrypted = encrypt_secret(fields["token"])
            return conn_to_dict(conn)

    async def delete_connection(self, who: Principal, conn_id: str) -> None:
        """删除仓库连接。"""
        async with session_scope() as session:
            conn = await session.get(RepoConnection, conn_id)
            if conn is None or not (conn.owner_id == who.user_id or who.is_admin):
                raise RepoError("仓库连接不存在")
            await session.delete(conn)

    def provider_for(self, conn: RepoConnection) -> RepoProvider:
        """根据连接类型构造对应的访问实现。"""
        token = decrypt_secret(conn.token_encrypted)
        if conn.provider == "github":
            return GitHubProvider(conn.repo, token=token, base_url=conn.base_url)
        if conn.provider == "gitlab":
            return GitLabProvider(conn.repo, token=token, base_url=conn.base_url)
        clone_dir = get_settings().data_path("repos", conn.id)
        return LocalGitProvider(conn.repo, clone_dir=clone_dir)

    async def provider(self, who: Principal, name_or_id: str = "") -> tuple:
        """按名称或 ID 取连接，返回 (连接, 访问实现)。"""
        conn = await self.get_connection(who, name_or_id)
        return conn, self.provider_for(conn)

    # ---- 高级功能 ----

    async def changelog(
        self, who: Principal, name_or_id: str = "", *, since: str = "", limit: int = 50, branch: str = ""
    ) -> str:
        """拉取提交并生成分组变更日志。"""
        conn, provider = await self.provider(who, name_or_id)
        commits = await provider.list_commits(
            branch=branch or conn.default_branch, limit=limit, since=since
        )
        title = f"{conn.name} 变更日志" + (f"（{since} 之后）" if since else "")
        return build_changelog(commits, title=title)

    async def index_to_knowledge(
        self,
        who: Principal,
        name_or_id: str,
        kb_name: str,
        *,
        path_prefix: str = "",
        ref: str = "",
        max_files: Optional[int] = None,
    ) -> Dict[str, Any]:
        """把仓库中的文本 / 代码文件写入知识库，之后可做代码问答。"""
        settings = get_settings()
        conn, provider = await self.provider(who, name_or_id)
        knowledge = get_knowledge_service()
        kb = await knowledge.find_kb(who, kb_name)
        if kb is None:
            created = await knowledge.create_kb(who, kb_name, description=f"仓库 {conn.name} 的代码索引")
            kb_id = created["id"]
        else:
            kb_id = kb.id
        limit = max_files or settings.repo_index_max_files
        max_bytes = settings.repo_index_max_file_kb * 1024
        files = await provider.list_files(ref=ref or conn.default_branch)
        candidates = [
            f for f in files
            if f["path"].startswith(path_prefix)
            and ("." + f["path"].rsplit(".", 1)[-1].lower() in TEXT_EXTENSIONS or f["path"].lower().endswith("dockerfile"))
            and (not f.get("size") or f["size"] <= max_bytes)
        ][:limit]
        indexed, failed, chunks = 0, 0, 0
        for f in candidates:
            try:
                content = await provider.read_file(f["path"], ref=ref or conn.default_branch)
                if not content.strip() or len(content.encode("utf-8")) > max_bytes:
                    continue
                doc = await knowledge.ingest_text(
                    who, kb_id, f"{conn.name}/{f['path']}", f"文件: {f['path']}\n\n{content}",
                    source="repo", source_uri=f"repo://{conn.id}/{f['path']}", replace_same_uri=True,
                )
                indexed += 1
                chunks += doc["chunk_count"]
            except Exception as exc:  # noqa: BLE001
                failed += 1
                logger.warning("仓库索引: %s 失败: %s", f["path"], exc)
        return {"kb_id": kb_id, "files": indexed, "failed": failed, "chunks": chunks, "candidates": len(candidates)}

    # ---- 写操作（二次确认） ----

    def write_allowed(self, conn: RepoConnection) -> bool:
        """该连接是否允许写操作：需要全局开关与连接级开关同时打开。"""
        return get_settings().repo_write_enabled and conn.allow_write

    async def propose_action(
        self, who: Principal, conn: RepoConnection, kind: str, payload: Dict[str, Any], summary: str
    ) -> Dict[str, Any]:
        """登记一个待用户确认的写操作，不直接执行。"""
        if not self.write_allowed(conn):
            raise PermissionError(
                "该仓库未开启写操作（需管理员设置 REPO_WRITE_ENABLED=true 并为仓库连接开启 allow_write）"
            )
        async with session_scope() as session:
            action = PendingAction(
                user_id=who.user_id,
                kind=kind,
                payload={"connection_id": conn.id, **payload},
                summary=summary,
            )
            session.add(action)
            await session.flush()
            return action_to_dict(action)

    async def list_actions(self, who: Principal, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出待确认的写操作。"""
        async with session_scope() as session:
            stmt = select(PendingAction).order_by(PendingAction.created_at.desc())
            if not who.is_admin:
                stmt = stmt.where(PendingAction.user_id == who.user_id)
            if status:
                stmt = stmt.where(PendingAction.status == status)
            return [action_to_dict(a) for a in (await session.execute(stmt)).scalars().all()]

    async def _get_pending(self, session, who: Principal, action_id: str) -> PendingAction:
        action = await session.get(PendingAction, action_id)
        if action is None or not (action.user_id == who.user_id or who.is_admin):
            raise RepoError("待确认操作不存在")
        if action.status != "pending":
            raise RepoError(f"该操作已处理（状态: {action.status}）")
        return action

    async def confirm_action(self, who: Principal, action_id: str) -> Dict[str, Any]:
        """执行已经确认的写操作。"""
        async with session_scope() as session:
            action = await self._get_pending(session, who, action_id)
            kind, payload = action.kind, dict(action.payload or {})
        async with session_scope() as session:
            conn = await session.get(RepoConnection, payload.get("connection_id"))
        status, result = "failed", ""
        try:
            if conn is None:
                raise RepoError("仓库连接已删除")
            if not self.write_allowed(conn):
                raise PermissionError("该仓库的写操作已被关闭")
            provider = self.provider_for(conn)
            if kind == "create_issue":
                out = await provider.create_issue(payload["title"], payload.get("body", ""))
            elif kind == "comment":
                out = await provider.comment(int(payload["number"]), payload["body"], payload.get("target", "issue"))
            else:
                raise RepoError(f"未知操作类型: {kind}")
            status, result = "done", str(out)
        except Exception as exc:  # noqa: BLE001
            result = str(exc)
        async with session_scope() as session:
            action = await session.get(PendingAction, action_id)
            action.status, action.result, action.resolved_at = status, result[:2000], utcnow()
            return action_to_dict(action)

    async def reject_action(self, who: Principal, action_id: str) -> Dict[str, Any]:
        """拒绝一个待确认的写操作。"""
        async with session_scope() as session:
            action = await self._get_pending(session, who, action_id)
            action.status, action.resolved_at = "rejected", utcnow()
            return action_to_dict(action)

    # 便捷：基于请求上下文

    async def provider_for_current_user(self, name_or_id: str = "") -> tuple:
        """用当前用户身份取连接与访问实现。"""
        return await self.provider(current_principal(), name_or_id)


_service: Optional[RepoService] = None


def get_repo_service() -> RepoService:
    """获取仓库服务单例。"""
    global _service
    if _service is None:
        _service = RepoService()
    return _service
