"""代码仓库能力的 LangChain 工具。

参数 repo 为仓库连接名称；用户只有一个仓库连接时可以留空。
写操作工具不会直接执行，而是生成「待确认操作」，用户确认后才真正调用仓库 API。
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Awaitable, Callable, Dict, List

from langchain_core.tools import tool

from app.capabilities.knowledge.service import current_principal
from app.capabilities.repo.providers import RepoError
from app.capabilities.repo.service import get_repo_service

_MAX_OUTPUT = 6000


def _clip(text: str) -> str:
    if len(text) > _MAX_OUTPUT:
        return text[:_MAX_OUTPUT] + f"\n...（输出过长已截断，共 {len(text)} 字符）"
    return text


def _safe(fn: Callable[..., Awaitable[str]]) -> Callable[..., Awaitable[str]]:
    """把仓库错误转成文本返回给模型，避免整个 Agent 失败。"""

    @wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> str:
        try:
            return _clip(await fn(*args, **kwargs))
        except (RepoError, PermissionError) as exc:
            return f"操作失败: {exc}"

    return wrapper


def _fmt_items(items: List[Dict[str, Any]], fields: List[str]) -> str:
    if not items:
        return "（无）"
    lines = []
    for it in items:
        lines.append(" | ".join(str(it.get(f, "")) for f in fields if it.get(f) not in (None, "")))
    return "\n".join(f"- {line}" for line in lines)


@tool("repo_list")
@_safe
async def repo_list_tool() -> str:
    """列出当前用户已连接的代码仓库（名称、类型、地址）。"""
    conns = await get_repo_service().list_connections(current_principal())
    if not conns:
        return "还没有连接任何代码仓库。"
    return "\n".join(
        f"- {c['name']}（{c['provider']}: {c['repo']}{'，可写' if c['allow_write'] else ''}）" for c in conns
    )


@tool("repo_commits")
@_safe
async def repo_commits_tool(repo: str = "", branch: str = "", limit: int = 20, path: str = "", since: str = "") -> str:
    """查看仓库的提交记录。

    Args:
        repo: 仓库连接名称（只有一个仓库时可留空）
        branch: 分支名，留空为默认分支
        limit: 条数，默认 20
        path: 只看某个文件 / 目录的提交
        since: 起始时间（ISO 格式，如 2026-10-01）
    """
    conn, provider = await get_repo_service().provider_for_current_user(repo)
    commits = await provider.list_commits(
        branch=branch or conn.default_branch, limit=max(1, min(int(limit or 20), 100)), path=path, since=since
    )
    return f"仓库 {conn.name} 的提交记录：\n" + _fmt_items(
        [{**c, "sha": c["sha"][:7]} for c in commits], ["sha", "date", "author", "title"]
    )


@tool("repo_commit_detail")
@_safe
async def repo_commit_detail_tool(sha: str, repo: str = "") -> str:
    """查看某次提交的详情与代码改动（diff）。

    Args:
        sha: 提交哈希（可用前 7 位）
        repo: 仓库连接名称
    """
    _, provider = await get_repo_service().provider_for_current_user(repo)
    c = await provider.get_commit(sha)
    files = "\n\n".join(
        f"### {f['path']} {f.get('status', '')} (+{f.get('additions', '?')}/-{f.get('deletions', '?')})\n{f.get('patch', '')}"
        for f in c.get("files", [])
    )
    return f"提交 {c['sha'][:7]} by {c['author']} @ {c['date']}\n{c['message']}\n\n{files}"


@tool("repo_issues")
@_safe
async def repo_issues_tool(repo: str = "", state: str = "open", limit: int = 20) -> str:
    """列出仓库的 Issue。

    Args:
        repo: 仓库连接名称
        state: open / closed / all
        limit: 条数，默认 20
    """
    _, provider = await get_repo_service().provider_for_current_user(repo)
    issues = await provider.list_issues(state=state, limit=max(1, min(int(limit or 20), 100)))
    return _fmt_items(
        [{**i, "number": f"#{i['number']}", "labels": ",".join(i.get("labels", []))} for i in issues],
        ["number", "state", "title", "author", "labels"],
    )


@tool("repo_issue_detail")
@_safe
async def repo_issue_detail_tool(number: int, repo: str = "") -> str:
    """查看 Issue 详情与评论。

    Args:
        number: Issue 编号
        repo: 仓库连接名称
    """
    _, provider = await get_repo_service().provider_for_current_user(repo)
    i = await provider.get_issue(int(number))
    comments = "\n".join(f"- {c['author']}: {c['body']}" for c in i.get("comments", []))
    return f"#{i['number']} [{i['state']}] {i['title']}（{i['author']}）\n{i['url']}\n\n{i['body']}\n\n评论:\n{comments or '（无）'}"


@tool("repo_pulls")
@_safe
async def repo_pulls_tool(repo: str = "", state: str = "open", limit: int = 20) -> str:
    """列出仓库的 Pull Request / Merge Request。

    Args:
        repo: 仓库连接名称
        state: open / closed / all（GitLab 还支持 merged）
        limit: 条数，默认 20
    """
    _, provider = await get_repo_service().provider_for_current_user(repo)
    pulls = await provider.list_pulls(state=state, limit=max(1, min(int(limit or 20), 100)))
    return _fmt_items(
        [{**p, "number": f"#{p['number']}", "branch": f"{p['head']} -> {p['base']}"} for p in pulls],
        ["number", "state", "title", "author", "branch"],
    )


@tool("repo_pull_detail")
@_safe
async def repo_pull_detail_tool(number: int, repo: str = "") -> str:
    """查看 PR / MR 详情及改动文件与 diff，可用于总结一个 PR 改了什么。

    Args:
        number: PR / MR 编号
        repo: 仓库连接名称
    """
    _, provider = await get_repo_service().provider_for_current_user(repo)
    p = await provider.get_pull(int(number))
    files = "\n\n".join(
        f"### {f['path']} {f.get('status', '')}\n{f.get('patch', '')}" for f in p.get("files", [])
    )
    return f"#{p['number']} [{p['state']}] {p['title']}（{p['author']}，{p['head']} -> {p['base']}）\n{p['url']}\n\n{p['body']}\n\n改动:\n{files}"


@tool("repo_read_file")
@_safe
async def repo_read_file_tool(path: str, repo: str = "", ref: str = "") -> str:
    """读取仓库中某个文件的内容。

    Args:
        path: 文件路径（相对仓库根目录）
        repo: 仓库连接名称
        ref: 分支 / 标签 / 提交，留空为默认分支
    """
    conn, provider = await get_repo_service().provider_for_current_user(repo)
    return await provider.read_file(path, ref=ref or conn.default_branch)


@tool("repo_list_dir")
@_safe
async def repo_list_dir_tool(path: str = "", repo: str = "", ref: str = "") -> str:
    """列出仓库某个目录下的文件与子目录。

    Args:
        path: 目录路径，留空为根目录
        repo: 仓库连接名称
        ref: 分支 / 标签 / 提交
    """
    conn, provider = await get_repo_service().provider_for_current_user(repo)
    items = await provider.list_dir(path, ref=ref or conn.default_branch)
    return _fmt_items(items, ["type", "path", "size"])


@tool("repo_search_code")
@_safe
async def repo_search_code_tool(query: str, repo: str = "") -> str:
    """在仓库代码中搜索关键字（GitHub 搜索需要配置访问令牌）。

    Args:
        query: 搜索关键字
        repo: 仓库连接名称
    """
    _, provider = await get_repo_service().provider_for_current_user(repo)
    results = await provider.search_code(query)
    return _fmt_items(results, ["path", "line", "snippet", "url"])


@tool("repo_changelog")
@_safe
async def repo_changelog_tool(repo: str = "", since: str = "", limit: int = 50, branch: str = "") -> str:
    """按约定式提交（feat/fix/docs...）分组生成变更日志（Markdown）。

    Args:
        repo: 仓库连接名称
        since: 起始时间（ISO 格式，如 2026-09-28）
        limit: 最多统计多少条提交，默认 50
        branch: 分支名
    """
    return await get_repo_service().changelog(
        current_principal(), repo, since=since, limit=max(1, min(int(limit or 50), 200)), branch=branch
    )


@tool("repo_index_to_knowledge")
@_safe
async def repo_index_tool(knowledge_base: str, repo: str = "", path_prefix: str = "") -> str:
    """把仓库的代码 / 文档文件索引进知识库，之后可以用知识问答做代码问答。

    Args:
        knowledge_base: 目标知识库名称（不存在时自动创建）
        repo: 仓库连接名称
        path_prefix: 只索引该路径前缀下的文件，如 app/
    """
    result = await get_repo_service().index_to_knowledge(
        current_principal(), repo, knowledge_base, path_prefix=path_prefix
    )
    return (
        f"已索引 {result['files']} 个文件（{result['chunks']} 个片段）到知识库「{knowledge_base}」，"
        f"失败 {result['failed']} 个。"
    )


@tool("repo_create_issue")
@_safe
async def repo_create_issue_tool(title: str, body: str = "", repo: str = "") -> str:
    """创建 Issue（写操作：不会立即执行，会生成待确认操作，需用户确认）。

    Args:
        title: Issue 标题
        body: Issue 内容（Markdown）
        repo: 仓库连接名称
    """
    service = get_repo_service()
    who = current_principal()
    conn = await service.get_connection(who, repo)
    action = await service.propose_action(
        who, conn, "create_issue", {"title": title, "body": body},
        summary=f"在 {conn.name} 创建 Issue：{title}",
    )
    return (
        f"已生成待确认操作（ID: {action['id']}）：{action['summary']}。"
        f"请用户调用 POST /repos/actions/{action['id']}/confirm 确认后执行。"
    )


@tool("repo_comment")
@_safe
async def repo_comment_tool(number: int, body: str, repo: str = "", target: str = "issue") -> str:
    """在 Issue 或 PR / MR 下发表评论（写操作：生成待确认操作，需用户确认）。

    Args:
        number: Issue / PR 编号
        body: 评论内容
        repo: 仓库连接名称
        target: issue 或 mr（GitLab 合并请求）
    """
    service = get_repo_service()
    who = current_principal()
    conn = await service.get_connection(who, repo)
    action = await service.propose_action(
        who, conn, "comment", {"number": int(number), "body": body, "target": target},
        summary=f"在 {conn.name} 的 #{number} 发表评论",
    )
    return (
        f"已生成待确认操作（ID: {action['id']}）：{action['summary']}。"
        f"请用户调用 POST /repos/actions/{action['id']}/confirm 确认后执行。"
    )


READ_TOOLS = [
    repo_list_tool,
    repo_commits_tool,
    repo_commit_detail_tool,
    repo_issues_tool,
    repo_issue_detail_tool,
    repo_pulls_tool,
    repo_pull_detail_tool,
    repo_read_file_tool,
    repo_list_dir_tool,
    repo_search_code_tool,
    repo_changelog_tool,
    repo_index_tool,
]
WRITE_TOOLS = [repo_create_issue_tool, repo_comment_tool]
