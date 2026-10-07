"""代码仓库访问层：GitHub / GitLab（REST API）与本地 Git（git 子进程）。

所有方法返回结构化 dict，由工具层格式化为文本；出错时抛出 RepoError。
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

_MAX_PATCH_CHARS = 4000
_HTTP_TIMEOUT = 20.0
_GIT_TIMEOUT = 120.0
#: 合法的 ref / sha：不允许以 - 开头（防止被当成 git 参数）
_REF_RE = re.compile(r"^[\w./@{}~^+-]+$")


class RepoError(Exception):
    """仓库访问失败。"""


def _check_ref(ref: str) -> str:
    ref = (ref or "").strip()
    if ref and (ref.startswith("-") or not _REF_RE.match(ref)):
        raise RepoError(f"非法的分支 / 提交引用: {ref}")
    return ref


def _check_path(path: str) -> str:
    path = (path or "").strip().lstrip("/")
    if path.startswith("-") or ".." in Path(path).parts:
        raise RepoError(f"非法的路径: {path}")
    return path


def _truncate(text: str, limit: int = _MAX_PATCH_CHARS) -> str:
    if text and len(text) > limit:
        return text[:limit] + f"\n...（已截断，共 {len(text)} 字符）"
    return text or ""


class RepoProvider(ABC):
    """仓库访问接口。"""

    provider_name: str = ""
    supports_issues: bool = True

    @abstractmethod
    async def list_commits(
        self, branch: str = "", limit: int = 20, path: str = "", since: str = ""
    ) -> List[Dict[str, Any]]:
        """列出提交记录，可按分支、路径、起始时间过滤。"""
        ...

    @abstractmethod
    async def get_commit(self, sha: str) -> Dict[str, Any]:
        """查看单个提交的详情与 diff。"""
        ...

    async def list_issues(self, state: str = "open", limit: int = 20) -> List[Dict[str, Any]]:
        """列出 Issue。"""
        raise RepoError("该仓库类型不支持 Issue")

    async def get_issue(self, number: int) -> Dict[str, Any]:
        """查看单个 Issue。"""
        raise RepoError("该仓库类型不支持 Issue")

    async def list_pulls(self, state: str = "open", limit: int = 20) -> List[Dict[str, Any]]:
        """列出 PR / MR。"""
        raise RepoError("该仓库类型不支持 PR / MR")

    async def get_pull(self, number: int) -> Dict[str, Any]:
        """查看单个 PR / MR。"""
        raise RepoError("该仓库类型不支持 PR / MR")

    @abstractmethod
    async def read_file(self, path: str, ref: str = "") -> str:
        """读取指定文件的文本内容。"""
        ...

    @abstractmethod
    async def list_dir(self, path: str = "", ref: str = "") -> List[Dict[str, Any]]:
        """列出目录下的条目。"""
        ...

    @abstractmethod
    async def list_files(self, ref: str = "") -> List[Dict[str, Any]]:
        """递归列出全部文件 [{path, size}]（用于索引）。"""

    @abstractmethod
    async def search_code(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """在仓库中搜索代码。"""
        ...

    async def create_issue(self, title: str, body: str) -> Dict[str, Any]:
        """创建 Issue。属于写操作，只会生成待确认记录。"""
        raise RepoError("该仓库类型不支持创建 Issue")

    async def comment(self, number: int, body: str, kind: str = "issue") -> Dict[str, Any]:
        """在 Issue 或 PR 上发表评论。属于写操作，只会生成待确认记录。"""
        raise RepoError("该仓库类型不支持评论")


# ---- GitHub ----


class GitHubProvider(RepoProvider):
    """GitHub REST API 的访问实现。"""
    provider_name = "github"

    def __init__(self, repo: str, token: str = "", base_url: str = "",
                 transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        if repo.count("/") != 1:
            raise RepoError("GitHub 仓库格式应为 owner/repo")
        self.repo = repo.strip("/")
        self.base_url = (base_url or "https://api.github.com").rstrip("/")
        self.token = token
        self._transport = transport

    def _client(self) -> httpx.AsyncClient:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "Synapse-Repo-Assistant",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return httpx.AsyncClient(
            base_url=self.base_url, headers=headers, timeout=_HTTP_TIMEOUT, transport=self._transport
        )

    async def _get(self, endpoint: str, **params: Any) -> Any:
        async with self._client() as client:
            resp = await client.get(endpoint, params={k: v for k, v in params.items() if v not in (None, "")})
        return self._json(resp)

    async def _post(self, endpoint: str, payload: Dict[str, Any]) -> Any:
        async with self._client() as client:
            resp = await client.post(endpoint, json=payload)
        return self._json(resp)

    @staticmethod
    def _json(resp: httpx.Response) -> Any:
        if resp.status_code >= 400:
            try:
                message = resp.json().get("message", resp.text)
            except Exception:  # noqa: BLE001
                message = resp.text
            raise RepoError(f"GitHub API {resp.status_code}: {message[:300]}")
        return resp.json()

    async def list_commits(self, branch="", limit=20, path="", since=""):
        """列出提交记录，可按分支、路径、起始时间过滤。"""
        data = await self._get(
            f"/repos/{self.repo}/commits",
            sha=_check_ref(branch), per_page=min(limit, 100), path=_check_path(path), since=since,
        )
        return [
            {
                "sha": c["sha"],
                "title": (c["commit"]["message"] or "").splitlines()[0],
                "author": (c["commit"].get("author") or {}).get("name", ""),
                "date": (c["commit"].get("author") or {}).get("date", ""),
                "url": c.get("html_url", ""),
            }
            for c in data[:limit]
        ]

    async def get_commit(self, sha):
        """查看单个提交的详情与 diff。"""
        c = await self._get(f"/repos/{self.repo}/commits/{_check_ref(sha)}")
        return {
            "sha": c["sha"],
            "message": c["commit"]["message"],
            "author": (c["commit"].get("author") or {}).get("name", ""),
            "date": (c["commit"].get("author") or {}).get("date", ""),
            "url": c.get("html_url", ""),
            "stats": c.get("stats", {}),
            "files": [
                {
                    "path": f["filename"],
                    "status": f.get("status", ""),
                    "additions": f.get("additions", 0),
                    "deletions": f.get("deletions", 0),
                    "patch": _truncate(f.get("patch", ""), 1500),
                }
                for f in c.get("files", [])[:50]
            ],
        }

    async def list_issues(self, state="open", limit=20):
        """列出 Issue。"""
        data = await self._get(f"/repos/{self.repo}/issues", state=state, per_page=min(limit, 100))
        return [
            {
                "number": i["number"],
                "title": i["title"],
                "state": i["state"],
                "author": (i.get("user") or {}).get("login", ""),
                "labels": [lb["name"] for lb in i.get("labels", [])],
                "created_at": i.get("created_at", ""),
                "url": i.get("html_url", ""),
            }
            for i in data
            if "pull_request" not in i
        ][:limit]

    async def get_issue(self, number):
        """查看单个 Issue。"""
        issue = await self._get(f"/repos/{self.repo}/issues/{int(number)}")
        comments = await self._get(f"/repos/{self.repo}/issues/{int(number)}/comments", per_page=20)
        return {
            "number": issue["number"],
            "title": issue["title"],
            "state": issue["state"],
            "author": (issue.get("user") or {}).get("login", ""),
            "body": _truncate(issue.get("body") or "", 4000),
            "url": issue.get("html_url", ""),
            "comments": [
                {"author": (c.get("user") or {}).get("login", ""), "body": _truncate(c.get("body") or "", 1000)}
                for c in comments
            ],
        }

    async def list_pulls(self, state="open", limit=20):
        """列出 PR / MR。"""
        data = await self._get(f"/repos/{self.repo}/pulls", state=state, per_page=min(limit, 100))
        return [
            {
                "number": p["number"],
                "title": p["title"],
                "state": p["state"],
                "author": (p.get("user") or {}).get("login", ""),
                "head": (p.get("head") or {}).get("ref", ""),
                "base": (p.get("base") or {}).get("ref", ""),
                "created_at": p.get("created_at", ""),
                "url": p.get("html_url", ""),
            }
            for p in data[:limit]
        ]

    async def get_pull(self, number):
        """查看单个 PR / MR。"""
        pr = await self._get(f"/repos/{self.repo}/pulls/{int(number)}")
        files = await self._get(f"/repos/{self.repo}/pulls/{int(number)}/files", per_page=100)
        return {
            "number": pr["number"],
            "title": pr["title"],
            "state": pr["state"],
            "author": (pr.get("user") or {}).get("login", ""),
            "body": _truncate(pr.get("body") or "", 4000),
            "head": (pr.get("head") or {}).get("ref", ""),
            "base": (pr.get("base") or {}).get("ref", ""),
            "url": pr.get("html_url", ""),
            "files": [
                {
                    "path": f["filename"],
                    "status": f.get("status", ""),
                    "additions": f.get("additions", 0),
                    "deletions": f.get("deletions", 0),
                    "patch": _truncate(f.get("patch", ""), 1500),
                }
                for f in files[:50]
            ],
        }

    async def read_file(self, path, ref=""):
        """读取指定文件的文本内容。"""
        data = await self._get(
            f"/repos/{self.repo}/contents/{quote(_check_path(path))}", ref=_check_ref(ref)
        )
        if isinstance(data, list):
            raise RepoError(f"{path} 是目录，请使用列目录")
        content = data.get("content", "")
        if data.get("encoding") == "base64":
            return base64.b64decode(content).decode("utf-8", errors="replace")
        return content

    async def list_dir(self, path="", ref=""):
        """列出目录下的条目。"""
        data = await self._get(
            f"/repos/{self.repo}/contents/{quote(_check_path(path))}", ref=_check_ref(ref)
        )
        if not isinstance(data, list):
            data = [data]
        return [{"path": d["path"], "type": d["type"], "size": d.get("size", 0)} for d in data]

    async def _default_branch(self) -> str:
        info = await self._get(f"/repos/{self.repo}")
        return info.get("default_branch", "main")

    async def list_files(self, ref=""):
        """递归列出仓库中的文件，用于建立索引。"""
        ref = _check_ref(ref) or await self._default_branch()
        data = await self._get(f"/repos/{self.repo}/git/trees/{ref}", recursive=1)
        return [
            {"path": t["path"], "size": t.get("size", 0)}
            for t in data.get("tree", [])
            if t.get("type") == "blob"
        ]

    async def search_code(self, query, limit=20):
        """在仓库中搜索代码。"""
        data = await self._get("/search/code", q=f"{query} repo:{self.repo}", per_page=min(limit, 50))
        return [{"path": item["path"], "url": item.get("html_url", "")} for item in data.get("items", [])]

    async def create_issue(self, title, body):
        """创建 Issue。属于写操作，只会生成待确认记录。"""
        issue = await self._post(f"/repos/{self.repo}/issues", {"title": title, "body": body})
        return {"number": issue["number"], "url": issue.get("html_url", "")}

    async def comment(self, number, body, kind="issue"):
        # GitHub 的 PR 与 Issue 共用评论接口
        """在 Issue 或 PR 上发表评论。属于写操作，只会生成待确认记录。"""
        c = await self._post(f"/repos/{self.repo}/issues/{int(number)}/comments", {"body": body})
        return {"url": c.get("html_url", "")}


# ---- GitLab ----


class GitLabProvider(RepoProvider):
    """GitLab REST API 的访问实现。"""
    provider_name = "gitlab"

    def __init__(self, repo: str, token: str = "", base_url: str = "",
                 transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        if "/" not in repo.strip("/"):
            raise RepoError("GitLab 仓库格式应为 group/project")
        self.repo = repo.strip("/")
        self.project = quote(self.repo, safe="")
        self.base_url = (base_url or "https://gitlab.com/api/v4").rstrip("/")
        self.token = token
        self._transport = transport

    def _client(self) -> httpx.AsyncClient:
        headers = {"User-Agent": "Synapse-Repo-Assistant"}
        if self.token:
            headers["PRIVATE-TOKEN"] = self.token
        return httpx.AsyncClient(
            base_url=self.base_url, headers=headers, timeout=_HTTP_TIMEOUT, transport=self._transport
        )

    async def _request(self, method: str, path: str, params=None, payload=None, raw: bool = False) -> Any:
        async with self._client() as client:
            resp = await client.request(
                method, path,
                params={k: v for k, v in (params or {}).items() if v not in (None, "")},
                json=payload,
            )
        if resp.status_code >= 400:
            try:
                message = str(resp.json().get("message", resp.text))
            except Exception:  # noqa: BLE001
                message = resp.text
            raise RepoError(f"GitLab API {resp.status_code}: {message[:300]}")
        return resp.text if raw else resp.json()

    @staticmethod
    def _state(state: str) -> str:
        return {"open": "opened", "closed": "closed", "all": "all", "merged": "merged"}.get(state, state)

    async def list_commits(self, branch="", limit=20, path="", since=""):
        """列出提交记录，可按分支、路径、起始时间过滤。"""
        data = await self._request("GET", f"/projects/{self.project}/repository/commits", params={
            "ref_name": _check_ref(branch), "per_page": min(limit, 100),
            "path": _check_path(path), "since": since,
        })
        return [
            {
                "sha": c["id"],
                "title": c.get("title", ""),
                "author": c.get("author_name", ""),
                "date": c.get("committed_date") or c.get("created_at", ""),
                "url": c.get("web_url", ""),
            }
            for c in data[:limit]
        ]

    async def get_commit(self, sha):
        """查看单个提交的详情与 diff。"""
        sha = _check_ref(sha)
        c = await self._request("GET", f"/projects/{self.project}/repository/commits/{sha}")
        diffs = await self._request("GET", f"/projects/{self.project}/repository/commits/{sha}/diff")
        return {
            "sha": c["id"],
            "message": c.get("message", ""),
            "author": c.get("author_name", ""),
            "date": c.get("committed_date", ""),
            "url": c.get("web_url", ""),
            "stats": c.get("stats", {}),
            "files": [
                {"path": d.get("new_path", ""), "status": "", "patch": _truncate(d.get("diff", ""), 1500)}
                for d in diffs[:50]
            ],
        }

    async def list_issues(self, state="open", limit=20):
        """列出 Issue。"""
        data = await self._request("GET", f"/projects/{self.project}/issues", params={
            "state": self._state(state), "per_page": min(limit, 100),
        })
        return [
            {
                "number": i["iid"],
                "title": i["title"],
                "state": i["state"],
                "author": (i.get("author") or {}).get("username", ""),
                "labels": i.get("labels", []),
                "created_at": i.get("created_at", ""),
                "url": i.get("web_url", ""),
            }
            for i in data[:limit]
        ]

    async def get_issue(self, number):
        """查看单个 Issue。"""
        issue = await self._request("GET", f"/projects/{self.project}/issues/{int(number)}")
        notes = await self._request(
            "GET", f"/projects/{self.project}/issues/{int(number)}/notes",
            params={"per_page": 20, "sort": "asc"},
        )
        return {
            "number": issue["iid"],
            "title": issue["title"],
            "state": issue["state"],
            "author": (issue.get("author") or {}).get("username", ""),
            "body": _truncate(issue.get("description") or "", 4000),
            "url": issue.get("web_url", ""),
            "comments": [
                {"author": (n.get("author") or {}).get("username", ""), "body": _truncate(n.get("body") or "", 1000)}
                for n in notes if not n.get("system")
            ],
        }

    async def list_pulls(self, state="open", limit=20):
        """列出 PR / MR。"""
        data = await self._request("GET", f"/projects/{self.project}/merge_requests", params={
            "state": self._state(state), "per_page": min(limit, 100),
        })
        return [
            {
                "number": m["iid"],
                "title": m["title"],
                "state": m["state"],
                "author": (m.get("author") or {}).get("username", ""),
                "head": m.get("source_branch", ""),
                "base": m.get("target_branch", ""),
                "created_at": m.get("created_at", ""),
                "url": m.get("web_url", ""),
            }
            for m in data[:limit]
        ]

    async def get_pull(self, number):
        """查看单个 PR / MR。"""
        mr = await self._request("GET", f"/projects/{self.project}/merge_requests/{int(number)}")
        try:
            diffs = await self._request(
                "GET", f"/projects/{self.project}/merge_requests/{int(number)}/diffs",
                params={"per_page": 100},
            )
        except RepoError:
            changes = await self._request(
                "GET", f"/projects/{self.project}/merge_requests/{int(number)}/changes"
            )
            diffs = changes.get("changes", [])
        return {
            "number": mr["iid"],
            "title": mr["title"],
            "state": mr["state"],
            "author": (mr.get("author") or {}).get("username", ""),
            "body": _truncate(mr.get("description") or "", 4000),
            "head": mr.get("source_branch", ""),
            "base": mr.get("target_branch", ""),
            "url": mr.get("web_url", ""),
            "files": [
                {"path": d.get("new_path", ""), "status": "", "patch": _truncate(d.get("diff", ""), 1500)}
                for d in diffs[:50]
            ],
        }

    async def read_file(self, path, ref=""):
        """读取指定文件的文本内容。"""
        path = quote(_check_path(path), safe="")
        return await self._request(
            "GET", f"/projects/{self.project}/repository/files/{path}/raw",
            params={"ref": _check_ref(ref) or "HEAD"}, raw=True,
        )

    async def list_dir(self, path="", ref=""):
        """列出目录下的条目。"""
        data = await self._request("GET", f"/projects/{self.project}/repository/tree", params={
            "path": _check_path(path), "ref": _check_ref(ref), "per_page": 100,
        })
        return [{"path": d["path"], "type": "dir" if d["type"] == "tree" else "file", "size": 0} for d in data]

    async def list_files(self, ref=""):
        """递归列出仓库中的文件，用于建立索引。"""
        files: List[Dict[str, Any]] = []
        for page in range(1, 21):
            data = await self._request("GET", f"/projects/{self.project}/repository/tree", params={
                "ref": _check_ref(ref), "recursive": "true", "per_page": 100, "page": page,
            })
            files.extend({"path": d["path"], "size": 0} for d in data if d.get("type") == "blob")
            if len(data) < 100:
                break
        return files

    async def search_code(self, query, limit=20):
        """在仓库中搜索代码。"""
        data = await self._request("GET", f"/projects/{self.project}/search", params={
            "scope": "blobs", "search": query, "per_page": min(limit, 50),
        })
        return [
            {"path": d.get("path") or d.get("filename", ""), "line": d.get("startline"), "snippet": _truncate(d.get("data", ""), 300)}
            for d in data
        ]

    async def create_issue(self, title, body):
        """创建 Issue。属于写操作，只会生成待确认记录。"""
        issue = await self._request(
            "POST", f"/projects/{self.project}/issues", payload={"title": title, "description": body}
        )
        return {"number": issue["iid"], "url": issue.get("web_url", "")}

    async def comment(self, number, body, kind="issue"):
        """在 Issue 或 PR 上发表评论。属于写操作，只会生成待确认记录。"""
        target = "merge_requests" if kind in ("mr", "pr", "pull") else "issues"
        note = await self._request(
            "POST", f"/projects/{self.project}/{target}/{int(number)}/notes", payload={"body": body}
        )
        return {"id": note.get("id")}


# ---- 本地 Git ----


class LocalGitProvider(RepoProvider):
    """本地仓库：repo 为本地路径，或 clone 地址（自动 clone 到 data_dir/repos）。"""

    provider_name = "local"
    supports_issues = False
    #: clone 仓库的自动 fetch 间隔（秒）
    _FETCH_INTERVAL = 600
    _last_fetch: Dict[str, float] = {}

    def __init__(self, repo: str, clone_dir: Optional[Path] = None) -> None:
        self.source = repo.strip()
        self.is_remote = bool(re.match(r"^(https?://|git@|ssh://)", self.source))
        if self.is_remote:
            if clone_dir is None:
                raise RepoError("远程仓库需要指定 clone 目录")
            self.path = clone_dir
        else:
            self.path = Path(self.source)

    async def _git(self, *args: str, check: bool = True) -> str:
        cmd = ["git", "-c", "core.quotepath=false", "--git-dir", str(self._git_dir()), *args]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_GIT_TIMEOUT)
        except asyncio.TimeoutError as exc:
            proc.kill()
            raise RepoError("git 命令超时") from exc
        if check and proc.returncode != 0:
            raise RepoError(f"git 执行失败: {stderr.decode('utf-8', errors='replace').strip()[:300]}")
        return stdout.decode("utf-8", errors="replace")

    def _git_dir(self) -> Path:
        dot_git = self.path / ".git"
        return dot_git if dot_git.exists() else self.path

    async def ensure_ready(self) -> None:
        """远程仓库首次使用时 clone（bare），之后定期 fetch。"""
        if not self.is_remote:
            if not self.path.exists():
                raise RepoError(f"本地仓库路径不存在: {self.path}")
            return
        key = str(self.path)
        if not (self.path / "HEAD").exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            proc = await asyncio.create_subprocess_exec(
                "git", "clone", "--bare", "--", self.source, str(self.path),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)
            if proc.returncode != 0:
                raise RepoError(f"clone 失败: {stderr.decode('utf-8', errors='replace')[:300]}")
            self._last_fetch[key] = time.monotonic()
        elif time.monotonic() - self._last_fetch.get(key, 0) > self._FETCH_INTERVAL:
            await self._git("fetch", "--prune", "origin", "+refs/heads/*:refs/heads/*", check=False)
            self._last_fetch[key] = time.monotonic()

    async def list_commits(self, branch="", limit=20, path="", since=""):
        """列出提交记录，可按分支、路径、起始时间过滤。"""
        await self.ensure_ready()
        args = [
            "log", f"--max-count={max(1, min(limit, 200))}", "--date=iso-strict",
            "--pretty=format:%H%x1f%an%x1f%ad%x1f%s",
        ]
        if since:
            args.append(f"--since={since}")
        branch = _check_ref(branch)
        if branch:
            args.append(branch)
        if path:
            args += ["--", _check_path(path)]
        out = await self._git(*args)
        commits = []
        for line in out.splitlines():
            parts = line.split("\x1f")
            if len(parts) == 4:
                commits.append({"sha": parts[0], "author": parts[1], "date": parts[2], "title": parts[3], "url": ""})
        return commits

    async def get_commit(self, sha):
        """查看单个提交的详情与 diff。"""
        await self.ensure_ready()
        sha = _check_ref(sha)
        meta = await self._git("show", "-s", "--date=iso-strict", "--format=%H%x1f%an%x1f%ad%x1f%B", sha)
        parts = meta.split("\x1f", 3)
        stat = await self._git("show", "--stat", "--format=", sha)
        patch = await self._git("show", "--format=", "--patch", sha)
        return {
            "sha": parts[0].strip() if parts else sha,
            "author": parts[1] if len(parts) > 1 else "",
            "date": parts[2] if len(parts) > 2 else "",
            "message": parts[3].strip() if len(parts) > 3 else "",
            "url": "",
            "stats": {"summary": stat.strip().splitlines()[-1] if stat.strip() else ""},
            "files": [{"path": "(全部)", "status": "", "patch": _truncate(patch, _MAX_PATCH_CHARS)}],
        }

    async def read_file(self, path, ref=""):
        """读取指定文件的文本内容。"""
        await self.ensure_ready()
        return await self._git("show", f"{_check_ref(ref) or 'HEAD'}:{_check_path(path)}")

    async def list_dir(self, path="", ref=""):
        """列出目录下的条目。"""
        await self.ensure_ready()
        path = _check_path(path)
        target = f"{path.rstrip('/')}/" if path else ""
        args = ["ls-tree", "--long", _check_ref(ref) or "HEAD"]
        if target:
            args += ["--", target]
        out = await self._git(*args)
        items = []
        for line in out.splitlines():
            meta, _, name = line.partition("\t")
            fields = meta.split()
            if len(fields) >= 4:
                items.append({
                    "path": name,
                    "type": "dir" if fields[1] == "tree" else "file",
                    "size": int(fields[3]) if fields[3].isdigit() else 0,
                })
        return items

    async def list_files(self, ref=""):
        """递归列出仓库中的文件，用于建立索引。"""
        await self.ensure_ready()
        out = await self._git("ls-tree", "-r", "--long", _check_ref(ref) or "HEAD")
        files = []
        for line in out.splitlines():
            meta, _, name = line.partition("\t")
            fields = meta.split()
            if len(fields) >= 4 and fields[1] == "blob":
                files.append({"path": name, "size": int(fields[3]) if fields[3].isdigit() else 0})
        return files

    async def search_code(self, query, limit=20):
        """在仓库中搜索代码。"""
        await self.ensure_ready()
        out = await self._git(
            "grep", "-n", "-I", "-i", "--max-count=3", "-e", query, "HEAD", check=False
        )
        results = []
        for line in out.splitlines()[:limit]:
            # 格式: HEAD:path:line:content
            parts = line.split(":", 3)
            if len(parts) == 4:
                results.append({"path": parts[1], "line": parts[2], "snippet": parts[3].strip()[:300]})
        return results
