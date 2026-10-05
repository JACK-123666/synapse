"""阶段 4：代码仓库助手。"""

from __future__ import annotations

import subprocess
from pathlib import Path

import httpx
import pytest

from app.capabilities.knowledge.service import Principal
from app.capabilities.repo.providers import (
    GitHubProvider,
    GitLabProvider,
    LocalGitProvider,
    RepoError,
    _check_ref,
)
from app.capabilities.repo.service import build_changelog, get_repo_service
from app.core.context import LOCAL_ADMIN_ID, RequestContext, request_context

ADMIN = Principal(LOCAL_ADMIN_ID, is_admin=True)


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def local_repo(tmp_path) -> Path:
    repo = tmp_path / "demo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "dev@example.com")
    _git(repo, "config", "user.name", "Dev")
    (repo / "app").mkdir()
    (repo / "app" / "main.py").write_text("def handler():\n    return 'hello synapse'\n", encoding="utf-8")
    (repo / "README.md").write_text("# Demo\n部署说明：docker compose up\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "feat(api): 新增 handler 接口")
    (repo / "app" / "main.py").write_text("def handler():\n    return 'hello synapse v2'\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "fix: 修正返回值")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "整理代码")
    return repo


# ---- 本地 Git ----


async def test_local_git_provider(local_repo):
    provider = LocalGitProvider(str(local_repo))
    commits = await provider.list_commits(limit=10)
    assert [c["title"] for c in commits] == ["整理代码", "fix: 修正返回值", "feat(api): 新增 handler 接口"]

    detail = await provider.get_commit(commits[1]["sha"][:7])
    assert "hello synapse v2" in detail["files"][0]["patch"]

    assert "hello synapse v2" in await provider.read_file("app/main.py")
    assert {i["path"] for i in await provider.list_dir()} == {"app", "README.md"}
    assert {f["path"] for f in await provider.list_files()} == {"app/main.py", "README.md"}

    hits = await provider.search_code("hello synapse")
    assert hits and hits[0]["path"] == "app/main.py"

    file_commits = await provider.list_commits(path="README.md")
    assert len(file_commits) == 1


def test_ref_and_path_injection_rejected():
    from app.capabilities.repo.providers import _check_path

    with pytest.raises(RepoError):
        _check_ref("--output=/tmp/x")
    with pytest.raises(RepoError):
        _check_path("../etc/passwd")
    assert _check_ref("feature/x-1") == "feature/x-1"


def test_build_changelog_groups_conventional_commits():
    text = build_changelog([
        {"sha": "a" * 40, "title": "feat(api): 新增接口"},
        {"sha": "b" * 40, "title": "fix: 修复空指针"},
        {"sha": "c" * 40, "title": "随手改改"},
    ], title="v1.1")
    assert "### 新功能" in text and "**api**: 新增接口 (aaaaaaa)" in text
    assert "### 问题修复" in text and "修复空指针" in text
    assert "### 其他" in text and "随手改改" in text


# ---- GitHub / GitLab（模拟 API） ----


async def test_github_provider_with_mock_api():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer tok"
        path = request.url.path
        if path == "/repos/o/r/commits":
            return httpx.Response(200, json=[{
                "sha": "abc1234def", "html_url": "u",
                "commit": {"message": "feat: x\n\nbody", "author": {"name": "A", "date": "2026-10-01"}},
            }])
        if path == "/repos/o/r/issues":
            return httpx.Response(200, json=[
                {"number": 1, "title": "Bug", "state": "open", "user": {"login": "u1"}, "labels": []},
                {"number": 2, "title": "PR", "state": "open", "user": {"login": "u2"}, "labels": [], "pull_request": {}},
            ])
        if path == "/repos/o/r/pulls/5":
            return httpx.Response(200, json={"number": 5, "title": "改进", "state": "open", "user": {"login": "u"},
                                             "body": "说明", "head": {"ref": "dev"}, "base": {"ref": "main"}})
        if path == "/repos/o/r/pulls/5/files":
            return httpx.Response(200, json=[{"filename": "a.py", "status": "modified", "additions": 1, "deletions": 0, "patch": "+x"}])
        return httpx.Response(404, json={"message": "Not Found"})

    provider = GitHubProvider("o/r", token="tok", transport=httpx.MockTransport(handler))
    commits = await provider.list_commits(limit=5)
    assert commits[0]["title"] == "feat: x"
    issues = await provider.list_issues()
    assert [i["number"] for i in issues] == [1]
    pr = await provider.get_pull(5)
    assert pr["files"][0]["path"] == "a.py" and pr["base"] == "main"
    with pytest.raises(RepoError):
        await provider.get_issue(99)


async def test_gitlab_provider_encodes_project_and_maps_state():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.raw_path.decode()
        seen["params"] = dict(request.url.params)
        assert request.headers["PRIVATE-TOKEN"] == "gl"
        return httpx.Response(200, json=[{"iid": 7, "title": "T", "state": "opened", "author": {"username": "x"}, "labels": ["bug"]}])

    provider = GitLabProvider("group/sub/proj", token="gl", transport=httpx.MockTransport(handler))
    issues = await provider.list_issues(state="open")
    assert issues[0]["number"] == 7
    assert "/projects/group%2Fsub%2Fproj/issues" in seen["path"]
    assert seen["params"]["state"] == "opened"


# ---- 服务：连接、写操作确认、索引 ----


async def test_connection_token_encrypted_and_permissions(db):
    from app.core.db import session_scope
    from app.models import RepoConnection, User

    async with session_scope() as session:
        session.add(User(id="dev-id", username="dev", role="user"))

    service = get_repo_service()
    conn = await service.create_connection(
        ADMIN, name="synapse", provider="github", repo="o/r", token="ghp_secret"
    )
    assert conn["has_token"] is True
    async with session_scope() as session:
        row = await session.get(RepoConnection, conn["id"])
        assert "ghp_secret" not in row.token_encrypted
        assert service.provider_for(row).token == "ghp_secret"

    dev = Principal("dev-id")
    with pytest.raises(PermissionError):
        await service.create_connection(dev, name="local", provider="local", repo="C:/secret")
    with pytest.raises(PermissionError):
        await service.create_connection(dev, name="w", provider="github", repo="o/r", allow_write=True)
    # 其他用户看不到管理员的连接
    with pytest.raises(RepoError):
        await service.get_connection(dev, "synapse")


async def test_write_action_requires_switch_and_confirmation(db, monkeypatch):
    from app.capabilities.repo.tools import repo_create_issue_tool
    from app.config import get_settings

    service = get_repo_service()
    await service.create_connection(ADMIN, name="r1", provider="github", repo="o/r", allow_write=True)

    with request_context(RequestContext()):
        out = await repo_create_issue_tool.ainvoke({"title": "新 Bug", "body": "详情", "repo": "r1"})
    assert "未开启写操作" in out

    monkeypatch.setattr(get_settings(), "repo_write_enabled", True)
    with request_context(RequestContext()):
        out = await repo_create_issue_tool.ainvoke({"title": "新 Bug", "body": "详情", "repo": "r1"})
    assert "待确认操作" in out
    pending = await service.list_actions(ADMIN, "pending")
    assert len(pending) == 1

    created = {}

    class FakeProvider:
        async def create_issue(self, title, body):
            created["title"] = title
            return {"number": 42, "url": "https://github.com/o/r/issues/42"}

    monkeypatch.setattr(service, "provider_for", lambda conn: FakeProvider())
    result = await service.confirm_action(ADMIN, pending[0]["id"])
    assert result["status"] == "done" and created["title"] == "新 Bug"
    with pytest.raises(RepoError):
        await service.confirm_action(ADMIN, pending[0]["id"])


async def test_index_local_repo_into_knowledge(db, chroma, fake_embeddings, local_repo):
    from app.capabilities.knowledge.service import get_knowledge_service
    from app.capabilities.repo.tools import repo_changelog_tool, repo_commits_tool

    service = get_repo_service()
    await service.create_connection(ADMIN, name="demo", provider="local", repo=str(local_repo))
    result = await service.index_to_knowledge(ADMIN, "demo", "demo-code")
    assert result["files"] == 2 and result["failed"] == 0

    # 假 embedding 按文本哈希生成，用入库时的原文检索才能精确命中
    hits = await get_knowledge_service().search(ADMIN, "文件: app/main.py\n\ndef handler():\n    return 'hello synapse v2'")
    assert hits and hits[0]["filename"] == "demo/app/main.py"

    with request_context(RequestContext()):
        assert "修正返回值" in await repo_commits_tool.ainvoke({})
        changelog = await repo_changelog_tool.ainvoke({"repo": "demo"})
        assert "### 新功能" in changelog and "新增 handler 接口" in changelog


async def test_repo_api_routes(db, local_repo):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.repos import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    created = client.post("/repos", json={"name": "demo", "provider": "local", "repo": str(local_repo)})
    assert created.status_code == 200, created.text
    conn_id = created.json()["id"]
    assert client.post(f"/repos/{conn_id}/test").json()["latest_commit"]["title"] == "整理代码"
    assert "新功能" in client.get(f"/repos/{conn_id}/changelog").json()["markdown"]
    assert client.get("/repos/actions").json() == []
    assert client.delete(f"/repos/{conn_id}").status_code == 200
