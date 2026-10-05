"""阶段 3：知识库（RAG）、记忆检索、网页抓取。"""

from __future__ import annotations

import asyncio
import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.capabilities.knowledge.service import KnowledgeError, Principal, get_knowledge_service
from app.core.context import RequestContext, request_context

ALICE = Principal("alice-id")
BOB = Principal("bob-id")


async def _make_users():
    from app.core.db import session_scope
    from app.models import User

    async with session_scope() as session:
        session.add(User(id="alice-id", username="alice", role="user"))
        session.add(User(id="bob-id", username="bob", role="user"))


# ---- 知识库 ----


async def test_kb_ingest_search_and_permissions(db, chroma, fake_embeddings):
    await _make_users()
    service = get_knowledge_service()
    kb = await service.create_kb(ALICE, "产品手册")
    text = "报销流程：提交发票后由财务审核，三个工作日内到账。\n\n" + "其他无关内容。" * 50
    doc = await service.ingest_text(ALICE, kb["id"], "报销制度.md", text)
    assert doc["status"] == "ready" and doc["chunk_count"] >= 1

    results = await service.search(ALICE, "报销流程：提交发票后由财务审核，三个工作日内到账。")
    assert results and results[0]["kb"] == "产品手册" and results[0]["filename"] == "报销制度.md"

    # 私有知识库：其他用户不可见，也不能写
    assert await service.search(BOB, "报销流程") == []
    with pytest.raises(KnowledgeError):
        await service.ingest_text(BOB, kb["id"], "x", "y")

    # 共享后其他用户可检索，但仍不能写
    await service.update_kb(ALICE, kb["id"], visibility="shared")
    assert await service.search(BOB, "报销流程")
    with pytest.raises(PermissionError):
        await service.ingest_text(BOB, kb["id"], "x", "y")

    # 删除文档后检索不到
    await service.delete_document(ALICE, doc["id"])
    assert await service.search(ALICE, "报销流程") == []


async def test_kb_reindex_and_duplicate_name(db, chroma, fake_embeddings):
    await _make_users()
    service = get_knowledge_service()
    kb = await service.create_kb(ALICE, "kb1")
    with pytest.raises(KnowledgeError):
        await service.create_kb(ALICE, "kb1")
    await service.ingest_text(ALICE, kb["id"], "a.txt", "第一段内容")
    result = await service.reindex_kb(ALICE, kb["id"])
    assert result == {"reindexed": 1, "failed": 0}
    assert await service.search(ALICE, "第一段内容")


def test_loaders_docx_and_unsupported():
    import docx

    from app.capabilities.knowledge.loaders import UnsupportedFileType, extract_text

    buffer = io.BytesIO()
    document = docx.Document()
    document.add_paragraph("合同条款第一条")
    document.save(buffer)
    assert "合同条款第一条" in extract_text("合同.docx", buffer.getvalue())
    assert extract_text("a.md", "# 标题".encode("utf-8")) == "# 标题"
    assert "中文" in extract_text("gbk.txt", "中文内容".encode("gb18030"))
    with pytest.raises(UnsupportedFileType):
        extract_text("a.exe", b"MZ")


async def test_knowledge_api_upload_and_search(db, chroma, fake_embeddings):
    from app.api.knowledge import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    kb = client.post("/knowledge-bases", json={"name": "团队文档"}).json()
    files = {"file": ("faq.txt", "问：如何申请服务器？答：在运维平台提交工单。".encode("utf-8"), "text/plain")}
    uploaded = client.post(f"/knowledge-bases/{kb['id']}/documents", files=files)
    assert uploaded.status_code == 200, uploaded.text
    hits = client.post("/knowledge-bases/search", json={"query": "问：如何申请服务器？答：在运维平台提交工单。"}).json()
    assert hits[0]["filename"] == "faq.txt"
    assert client.get(f"/knowledge-bases/{kb['id']}/documents").json()[0]["status"] == "ready"
    bad = client.post(f"/knowledge-bases/{kb['id']}/documents", files={"file": ("a.exe", b"MZ", "application/octet-stream")})
    assert bad.status_code == 400


async def test_knowledge_search_tool_uses_request_user(db, chroma, fake_embeddings):
    from app.capabilities.knowledge.tools import knowledge_search_tool

    await _make_users()
    service = get_knowledge_service()
    kb = await service.create_kb(ALICE, "私人笔记")
    await service.ingest_text(ALICE, kb["id"], "note.txt", "我的 WiFi 密码放在抽屉里")

    with request_context(RequestContext(user_id="alice-id", role="user")):
        out = await knowledge_search_tool.ainvoke({"query": "我的 WiFi 密码放在抽屉里"})
        assert "note.txt" in out
    with request_context(RequestContext(user_id="bob-id", role="user")):
        out = await knowledge_search_tool.ainvoke({"query": "我的 WiFi 密码放在抽屉里"})
        assert "没有找到" in out


# ---- 网页 ----


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://localhost:8000/",
        "http://10.0.0.5/",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/",
        "ftp://example.com/file",
        "http://foo.internal/",
    ],
)
async def test_ssrf_blocked(url):
    from app.capabilities.web.fetch import FetchError, validate_public_url

    with pytest.raises(FetchError):
        await validate_public_url(url, allow_private=False)


async def test_public_ip_allowed():
    from app.capabilities.web.fetch import validate_public_url

    await validate_public_url("https://8.8.8.8/", allow_private=False)


def test_html_to_text_extracts_title_and_body():
    from app.capabilities.web.fetch import html_to_text

    html = (
        "<html><head><title>测试页面</title><script>var x=1;</script></head>"
        "<body><nav>导航</nav><article><h1>向量数据库简介</h1>"
        "<p>向量数据库用于存储和检索高维向量，常用于语义搜索与 RAG 场景。</p>"
        "<p>典型产品包括 Milvus、Chroma 等。</p></article></body></html>"
    )
    title, text = html_to_text(html)
    # trafilatura 会优先使用正文主标题，退化路径使用 <title>
    assert title in ("测试页面", "向量数据库简介")
    assert "向量数据库用于存储和检索高维向量" in text
    assert "var x" not in text


async def test_fetch_url_tool_reports_blocked():
    from app.capabilities.web.tools import fetch_url_tool

    out = await fetch_url_tool.ainvoke({"url": "http://127.0.0.1:6379/"})
    assert "抓取失败" in out


# ---- 记忆 ----


async def test_compress_keeps_messages_appended_during_compression(fake_redis, chroma, fake_embeddings):
    from app.memory.compress import MemoryCompressor
    from app.memory.recent import get_short_term_memory

    short = get_short_term_memory()
    for i in range(4):
        await short.append("s-race", "user", f"问题{i}")
        await short.append("s-race", "assistant", f"回答{i}")

    compressor = MemoryCompressor()

    async def slow_summary(dialogue_text: str) -> str:
        # 压缩进行中，用户又发来了新消息
        await short.append("s-race", "user", "压缩期间的新问题")
        return "摘要"

    compressor._generate_summary = slow_summary  # type: ignore[method-assign]
    assert await compressor.compress("s-race") == "摘要"
    remaining = await short.get_messages("s-race")
    assert [m["content"] for m in remaining] == ["压缩期间的新问题"]


async def test_profile_concurrent_increments(fake_redis):
    from app.memory.profile import get_user_profile_manager

    manager = get_user_profile_manager()
    await asyncio.gather(*(manager.increment_interaction("u-concurrent") for _ in range(20)))
    profile = await manager.get_profile("u-concurrent")
    assert profile["interaction_count"] == 20

    await manager.add_frequent_terms("u-concurrent", ["向量", "RAG"])
    await manager.add_frequent_terms("u-concurrent", ["RAG", "Agent"])
    profile = await manager.get_profile("u-concurrent")
    assert profile["frequent_terms"] == ["向量", "RAG", "Agent"]


async def test_memory_tools_and_api(db, fake_redis, chroma, fake_embeddings):
    from app.api.memory import router
    from app.capabilities.memory import memory_recent_tool, memory_search_tool
    from app.memory.archive import get_long_term_memory
    from app.memory.recent import get_short_term_memory

    archive = get_long_term_memory()
    archive._collection = None
    record_id = await archive.store_summary("s1", "讨论了 Milvus 与 Chroma 的选型", user_id="u1")
    await archive.store_summary("s2", "别人的记忆", user_id="u2")
    await get_short_term_memory().append("s1", "user", "上一句话")

    with request_context(RequestContext(session_id="s1", memory_user_id="u1")):
        found = await memory_search_tool.ainvoke({"query": "讨论了 Milvus 与 Chroma 的选型"})
        assert "Milvus" in found and "别人的记忆" not in found
        assert "上一句话" in await memory_recent_tool.ainvoke({})

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    listed = client.get("/memories", params={"user_id": "u1"}).json()
    assert [m["id"] for m in listed] == [record_id]
    assert client.delete(f"/memories/{record_id}").status_code == 200
    assert client.get("/memories", params={"user_id": "u1"}).json() == []
