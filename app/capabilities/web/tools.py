"""网页能力的 LangChain 工具。"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from langchain_core.tools import tool

from app.capabilities.web.fetch import FetchError, fetch_page
from app.tools.registry import call_tool


@tool("web_search", response_format="content_and_artifact")
async def web_search_tool(query: str, max_results: int = 5) -> Tuple[str, List[Dict[str, str]]]:
    """联网搜索互联网获取最新/实时信息。当本地知识库没有答案、或用户需要最新资料时调用。返回标题、摘要与链接。

    Args:
        query: 要搜索的关键词或完整问题
        max_results: 返回结果条数，默认 5（1~10）
    """
    collector: List[Dict[str, str]] = []
    max_results = max(1, min(int(max_results or 5), 10))
    text = await call_tool(
        "web_search", {"query": query, "max_results": max_results}, collector=collector
    )
    return text, collector


@tool("fetch_url", response_format="content_and_artifact")
async def fetch_url_tool(url: str, max_chars: int = 8000) -> Tuple[str, Dict[str, Any]]:
    """抓取指定网页（http/https）并提取正文（Markdown 格式）。用于阅读链接内容、总结网页、获取页面中的具体信息。

    Args:
        url: 网页地址
        max_chars: 返回正文的最大字符数，默认 8000
    """
    try:
        page = await fetch_page(url, max_chars=max(500, min(int(max_chars or 8000), 30000)))
    except FetchError as exc:
        return f"抓取失败: {exc}", {"url": url, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return f"抓取失败: {exc}", {"url": url, "error": str(exc)}
    header = f"标题: {page.title}\n链接: {page.final_url}\n"
    if page.truncated:
        header += "（内容较长，已截断）\n"
    artifact = {"url": page.final_url, "title": page.title, "truncated": page.truncated}
    return f"{header}\n{page.text}", artifact


@tool("save_url_to_knowledge")
async def save_url_to_knowledge_tool(url: str, knowledge_base: str) -> str:
    """抓取网页正文并存入指定名称的知识库（不存在时自动创建私有知识库），之后可在知识问答中引用。

    Args:
        url: 网页地址
        knowledge_base: 知识库名称
    """
    from app.capabilities.knowledge.service import KnowledgeError, get_knowledge_service

    service = get_knowledge_service()
    try:
        kb = await service.get_or_create_kb_for_current_user(knowledge_base)
        doc = await service.ingest_url_for_current_user(kb.id, url)
    except (KnowledgeError, FetchError, PermissionError) as exc:
        return f"保存失败: {exc}"
    return (
        f"已保存到知识库「{kb.name}」：{doc['filename']}（{doc['chunk_count']} 个片段）。"
    )
