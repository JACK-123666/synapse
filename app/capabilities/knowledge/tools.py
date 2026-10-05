"""知识库能力的 LangChain 工具。"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from langchain_core.tools import tool

from app.capabilities.knowledge.service import (
    KnowledgeError,
    current_principal,
    get_knowledge_service,
)


@tool("knowledge_search", response_format="content_and_artifact")
async def knowledge_search_tool(
    query: str, knowledge_base: str = "", top_k: int = 5
) -> Tuple[str, List[Dict[str, Any]]]:
    """在知识库中检索与问题相关的内容片段，回答时请注明来源（知识库 / 文件名）。

    Args:
        query: 检索问题或关键词
        knowledge_base: 知识库名称；留空表示检索当前用户可访问的全部知识库
        top_k: 返回片段数，默认 5
    """
    try:
        results = await get_knowledge_service().search_for_current_user(
            query, kb_name=knowledge_base or None, top_k=max(1, min(int(top_k or 5), 20))
        )
    except KnowledgeError as exc:
        return str(exc), []
    if not results:
        return "知识库中没有找到相关内容。", []
    lines = [
        f"[{i}] (来源: {r['kb']} / {r['filename']}, 相似度 {r['score']:.2f})\n{r['text']}"
        for i, r in enumerate(results, 1)
    ]
    artifact = [{k: v for k, v in r.items() if k != "text"} | {"text": r["text"][:200]} for r in results]
    return "\n\n".join(lines), artifact


@tool("list_knowledge_bases")
async def list_knowledge_bases_tool() -> str:
    """列出当前用户可访问的知识库（名称、可见性、文档数）。"""
    kbs = await get_knowledge_service().list_kbs(current_principal())
    if not kbs:
        return "当前没有可访问的知识库。"
    return "\n".join(
        f"- {kb['name']}（{'共享' if kb['visibility'] == 'shared' else '私有'}，"
        f"{kb.get('document_count', 0)} 篇文档）{kb['description'] or ''}"
        for kb in kbs
    )
