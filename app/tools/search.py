"""联网搜索工具 —— 基于 DuckDuckGo HTML 接口，无 API Key 依赖。"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

import httpx

logger = logging.getLogger(__name__)

# DuckDuckGo HTML 搜索端点（无 API Key，无需认证）
_DDG_URL = "https://html.duckduckgo.com/html/"


async def web_search(
    query: str,
    max_results: int = 5,
    timeout: float = 10.0,
) -> List[Dict[str, str]]:
    """联网搜索，返回 title / snippet / url 列表。

    失败时返回空列表，不抛异常 —— 由调用方按 best‑effort 降级。
    """
    if not query.strip():
        return []

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                _DDG_URL,
                data={"q": query, "b": ""},
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/125.0.0.0 Safari/537.36"
                    ),
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                },
            )
            resp.raise_for_status()
        results = _parse_ddg_html(resp.text, max_results)
        return results
    except Exception as exc:
        logger.warning("联网搜索失败: %s", exc)
        return []


def _parse_ddg_html(html: str, max_results: int) -> List[Dict[str, str]]:
    """从 DuckDuckGo HTML 结果页抽取标题 / 摘要 / 链接。"""
    results: List[Dict[str, str]] = []

    # 每个结果块由 class="result" 包裹
    blocks = re.split(r'class="result"', html)[1:]  # 跳过头部

    for block in blocks:
        if len(results) >= max_results:
            break

        # 标题 + 链接
        title_m = re.search(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)<',
            block,
        )
        if not title_m:
            continue

        url = title_m.group(1)
        title = _clean_html(title_m.group(2))

        # 摘要
        snippet_m = re.search(
            r'class="result__snippet"[^>]*>(.*?)</a>',
            block,
            re.DOTALL,
        )
        snippet = _clean_html(snippet_m.group(1)) if snippet_m else ""

        results.append({
            "title": title.strip(),
            "snippet": snippet.strip(),
            "url": url.strip(),
        })

    logger.info("联网搜索 → %d 条结果", len(results))
    return results


def _clean_html(raw: str) -> str:
    """移除 HTML 标签与实体，保留纯文本。"""
    text = re.sub(r"<[^>]+>", "", raw)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#x27;", "'").replace("&nbsp;", " ")
    return text
