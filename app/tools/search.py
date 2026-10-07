"""联网搜索工具 —— 无需 API Key。

主用 **Bing 中国站**（cn.bing.com）：国内可直连，HTML 结构稳定可解析。
备用 DuckDuckGo HTML，但它现在会对非浏览器流量返回 202 + 人机验证页
（"Unfortunately, bots use DuckDuckGo too"），解析结果恒为 0 条，
因此只作为境外环境的兜底，且不再作为首选。
"""

from __future__ import annotations

import logging
import re
from html import unescape
from typing import Dict, List

import httpx

logger = logging.getLogger(__name__)

_BING_URL = "https://cn.bing.com/search"
_DDG_URL = "https://html.duckduckgo.com/html/"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


async def web_search(
    query: str,
    max_results: int = 5,
    timeout: float = 10.0,
) -> List[Dict[str, str]]:
    """联网搜索，返回 title / snippet / url 列表。

    失败时返回空列表，不抛异常 —— 由调用方按 best-effort 降级。
    """
    if not query.strip():
        return []

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        try:
            resp = await client.get(
                _BING_URL,
                params={"q": query, "setlang": "zh-CN"},
                headers=_HEADERS,
            )
            resp.raise_for_status()
            results = _parse_bing(resp.text, max_results)
            if results:
                logger.info("联网搜索(Bing) → %d 条结果", len(results))
                return results
            logger.warning("联网搜索(Bing) 解析到 0 条，改用备用源")
        except Exception as exc:  # noqa: BLE001
            logger.warning("联网搜索(Bing) 失败: %s", exc)

        try:
            resp = await client.post(_DDG_URL, data={"q": query, "b": ""}, headers=_HEADERS)
            resp.raise_for_status()
            results = _parse_ddg(resp.text, max_results)
            logger.info("联网搜索(DuckDuckGo) → %d 条结果", len(results))
            return results
        except Exception as exc:  # noqa: BLE001
            logger.warning("联网搜索失败: %s", exc)
            return []


def _parse_bing(html: str, max_results: int) -> List[Dict[str, str]]:
    """从 Bing 结果页抽取条目。结果块是 <li class="b_algo">。"""
    results: List[Dict[str, str]] = []
    blocks = re.findall(r'<li class="b_algo".*?(?=<li class="b_algo"|</ol>)', html, re.S)

    for block in blocks:
        if len(results) >= max_results:
            break

        title_m = re.search(
            r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block, re.S
        )
        if not title_m:
            continue
        url = _clean(title_m.group(1))
        if not url.startswith("http"):
            continue

        snippet_m = re.search(
            r'class="b_lineclamp[^"]*"[^>]*>(.*?)</p>', block, re.S
        ) or re.search(r"<p[^>]*>(.*?)</p>", block, re.S)

        results.append({
            "title": _clean(title_m.group(2)),
            "snippet": _clean(snippet_m.group(1)) if snippet_m else "",
            "url": url,
        })

    return results


def _parse_ddg(html: str, max_results: int) -> List[Dict[str, str]]:
    """从 DuckDuckGo HTML 结果页抽取条目（备用源）。"""
    results: List[Dict[str, str]] = []
    blocks = re.split(r'class="result"', html)[1:]

    for block in blocks:
        if len(results) >= max_results:
            break
        title_m = re.search(r'class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)<', block)
        if not title_m:
            continue
        snippet_m = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', block, re.S)
        results.append({
            "title": _clean(title_m.group(2)),
            "snippet": _clean(snippet_m.group(1)) if snippet_m else "",
            "url": _clean(title_m.group(1)),
        })

    return results


def _clean(raw: str) -> str:
    """移除 HTML 标签、还原实体、压缩空白。"""
    text = re.sub(r"<[^>]+>", "", raw)
    return re.sub(r"\s+", " ", unescape(text)).strip()
