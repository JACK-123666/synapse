"""网页抓取（带 SSRF 防护）与正文提取。

安全措施：
- 只允许 http / https
- 解析域名后检查全部 IP，禁止访问回环、内网、链路本地、保留地址（WEB_ALLOW_PRIVATE=true 可放开）
- 手动处理重定向，每一跳都重新校验
- 限制下载字节数与超时

已知限制：校验与实际连接之间存在 DNS 重绑定的时间窗口，高安全场景建议配合出口代理使用。
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import socket
from dataclasses import dataclass
from typing import Optional, Tuple
from urllib.parse import urljoin, urlsplit

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 SynapseBot/2.0"
)
_MAX_REDIRECTS = 5
_BLOCKED_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".lan", ".home.arpa")


class FetchError(Exception):
    """抓取失败（地址不合法、被安全策略拦截、HTTP 错误等）。"""


@dataclass
class FetchResult:
    """抓取结果：最终 URL、标题、正文，以及正文是否被截断。"""
    url: str
    final_url: str
    status: int
    content_type: str
    title: str
    text: str
    truncated: bool


def _is_public_ip(ip: ipaddress._BaseAddress) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


async def validate_public_url(url: str, allow_private: Optional[bool] = None) -> None:
    """校验 URL 可以安全访问；不合法时抛出 FetchError。"""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise FetchError("只支持 http / https 链接")
    host = (parts.hostname or "").strip().lower().rstrip(".")
    if not host:
        raise FetchError("链接缺少主机名")
    if allow_private is None:
        allow_private = get_settings().web_allow_private
    if allow_private:
        return
    if host == "localhost" or host.endswith(_BLOCKED_HOST_SUFFIXES):
        raise FetchError("出于安全考虑，禁止访问本机或内网地址")

    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if not _is_public_ip(literal):
            raise FetchError("出于安全考虑，禁止访问本机或内网地址")
        return

    port = parts.port or (443 if parts.scheme == "https" else 80)
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise FetchError(f"无法解析域名: {host}") from exc
    for info in infos:
        address = info[4][0].split("%")[0]
        if not _is_public_ip(ipaddress.ip_address(address)):
            raise FetchError("出于安全考虑，禁止访问解析到内网地址的域名")


async def fetch_raw(url: str) -> Tuple[str, int, str, bytes, bool]:
    """安全下载，返回 (最终 URL, 状态码, Content-Type, 内容, 是否被截断)。"""
    settings = get_settings()
    max_bytes = settings.web_fetch_max_bytes
    current = url
    async with httpx.AsyncClient(
        timeout=settings.web_fetch_timeout,
        follow_redirects=False,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/json,text/plain,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
    ) as client:
        for _ in range(_MAX_REDIRECTS + 1):
            await validate_public_url(current)
            async with client.stream("GET", current) as resp:
                if resp.is_redirect:
                    location = resp.headers.get("location", "")
                    if not location:
                        raise FetchError("重定向缺少目标地址")
                    current = urljoin(current, location)
                    continue
                if resp.status_code >= 400:
                    raise FetchError(f"HTTP {resp.status_code}")
                content_type = resp.headers.get("content-type", "")
                buffer = bytearray()
                truncated = False
                async for chunk in resp.aiter_bytes():
                    buffer.extend(chunk)
                    if len(buffer) > max_bytes:
                        truncated = True
                        del buffer[max_bytes:]
                        break
                return str(resp.url), resp.status_code, content_type, bytes(buffer), truncated
    raise FetchError("重定向次数过多")


def decode_bytes(data: bytes, content_type: str = "") -> str:
    """按 Content-Type 中的 charset 解码，失败时依次尝试 utf-8 / gb18030。"""
    match = re.search(r"charset=([\w-]+)", content_type or "", re.I)
    encodings = [match.group(1)] if match else []
    encodings += ["utf-8-sig", "gb18030"]
    for enc in encodings:
        try:
            return data.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue
    return data.decode("utf-8", errors="replace")


def _strip_tags(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", html)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def html_to_text(html: str, url: Optional[str] = None) -> Tuple[str, str]:
    """从 HTML 提取 (标题, 正文 Markdown)。trafilatura 失败时退化为去标签。"""
    title = ""
    text = ""
    try:
        import trafilatura

        text = trafilatura.extract(
            html,
            url=url,
            output_format="markdown",
            include_links=True,
            include_tables=True,
            favor_recall=True,
        ) or ""
        metadata = trafilatura.extract_metadata(html)
        if metadata is not None and getattr(metadata, "title", None):
            title = metadata.title or ""
    except Exception as exc:  # noqa: BLE001
        logger.debug("trafilatura 提取失败，退化为去标签: %s", exc)
    if not title:
        match = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)
        title = _strip_tags(match.group(1)) if match else ""
    if not text.strip():
        text = _strip_tags(html)
    return title.strip(), text.strip()


async def fetch_page(url: str, max_chars: Optional[int] = None) -> FetchResult:
    """抓取网页并提取正文；支持 HTML、纯文本、JSON、PDF。"""
    url = url.strip()
    final_url, status, content_type, data, truncated = await fetch_raw(url)
    ctype = content_type.lower()

    if "pdf" in ctype or final_url.lower().endswith(".pdf"):
        from app.capabilities.knowledge.loaders import extract_text

        title, text = final_url.rsplit("/", 1)[-1], extract_text("page.pdf", data)
    elif "html" in ctype or (not ctype and data[:200].lstrip().lower().startswith(b"<")):
        title, text = html_to_text(decode_bytes(data, content_type), final_url)
    elif ctype.startswith("text/") or "json" in ctype or "xml" in ctype:
        title, text = final_url, decode_bytes(data, content_type)
    else:
        raise FetchError(f"不支持的内容类型: {content_type or '未知'}")

    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
        truncated = True
    logger.info("网页抓取: %s -> %d 字符 (截断=%s)", final_url, len(text), truncated)
    return FetchResult(
        url=url,
        final_url=final_url,
        status=status,
        content_type=content_type,
        title=title,
        text=text,
        truncated=truncated,
    )
