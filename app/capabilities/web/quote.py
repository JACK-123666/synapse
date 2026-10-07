"""股票行情查询工具。

数据源是腾讯财经的公开接口，无需 API Key、国内可直连：

- 代码搜索  https://smartbox.gtimg.cn/s3/?q=<关键词>&t=all
- 实时行情  https://qt.gtimg.cn/q=<市场前缀+代码>

为什么不用"搜索股价"：搜索只能拿到网页摘要，价格往往滞后甚至过期；
行情接口返回的是结构化实时数据，精确、快、且不会因为反爬而失败。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

import httpx
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://smartbox.gtimg.cn/s3/"
_QUOTE_URL = "https://qt.gtimg.cn/q="

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Referer": "https://gu.qq.com/",
}

#: 腾讯行情返回以 ~ 分隔的字段，这里是各字段的下标
_IDX = {
    "name": 1, "code": 2, "price": 3, "prev_close": 4, "open": 5, "volume": 6,
    "time": 30, "change": 31, "pct": 32, "high": 33, "low": 34,
    "turnover": 38, "pe": 39,
}

#: 各市场的计价单位
_UNIT = {"sh": "元", "sz": "元", "hk": "港元", "us": "美元"}


def _unescape_u(text: str) -> str:
    """把 \\uXXXX 转成真实字符（腾讯搜索接口会这样转义中文）。"""
    return re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), text)


def _norm_code(market: str, code: str) -> str:
    """规整代码本身。

    腾讯搜索接口对美股返回的是小写带交易所后缀的形式（如 aapl.oq），
    而行情接口要的是 usAAPL，因此去掉后缀并转大写。
    """
    if market == "us":
        return code.split(".")[0].upper()
    return code


def _norm_symbol(symbol: str) -> Optional[Dict[str, str]]:
    """把已带市场前缀或纯数字的输入规整成 {market, code, name}。"""
    s = symbol.strip()
    m = re.fullmatch(r"(sh|sz|hk|us)\.?([0-9A-Za-z.]+)", s, re.I)
    if m:
        market = m.group(1).lower()
        return {"market": market, "code": _norm_code(market, m.group(2)), "name": s}
    if re.fullmatch(r"\d{5}", s):
        return {"market": "hk", "code": s, "name": s}
    if re.fullmatch(r"\d{6}", s):
        # 6 开头是沪市主板，其余按深市处理
        return {"market": "sh" if s[0] == "6" else "sz", "code": s, "name": s}
    return None


async def _resolve(client: httpx.AsyncClient, symbol: str) -> Optional[Dict[str, str]]:
    """把用户输入解析成腾讯行情的完整代码。支持代码、名称、拼音首字母。"""
    direct = _norm_symbol(symbol)
    if direct:
        return direct

    resp = await client.get(_SEARCH_URL, params={"q": symbol.strip(), "t": "all"}, headers=_HEADERS)
    resp.raise_for_status()
    text = _unescape_u(resp.content.decode("gbk", errors="replace"))

    match = re.search(r'v_hint="([^"]*)"', text)
    if not match or not match.group(1):
        return None
    # 多条结果用 ^ 分隔，取第一条
    parts = match.group(1).split("^")[0].split("~")
    if len(parts) < 3:
        return None
    market = parts[0].lower()
    return {"market": market, "code": _norm_code(market, parts[1]), "name": parts[2]}


async def _fetch(client: httpx.AsyncClient, full_code: str) -> Optional[Dict[str, str]]:
    resp = await client.get(_QUOTE_URL + full_code, headers=_HEADERS)
    resp.raise_for_status()
    text = resp.content.decode("gbk", errors="replace")

    match = re.search(r'="([^"]*)"', text)
    if not match:
        return None
    fields = match.group(1).split("~")
    if len(fields) <= max(_IDX.values()):
        return None
    return {key: fields[i] for key, i in _IDX.items()}


def _fmt_volume(raw: str) -> str:
    try:
        return f"{int(raw) / 10000:.2f} 万手"
    except (TypeError, ValueError):
        return raw


def _fmt_time(raw: str) -> str:
    if len(raw) == 14 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]} {raw[8:10]}:{raw[10:12]}:{raw[12:]}"
    return raw


def _fmt_pct(raw: str) -> str:
    try:
        return f"{float(raw):+.2f}%"
    except (TypeError, ValueError):
        return raw


@tool("stock_quote")
async def stock_quote_tool(symbol: str) -> str:
    """查询股票实时行情，返回现价、涨跌额与涨跌幅、开高低、成交量、换手率、市盈率。

    支持 A 股 / 港股 / 美股，可直接用名称查询。查股价、行情、涨跌时优先用它，
    不要用 web_search —— 搜索引擎给的是网页摘要，价格往往不是最新的。

    Args:
        symbol: 股票名称或代码。例如：太极实业、600667、sh600667、00700、腾讯控股、AAPL
    """
    symbol = (symbol or "").strip()
    if not symbol:
        return "请提供股票名称或代码。"

    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            target = await _resolve(client, symbol)
            if target is None:
                return f"没有找到「{symbol}」对应的股票，请确认名称或代码是否正确。"

            full_code = f"{target['market']}{target['code']}"
            quote = await _fetch(client, full_code)
            if quote is None:
                return f"未能获取 {full_code} 的行情数据，该代码可能已停牌或不存在。"
    except Exception as exc:  # noqa: BLE001
        logger.warning("行情查询失败: %s", exc)
        return f"行情查询失败: {exc}"

    unit = _UNIT.get(target["market"], "")
    name = quote["name"] or target["name"]
    return "\n".join([
        f"{name}（{quote['code']}）实时行情",
        f"  现价    {quote['price']} {unit}    {quote['change']} ({_fmt_pct(quote['pct'])})",
        f"  今开    {quote['open']}        昨收  {quote['prev_close']}",
        f"  最高    {quote['high']}        最低  {quote['low']}",
        f"  成交量  {_fmt_volume(quote['volume'])}",
        f"  换手率  {quote['turnover']}%      市盈率  {quote['pe']}",
        f"  更新时间 {_fmt_time(quote['time'])}",
    ])
