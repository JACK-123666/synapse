"""示例插件：演示如何用 Capability 接口扩展 Synapse。

- 两个工具：get_current_time、calculator
- 一个意图：utility_tools
- 不提供 Agent / 路由：平台会自动生成 example_plugin_agent，
  并路由为 [example_plugin_agent, general_agent, fallback_agent]
"""

from __future__ import annotations

import ast
import operator
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool

from app.capabilities.base import Capability, CapabilityTool
from app.intent.catalog import IntentSpec

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    """只允许数字与四则运算，拒绝任何名称、函数调用、属性访问。"""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("指数过大")
        return _OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("只支持数字与 + - * / // % ** 运算")


@tool("get_current_time")
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """获取指定时区的当前日期与时间。

    Args:
        timezone: IANA 时区名，如 Asia/Shanghai、America/New_York、Asia/Tokyo
    """
    try:
        now = datetime.now(ZoneInfo(timezone))
    except ZoneInfoNotFoundError:
        return f"未知时区: {timezone}"
    weekday = "一二三四五六日"[now.weekday()]
    return f"{timezone} 当前时间：{now:%Y-%m-%d %H:%M:%S}（星期{weekday}）"


@tool("calculator")
def calculator(expression: str) -> str:
    """安全计算数学表达式，支持 + - * / // % ** 和括号。

    Args:
        expression: 数学表达式，如 (12 + 30) * 4
    """
    try:
        value = _safe_eval(ast.parse(expression.replace("×", "*").replace("÷", "/"), mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError) as exc:
        return f"无法计算: {exc}"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{expression} = {value}"


class ExamplePlugin(Capability):
    name = "example_plugin"
    description = "时间与计算小工具"

    def tools(self):
        return [
            CapabilityTool(get_current_time, tags=("utility",)),
            CapabilityTool(calculator, tags=("utility",)),
        ]

    def intents(self):
        return [
            IntentSpec(
                name="utility_tools",
                description="查询当前时间 / 日期，或计算数学表达式",
                keywords=["几点", "现在时间", "今天几号", "星期几", "等于多少", "算一下"],
                examples=["现在几点了", "今天星期几", "帮我算一下 (12+30)*4", "What time is it in Tokyo?"],
            )
        ]
