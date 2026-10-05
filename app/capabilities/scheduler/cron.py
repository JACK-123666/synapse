"""cron 表达式校验与自然语言时间解析。

解析顺序：
1. 已经是 5 段 cron 表达式 → 直接校验
2. 常见中文 / 英文描述（每天 9 点、每周一上午 10:30、每隔 15 分钟……）→ 规则解析
3. 其他描述 → 交给 LLM 结构化输出 cron（失败时报错，提示用户改用 cron）
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import List, Optional

from apscheduler.triggers.cron import CronTrigger
from pydantic import BaseModel, Field

from app.config import get_settings

logger = logging.getLogger(__name__)


class ScheduleError(ValueError):
    """定时任务参数错误。"""


_WEEKDAYS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "日": 0, "天": 0, "末": 6}
_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _cn_number(text: str) -> Optional[int]:
    """解析 1~59 的阿拉伯数字或中文数字（如 九、十二、二十三）。"""
    text = text.strip()
    if text.isdigit():
        return int(text)
    if not text or any(ch not in _CN_DIGITS for ch in text):
        return None
    if text == "十":
        return 10
    if "十" in text:
        tens, _, ones = text.partition("十")
        return (_CN_DIGITS.get(tens, 1) if tens else 1) * 10 + (_CN_DIGITS.get(ones, 0) if ones else 0)
    return _CN_DIGITS.get(text)


_DOW_NAMES = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"]


def _dow_number(token: str) -> int:
    token = token.strip().lower()
    if token.isdigit() and 0 <= int(token) <= 7:
        return int(token) % 7
    if token[:3] in _DOW_NAMES:
        return _DOW_NAMES.index(token[:3])
    raise ScheduleError(f"星期字段不合法: {token}")


def _expand_dow(field: str) -> str:
    """把标准 crontab 的星期字段（0/7=周日，1=周一）转换为 APScheduler 的英文名称列表。

    APScheduler 3.x 的 from_crontab 不做转换（它的 0 表示周一），直接使用会整体错位一天。
    """
    if field == "*":
        return "*"
    days = set()
    for part in field.split(","):
        step = 1
        if "/" in part:
            part, step_text = part.split("/", 1)
            if not step_text.isdigit() or int(step_text) < 1:
                raise ScheduleError(f"星期字段步长不合法: {field}")
            step = int(step_text)
        if part == "*":
            start, end = 0, 6
        elif "-" in part:
            a, b = part.split("-", 1)
            start, end = _dow_number(a), _dow_number(b)
            if b.strip() == "7":
                end = 7
        else:
            start = _dow_number(part)
            end = 6 if step > 1 else start
        if end < start:
            end += 7
        for day in range(start, end + 1, step):
            days.add(day % 7)
    return ",".join(_DOW_NAMES[d] for d in sorted(days))


def build_trigger(expr: str) -> CronTrigger:
    """由标准 5 段 crontab 表达式构造 APScheduler 触发器（星期字段按标准语义转换）。"""
    fields = " ".join((expr or "").split()).split(" ")
    if len(fields) != 5:
        raise ScheduleError("cron 表达式需要 5 段：分 时 日 月 周")
    minute, hour, day, month, dow = fields
    try:
        return CronTrigger(
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            day_of_week=_expand_dow(dow),
            timezone=get_settings().scheduler_timezone,
        )
    except ValueError as exc:
        raise ScheduleError(f"cron 表达式不合法: {exc}") from exc


def validate_cron(expr: str) -> str:
    """校验 5 段 cron 表达式，返回规范化后的表达式。"""
    expr = " ".join((expr or "").split())
    build_trigger(expr)
    return expr


def next_runs(expr: str, count: int = 3) -> List[str]:
    """返回接下来几次触发时间（ISO 格式）。"""
    trigger = build_trigger(validate_cron(expr))
    runs: List[str] = []
    previous = None
    now = datetime.now(trigger.timezone)
    for _ in range(count):
        fire = trigger.get_next_fire_time(previous, previous or now)
        if fire is None:
            break
        runs.append(fire.isoformat())
        previous = fire
    return runs


def _parse_time_of_day(text: str) -> Optional[tuple]:
    """从文本中提取 (时, 分)，支持 9点、9:30、下午3点半、晚上8点15分、上午十点。"""
    match = re.search(
        r"(凌晨|早上|早晨|上午|中午|下午|傍晚|晚上)?\s*([0-9一二两三四五六七八九十]{1,3})\s*(?:[:：点时])\s*(半|[0-9一二三四五六七八九十]{1,3})?\s*分?",
        text,
    )
    if not match:
        return None
    period, hour_text, minute_text = match.groups()
    hour = _cn_number(hour_text)
    if hour is None or hour > 24:
        return None
    if minute_text == "半":
        minute = 30
    elif minute_text:
        minute = _cn_number(minute_text)
        if minute is None or minute > 59:
            return None
    else:
        minute = 0
    if period in ("下午", "傍晚", "晚上") and hour < 12:
        hour += 12
    if period == "中午" and hour < 6:
        hour += 12
    return hour % 24, minute


def parse_rule_based(text: str) -> Optional[str]:
    """规则解析常见时间描述；无法解析时返回 None。"""
    t = text.strip().lower().replace("周日", "周天").replace("星期", "周").replace("礼拜", "周")

    every_minutes = re.search(r"每隔?\s*([0-9一二三四五六七八九十]+)\s*分钟|every\s+(\d+)\s+minutes?", t)
    if every_minutes:
        n = _cn_number(every_minutes.group(1) or every_minutes.group(2))
        if n and 1 <= n <= 59:
            return f"*/{n} * * * *"
    every_hours = re.search(r"每隔?\s*([0-9一二三四五六七八九十]+)\s*(?:个)?小时|every\s+(\d+)\s+hours?", t)
    if every_hours:
        n = _cn_number(every_hours.group(1) or every_hours.group(2))
        if n and 1 <= n <= 23:
            return f"0 */{n} * * *"
    if re.search(r"每(个)?小时|every\s+hour|hourly", t):
        return "0 * * * *"

    time_of_day = _parse_time_of_day(t)
    english = re.search(r"at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", t)
    if time_of_day is None and english:
        hour = int(english.group(1)) % 12 + (12 if english.group(3) == "pm" else 0)
        if english.group(3) is None:
            hour = int(english.group(1))
        time_of_day = (hour % 24, int(english.group(2) or 0))
    hour, minute = time_of_day if time_of_day else (9, 0)

    if re.search(r"工作日|weekday", t):
        return f"{minute} {hour} * * 1-5"
    weekly = re.search(r"每周([一二三四五六天日末])", t)
    if weekly:
        return f"{minute} {hour} * * {_WEEKDAYS[weekly.group(1)]}"
    monthly = re.search(r"每月\s*([0-9一二三四五六七八九十]+)\s*[号日]", t)
    if monthly:
        day = _cn_number(monthly.group(1))
        if day and 1 <= day <= 31:
            return f"{minute} {hour} {day} * *"
    if re.search(r"每天|每日|每晚|每早|daily|every\s+day", t) and (time_of_day or "每天" in t or "每日" in t):
        return f"{minute} {hour} * * *"
    return None


class CronSpec(BaseModel):
    """LLM 结构化输出：把时间描述转换为 cron。"""

    cron: str = Field(description="5 段 crontab 表达式：分 时 日 月 周（周日=0）")
    explanation: str = Field(default="", description="用中文解释这个 cron 的含义")


async def to_cron(text: str) -> str:
    """把 cron 表达式或自然语言时间描述转换为校验过的 cron。"""
    text = (text or "").strip()
    if not text:
        raise ScheduleError("缺少执行时间")
    if re.fullmatch(r"[\d*/,\-]+(\s+[\d*/,\-a-zA-Z]+){4}", text):
        return validate_cron(text)
    rule = parse_rule_based(text)
    if rule:
        return validate_cron(rule)

    from app.llm.factory import get_chat_model

    try:
        model = get_chat_model(temperature=0.0, max_tokens=200)
        structured = model.with_structured_output(CronSpec, method="function_calling")
        spec = await structured.ainvoke(
            f"把下面的执行时间描述转换为 5 段 crontab 表达式（时区 {get_settings().scheduler_timezone}）：{text}"
        )
        return validate_cron(spec.cron)
    except ScheduleError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM 解析时间描述失败: %s", exc)
        raise ScheduleError(f"无法理解执行时间「{text}」，请改用 cron 表达式，如 0 9 * * *") from exc
