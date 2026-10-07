"""定时任务服务。

- 任务定义持久化在数据库（schedules 表），启动时加载到 APScheduler（AsyncIOScheduler）
- 执行时用 Redis 锁防止多实例重复执行（Redis 不可用时降级为不加锁）
- 每次执行写入 schedule_runs，结果可推送到 Webhook（飞书 / 企业微信 / 钉钉 / 通用 JSON）

任务类型：
- prompt    ：以任务所有者身份让助手执行一句指令（payload.prompt）
- web_watch ：监控网页正文变化（payload.url），有变化时生成摘要
- kb_sync   ：定期重新抓取 URL 列表写入知识库（payload.kb_id、payload.urls）
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

import httpx
from sqlalchemy import select

from app.capabilities.knowledge.service import Principal
from app.capabilities.scheduler.cron import ScheduleError, build_trigger, next_runs, validate_cron
from app.config import get_settings
from app.core.db import session_scope
from app.models import Schedule, ScheduleRun, User, utcnow

logger = logging.getLogger(__name__)

KINDS = ("prompt", "web_watch", "kb_sync")
_LOCK_PREFIX = "synapse:schedule_lock"
_LOCK_TTL = 900
_MAX_OUTPUT = 8000


def schedule_to_dict(s: Schedule, with_next: bool = True) -> Dict[str, Any]:
    """把定时任务转成对外字典；with_next 为真时附带接下来几次执行时间。"""
    data = {
        "id": s.id,
        "name": s.name,
        "kind": s.kind,
        "cron": s.cron,
        "payload": s.payload,
        "webhook_url": s.webhook_url,
        "enabled": s.enabled,
        "session_id": s.session_id,
        "owner_id": s.owner_id,
        "last_run_at": s.last_run_at.isoformat() if s.last_run_at else None,
        "last_status": s.last_status,
        "created_at": s.created_at.isoformat() if s.created_at else None,
    }
    if with_next and s.enabled:
        try:
            data["next_runs"] = next_runs(s.cron, 3)
        except ScheduleError:
            data["next_runs"] = []
    return data


def run_to_dict(r: ScheduleRun) -> Dict[str, Any]:
    """把一次执行记录转成对外字典。"""
    return {
        "id": r.id,
        "schedule_id": r.schedule_id,
        "status": r.status,
        "output": r.output,
        "started_at": r.started_at.isoformat() if r.started_at else None,
        "finished_at": r.finished_at.isoformat() if r.finished_at else None,
    }


def build_webhook_payload(url: str, title: str, content: str) -> Dict[str, Any]:
    """按 Webhook 地址识别平台，生成对应的消息格式。"""
    text = f"{title}\n\n{content}"[:4000]
    if "open.feishu.cn" in url or "open.larksuite.com" in url:
        return {"msg_type": "text", "content": {"text": text}}
    if "qyapi.weixin.qq.com" in url:
        return {"msgtype": "text", "text": {"content": text}}
    if "oapi.dingtalk.com" in url:
        return {"msgtype": "text", "text": {"content": text}}
    return {"title": title, "content": content[:8000]}


def _validate_payload(kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if kind not in KINDS:
        raise ScheduleError(f"任务类型只能是 {KINDS}")
    payload = dict(payload or {})
    if kind == "prompt" and not str(payload.get("prompt", "")).strip():
        raise ScheduleError("prompt 任务需要 payload.prompt")
    if kind == "web_watch" and not str(payload.get("url", "")).startswith(("http://", "https://")):
        raise ScheduleError("web_watch 任务需要 payload.url（http/https）")
    if kind == "kb_sync":
        if not payload.get("kb_id"):
            raise ScheduleError("kb_sync 任务需要 payload.kb_id")
        urls = payload.get("urls") or []
        if not isinstance(urls, list) or not urls:
            raise ScheduleError("kb_sync 任务需要 payload.urls（URL 列表）")
    return payload


class SchedulerService:
    """定时任务服务。"""

    def __init__(self) -> None:
        self._scheduler = None

    # ---- 生命周期 ----

    @property
    def running(self) -> bool:
        """调度器当前是否在运行。"""
        return self._scheduler is not None and self._scheduler.running

    async def start(self) -> None:
        """启动 APScheduler，并把数据库中已启用的任务装载进调度器。"""
        settings = get_settings()
        if not settings.scheduler_enabled:
            logger.info("定时任务调度器未启用（SCHEDULER_ENABLED=false）")
            return
        if self.running:
            return
        from apscheduler.schedulers.asyncio import AsyncIOScheduler

        self._scheduler = AsyncIOScheduler(timezone=settings.scheduler_timezone)
        self._scheduler.start()
        async with session_scope() as session:
            rows = (
                await session.execute(select(Schedule).where(Schedule.enabled.is_(True)))
            ).scalars().all()
        for row in rows:
            self._add_job(row)
        logger.info("定时任务调度器已启动，加载 %d 个任务", len(rows))

    async def shutdown(self) -> None:
        """停止调度器并释放线程池。"""
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None
            logger.info("定时任务调度器已停止")

    def _add_job(self, schedule: Schedule) -> None:
        if not self.running:
            return
        try:
            self._scheduler.add_job(
                self.execute,
                trigger=build_trigger(schedule.cron),
                id=schedule.id,
                args=[schedule.id],
                replace_existing=True,
                coalesce=True,
                max_instances=1,
                misfire_grace_time=300,
            )
        except ScheduleError as exc:
            logger.error("定时任务 %s 的 cron 无效，未加载: %s", schedule.name, exc)

    def _remove_job(self, schedule_id: str) -> None:
        if self.running and self._scheduler.get_job(schedule_id):
            self._scheduler.remove_job(schedule_id)

    def job_ids(self) -> List[str]:
        """当前调度器里已装载的任务 ID 列表。"""
        return [job.id for job in self._scheduler.get_jobs()] if self.running else []

    # ---- CRUD ----

    async def _get(self, session, who: Principal, schedule_id: str) -> Schedule:
        schedule = await session.get(Schedule, schedule_id)
        if schedule is None or not (schedule.owner_id == who.user_id or who.is_admin):
            raise ScheduleError("定时任务不存在")
        return schedule

    async def find(self, who: Principal, name_or_id: str) -> Schedule:
        """按名称或 ID 查找任务；找不到时抛异常。"""
        async with session_scope() as session:
            schedule = await session.get(Schedule, name_or_id)
            if schedule is not None and (schedule.owner_id == who.user_id or who.is_admin):
                return schedule
            schedule = (
                await session.execute(
                    select(Schedule).where(Schedule.owner_id == who.user_id, Schedule.name == name_or_id)
                )
            ).scalar_one_or_none()
            if schedule is None:
                raise ScheduleError(f"找不到定时任务「{name_or_id}」")
            return schedule

    async def create(
        self,
        who: Principal,
        *,
        name: str,
        cron: str,
        kind: str = "prompt",
        payload: Optional[Dict[str, Any]] = None,
        webhook_url: str = "",
        enabled: bool = True,
        session_id: str = "",
    ) -> Dict[str, Any]:
        """创建定时任务：解析时间表达式、落库、装载进调度器。"""
        name = name.strip()
        if not name:
            raise ScheduleError("任务名称不能为空")
        cron = validate_cron(cron)
        payload = _validate_payload(kind, payload or {})
        if webhook_url and not webhook_url.startswith(("http://", "https://")):
            raise ScheduleError("webhook_url 需要以 http(s):// 开头")
        async with session_scope() as session:
            exists = (
                await session.execute(
                    select(Schedule.id).where(Schedule.owner_id == who.user_id, Schedule.name == name)
                )
            ).first()
            if exists:
                raise ScheduleError(f"定时任务「{name}」已存在")
            schedule = Schedule(
                owner_id=who.user_id,
                name=name,
                kind=kind,
                cron=cron,
                payload=payload,
                webhook_url=webhook_url,
                enabled=enabled,
            )
            session.add(schedule)
            await session.flush()
            schedule.session_id = session_id or f"schedule-{schedule.id[:8]}"
            data = schedule_to_dict(schedule)
        if enabled:
            self._add_job(schedule)
        return data

    async def list(self, who: Principal) -> List[Dict[str, Any]]:
        """列出该用户创建的定时任务。"""
        async with session_scope() as session:
            stmt = select(Schedule).order_by(Schedule.created_at)
            if not who.is_admin:
                stmt = stmt.where(Schedule.owner_id == who.user_id)
            return [schedule_to_dict(s) for s in (await session.execute(stmt)).scalars().all()]

    async def get(self, who: Principal, schedule_id: str) -> Dict[str, Any]:
        """查看单个定时任务。"""
        async with session_scope() as session:
            return schedule_to_dict(await self._get(session, who, schedule_id))

    async def update(self, who: Principal, schedule_id: str, **fields: Any) -> Dict[str, Any]:
        """修改定时任务；时间表达式变化时会重新装载调度。"""
        async with session_scope() as session:
            schedule = await self._get(session, who, schedule_id)
            if fields.get("name"):
                schedule.name = fields["name"].strip()
            if fields.get("cron"):
                schedule.cron = validate_cron(fields["cron"])
            if fields.get("payload") is not None:
                schedule.payload = _validate_payload(schedule.kind, fields["payload"])
            if fields.get("webhook_url") is not None:
                schedule.webhook_url = fields["webhook_url"]
            if fields.get("enabled") is not None:
                schedule.enabled = bool(fields["enabled"])
            data = schedule_to_dict(schedule)
        if schedule.enabled:
            self._add_job(schedule)
        else:
            self._remove_job(schedule.id)
        return data

    async def delete(self, who: Principal, schedule_id: str) -> None:
        """删除定时任务，并从调度器卸载。"""
        async with session_scope() as session:
            schedule = await self._get(session, who, schedule_id)
            await session.delete(schedule)
        self._remove_job(schedule_id)

    async def runs(self, who: Principal, schedule_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """查看某个任务的执行历史。"""
        async with session_scope() as session:
            await self._get(session, who, schedule_id)
            rows = (
                await session.execute(
                    select(ScheduleRun)
                    .where(ScheduleRun.schedule_id == schedule_id)
                    .order_by(ScheduleRun.started_at.desc())
                    .limit(limit)
                )
            ).scalars().all()
            return [run_to_dict(r) for r in rows]

    async def run_now(self, who: Principal, schedule_id: str) -> Dict[str, Any]:
        """立即手动执行一次任务，不影响原有调度。"""
        async with session_scope() as session:
            await self._get(session, who, schedule_id)
        return await self.execute(schedule_id)

    # ---- 执行 ----

    async def _acquire_lock(self, schedule_id: str) -> Optional[bool]:
        """获取执行锁：True 成功，False 已被占用，None 表示 Redis 不可用（不加锁继续执行）。"""
        try:
            from app.store import get_redis

            redis = await get_redis()
            return bool(await redis.set(f"{_LOCK_PREFIX}:{schedule_id}", "1", nx=True, ex=_LOCK_TTL))
        except Exception as exc:  # noqa: BLE001
            logger.warning("定时任务锁不可用（不加锁执行）: %s", exc)
            return None

    async def _release_lock(self, schedule_id: str) -> None:
        try:
            from app.store import get_redis

            redis = await get_redis()
            await redis.delete(f"{_LOCK_PREFIX}:{schedule_id}")
        except Exception:  # noqa: BLE001
            pass

    async def execute(self, schedule_id: str) -> Dict[str, Any]:
        """执行一次定时任务，返回执行记录。"""
        async with session_scope() as session:
            schedule = await session.get(Schedule, schedule_id)
            if schedule is None:
                raise ScheduleError("定时任务不存在")
            owner = await session.get(User, schedule.owner_id)
            run = ScheduleRun(schedule_id=schedule_id, status="running")
            session.add(run)
            await session.flush()
            run_id = run.id

        lock = await self._acquire_lock(schedule_id)
        if lock is False:
            status, output = "skipped", "上一次执行尚未结束（或其他实例正在执行），本次跳过"
        else:
            try:
                output = await self._run_kind(schedule, owner)
                status = "success"
            except Exception as exc:  # noqa: BLE001
                logger.exception("定时任务 %s 执行失败: %s", schedule.name, exc)
                status, output = "failed", f"执行失败: {exc}"
            finally:
                if lock:
                    await self._release_lock(schedule_id)

            if schedule.webhook_url and output and not output.startswith("[无变化]"):
                try:
                    await self._push_webhook(schedule.webhook_url, f"[Synapse 定时任务] {schedule.name}", output)
                except Exception as exc:  # noqa: BLE001
                    output += f"\n\n（Webhook 推送失败: {exc}）"

        async with session_scope() as session:
            run = await session.get(ScheduleRun, run_id)
            run.status, run.output, run.finished_at = status, output[:_MAX_OUTPUT], utcnow()
            schedule_row = await session.get(Schedule, schedule_id)
            if schedule_row is not None:
                schedule_row.last_run_at, schedule_row.last_status = utcnow(), status
                if schedule.kind == "web_watch":
                    schedule_row.payload = dict(schedule.payload)
            data = run_to_dict(run)
        logger.info("定时任务 %s 执行完成: %s", schedule.name, status)
        return data

    async def _run_kind(self, schedule: Schedule, owner: Optional[User]) -> str:
        if owner is None or not owner.is_active:
            raise ScheduleError("任务所有者不存在或已停用")
        if schedule.kind == "prompt":
            return await self._run_prompt(schedule, owner)
        if schedule.kind == "web_watch":
            return await self._run_web_watch(schedule, owner)
        if schedule.kind == "kb_sync":
            return await self._run_kb_sync(schedule, owner)
        raise ScheduleError(f"未知任务类型: {schedule.kind}")

    async def _run_prompt(self, schedule: Schedule, owner: User) -> str:
        from app.core.deps import CurrentUser
        from app.services.chat import ChatInput, get_chat_service

        result = await get_chat_service().chat(
            CurrentUser(id=owner.id, username=owner.username, role=owner.role),
            ChatInput(
                session_id=schedule.session_id or f"schedule-{schedule.id[:8]}",
                message=schedule.payload["prompt"],
                user_id=owner.id,
                web_search=bool(schedule.payload.get("web_search", True)),
            ),
        )
        return result.reply

    async def _run_web_watch(self, schedule: Schedule, owner: User) -> str:
        from app.capabilities.web.fetch import fetch_page
        from app.core.context import RequestContext, request_context
        from app.llm.gateway import LLMError, get_llm_client

        payload = dict(schedule.payload)
        page = await fetch_page(payload["url"], max_chars=20000)
        digest = hashlib.sha256(page.text.encode("utf-8")).hexdigest()
        previous_hash = payload.get("last_hash")
        previous_excerpt = payload.get("last_excerpt", "")
        payload["last_hash"], payload["last_excerpt"] = digest, page.text[:3000]
        schedule.payload = payload

        if previous_hash is None:
            return f"已记录网页初始版本：{page.title or page.final_url}（{len(page.text)} 字符），之后有变化时会通知。"
        if previous_hash == digest:
            return f"[无变化] {page.title or page.final_url}"

        summary = page.text[:800]
        try:
            with request_context(RequestContext(user_id=owner.id, username=owner.username, role=owner.role)):
                summary = await get_llm_client().chat(
                    [{
                        "role": "user",
                        "content": f"旧版本摘录:\n{previous_excerpt}\n\n新版本:\n{page.text[:6000]}\n\n请用中文简要说明新版本有哪些变化。",
                    }],
                    system="你是网页变更分析助手，只描述实际发生的变化，控制在 300 字以内。",
                    temperature=0.2,
                    max_tokens=600,
                )
        except LLMError as exc:
            logger.warning("网页变化摘要生成失败，使用原文摘录: %s", exc)
        return f"网页有更新：{page.title or page.final_url}\n{page.final_url}\n\n{summary}"

    async def _run_kb_sync(self, schedule: Schedule, owner: User) -> str:
        from app.capabilities.knowledge.service import get_knowledge_service

        who = Principal(user_id=owner.id, is_admin=owner.role == "admin")
        service = get_knowledge_service()
        ok, failed = [], []
        for url in schedule.payload.get("urls", []):
            try:
                doc = await service.ingest_url(who, schedule.payload["kb_id"], url)
                ok.append(f"{doc['filename']}（{doc['chunk_count']} 片段）")
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{url}: {exc}")
        lines = [f"知识库同步完成：成功 {len(ok)} 个，失败 {len(failed)} 个"]
        lines += [f"- {x}" for x in ok] + [f"- 失败 {x}" for x in failed]
        return "\n".join(lines)

    async def _push_webhook(self, url: str, title: str, content: str) -> None:
        from app.capabilities.web.fetch import validate_public_url

        await validate_public_url(url)
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=build_webhook_payload(url, title, content))
            resp.raise_for_status()


_service: Optional[SchedulerService] = None


def get_scheduler_service() -> SchedulerService:
    """获取定时任务服务单例。"""
    global _service
    if _service is None:
        _service = SchedulerService()
    return _service
