from __future__ import annotations

import asyncio
import base64
import datetime as dt
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import aiohttp
from telethon import Button, TelegramClient, events, functions, types

from .config import Settings
from .database import Archive, StoredMessage
from .llm import (
    CONTENT_CATEGORIES,
    CheckinSuggestionDecision,
    EventDecision,
    InsightLLM,
    limit_summary_messages,
    render_answer,
)

log = logging.getLogger("tg_insight")

GROUPS_PAGE_SIZE = 8
MAX_SELECTED_SOURCES = 50
DEFAULT_RECENT_MESSAGES = 20
MAX_RECENT_MESSAGES = 50
CONTENT_CLASSIFICATION_MESSAGES = 60
CONTENT_CLASSIFICATION_MAX_CHARS = 12_000
CONTENT_CLASSIFICATION_TTL_DAYS = 7
CONTENT_CLASSIFICATIONS_STATE = "content_classifications"
CONTENT_OVERRIDES_STATE = "content_overrides"
CHECKIN_CONFIGS_STATE = "checkin_configs"
DEFAULT_CHECKIN_TEXT = "签到"
DEFAULT_CHECKIN_HOUR = 0
DEFAULT_CHECKIN_MINUTE = 30
CHECKIN_OFFSET_MIN_MS = 300
CHECKIN_OFFSET_MAX_MS = 800
MAX_CHECKIN_TARGETS = 50
MAX_CHECKIN_TEXT_LENGTH = 1_000
CHECKIN_MAX_ATTEMPTS = 3
CHECKIN_FAILURE_ESCALATION_DAYS = 3
CHECKIN_VERIFY_TIMEOUT_SECONDS = 8
CHECKIN_HISTORY_LIMIT = 30
CHECKIN_REPORT_STATE = "last_checkin_report_day"
CHECKIN_REPORT_ENABLED_STATE = "checkin_report_enabled"
CHECKIN_ALERT_STATE = "checkin_alerts"
ALERT_FEEDBACK_STATE = "alert_feedback"
ALERT_IGNORED_TOPICS_STATE = "alert_ignored_topics"
CHECKIN_SUGGESTION_STATE = "checkin_suggestions"
CHECKIN_SUGGESTION_IGNORES_STATE = "checkin_suggestion_ignores"
TOPIC_SUBSCRIPTIONS_STATE = "topic_subscriptions"
DIGEST_CURSORS_STATE = "digest_cursors"
CHECKIN_SUGGESTION_REPLY_WAIT_SECONDS = 12
CHECKIN_SUGGESTION_STATUS_URL = "https://status.input.im/api/status"
CHECKIN_SUGGESTION_STATUS_TIMEOUT_SECONDS = 8
CHECKIN_SUGGESTION_STATUS_MAX_AGE_SECONDS = 15 * 60
CHECKIN_SUGGESTION_RETRY_INITIAL_SECONDS = 30
CHECKIN_SUGGESTION_RETRY_MAX_SECONDS = 15 * 60
CHECKIN_SUGGESTION_MAX_AGE = dt.timedelta(hours=24)
CHECKIN_SUCCESS_KEYWORDS = (
    "签到成功",
    "已签到",
    "签到完成",
    "获得积分",
    "签到成功啦",
    "恭喜签到",
    "打卡成功",
    "打卡完成",
)
CHECKIN_FAILURE_KEYWORDS = ("签到失败", "操作失败", "请稍后重试", "无权限", "已过期")
ALERT_EVENT_HINTS = (
    "紧急", "重要通知", "故障", "中断", "截止", "封禁", "下架", "涨价", "降价",
    "维护", "漏洞", "攻击", "泄露", "发布", "报名", "活动", "规则更新", "breaking",
)
ALERT_KEYWORDS_DEFAULT = "紧急,重要通知,故障,截止,封禁,下架,涨价,维护,漏洞,攻击,泄露,发布,报名"
ALERT_CONTEXT_MESSAGES = 12
ALERT_MAX_MESSAGE_AGE = dt.timedelta(minutes=45)
ALERT_TOPIC_COOLDOWN = dt.timedelta(minutes=15)
TOPIC_SUBSCRIPTION_COOLDOWN = dt.timedelta(minutes=30)
MAX_TOPIC_SUBSCRIPTIONS = 20
CHECKIN_COMMAND_PATTERN = re.compile(
    r"(?<![a-z0-9_])/(?:qd|checkin)(?:@[a-z0-9_]{5,})?(?![a-z0-9_])", re.IGNORECASE
)


@dataclass(frozen=True)
class SourceChat:
    entity: Any
    chat_id: int
    name: str
    username: str | None
    target_kind: str = "group"


@dataclass(frozen=True)
class AlertRecord:
    sent_at: dt.datetime
    new_information: str


@dataclass
class CheckinSuggestionCandidate:
    chat_id: int
    message_id: int
    source_name: str
    source_username: str | None
    source_sender: str
    source_time: dt.datetime
    source_text: str
    bot_reply: str = ""


@dataclass(frozen=True)
class CheckinRecord:
    day: str
    at: str
    status: str
    detail: str = ""


class CheckinVerificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckinConfig:
    enabled: bool = True
    text: str = DEFAULT_CHECKIN_TEXT
    hour: int = DEFAULT_CHECKIN_HOUR
    minute: int = DEFAULT_CHECKIN_MINUTE
    last_success_day: str = ""
    attempt_day: str = ""
    attempt_count: int = 0
    retry_at: str = ""
    scheduled_for: str = ""
    topic_id: int | None = None
    last_status: str = ""
    last_detail: str = ""
    failure_streak: int = 0
    history: tuple[CheckinRecord, ...] = ()


class TelegramInsightService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.archive = Archive(
            settings.database_path,
            settings.archive_max_messages,
            settings.archive_min_free_mb,
            settings.archive_max_bytes,
        )
        self.llm = InsightLLM(
            settings.llm_api_key,
            settings.llm_model,
            settings.llm_base_url,
            settings.llm_fallback_model,
        )
        self.user = TelegramClient(
            str(settings.user_session), settings.api_id, settings.api_hash
        )
        self.bot = TelegramClient(
            str(settings.bot_session), settings.api_id, settings.api_hash
        )
        self.sources: dict[int, SourceChat] = {}
        self.available_sources: dict[int, SourceChat] = {}
        self.available_checkin_targets: dict[int, SourceChat] = {}
        self._configured_source_ids: set[int] = set()
        self._digest_lock = asyncio.Lock()
        self._source_lock = asyncio.Lock()
        self._content_lock = asyncio.Lock()
        self._checkin_lock = asyncio.Lock()
        self._backfill_tasks: set[asyncio.Task[None]] = set()
        self._classification_tasks: set[asyncio.Task[None]] = set()
        self._pending_schedule_users: set[int] = set()
        self._pending_checkin_text_users: dict[int, int] = {}
        self._pending_checkin_schedule_users: dict[int, int] = {}
        self._pending_checkin_topic_users: dict[int, int] = {}
        self._pending_alert_keywords_users: set[int] = set()
        self._pending_topic_users: dict[int, int] = {}
        self._checkin_waiters: dict[tuple[int, str], asyncio.Future[tuple[str, str]]] = {}
        self._refresh_lock = asyncio.Lock()
        self._alert_tasks: set[asyncio.Task[None]] = set()
        self._alert_last_sent: dict[tuple[int, str], AlertRecord] = {}
        self._alert_feedback: dict[tuple[int, str], str] = {}
        self._alert_lock = asyncio.Lock()
        self._checkin_suggestion_days: set[tuple[int, str]] = set()
        self._checkin_suggestion_candidates: dict[tuple[int, int], CheckinSuggestionCandidate] = {}
        self._checkin_suggestion_tasks: set[asyncio.Task[None]] = set()
        self._user_id: int | None = None

    async def run(self) -> None:
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.archive.initialize()
        await self.user.connect()
        if not await self.user.is_user_authorized():
            raise RuntimeError("Telegram user session is not authorized; run auth first")
        me = await self.user.get_me()
        self._user_id = int(me.id)

        await self.bot.start(bot_token=self.settings.bot_token)
        await self._discover_source_chats()
        await self._resolve_sources()
        self._register_handlers()
        await self._set_bot_commands()

        await self._backfill()
        await self._classify_sources(tuple(self.sources.values()))
        self.archive.prune(self.settings.retention_days)
        scheduler = asyncio.create_task(self._scheduler(), name="daily-digest")
        log.info("Service ready with %d source chats", len(self.sources))
        try:
            await asyncio.gather(
                self.user.run_until_disconnected(),
                self.bot.run_until_disconnected(),
                scheduler,
            )
        finally:
            scheduler.cancel()
            for task in self._backfill_tasks:
                task.cancel()
            for task in self._classification_tasks:
                task.cancel()
            for task in self._alert_tasks:
                task.cancel()
            for task in self._checkin_suggestion_tasks:
                task.cancel()
            await asyncio.gather(*self._backfill_tasks, return_exceptions=True)
            await asyncio.gather(*self._classification_tasks, return_exceptions=True)
            await asyncio.gather(*self._alert_tasks, return_exceptions=True)
            await asyncio.gather(*self._checkin_suggestion_tasks, return_exceptions=True)
            await self.user.disconnect()
            await self.bot.disconnect()

    async def auth(self, phone: str | None = None) -> None:
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        await self.user.start(phone=phone)
        me = await self.user.get_me()
        log.info("Authorized Telegram user id=%s", me.id)
        await self.user.disconnect()

    async def _discover_source_chats(self) -> None:
        self.available_sources.clear()
        self.available_checkin_targets.clear()
        async for dialog in self.user.iter_dialogs():
            if getattr(dialog, "is_group", False):
                source = await self._source_from_entity(dialog.entity, target_kind="group")
                self.available_sources[source.chat_id] = source
                self.available_checkin_targets[source.chat_id] = source
            elif getattr(dialog.entity, "bot", False):
                target = await self._source_from_entity(dialog.entity, target_kind="bot")
                self.available_checkin_targets[target.chat_id] = target
        log.info(
            "Discovered %d joined group chats and %d check-in targets",
            len(self.available_sources),
            len(self.available_checkin_targets),
        )

    async def _source_from_entity(self, entity: Any, target_kind: str = "group") -> SourceChat:
        chat_id = int(await self.user.get_peer_id(entity))
        title = (
            getattr(entity, "title", None)
            or _sender_name(entity)
            or getattr(entity, "username", None)
            or str(chat_id)
        )
        return SourceChat(
            entity=entity,
            chat_id=chat_id,
            name=str(title),
            username=getattr(entity, "username", None),
            target_kind=target_kind,
        )

    async def _resolve_sources(self) -> None:
        self.sources.clear()
        self._configured_source_ids.clear()
        for configured in self.settings.source_chats:
            entity = await self.user.get_entity(configured)
            source = await self._source_from_entity(entity)
            self.available_sources[source.chat_id] = source
            self.available_checkin_targets[source.chat_id] = source
            self.sources[source.chat_id] = source
            self._configured_source_ids.add(source.chat_id)
            log.info("Resolved configured source %s as %s", source.name, source.chat_id)

        for chat_id in _source_ids_from_state(
            self.archive.get_state("selected_source_chats")
        ):
            source = self.available_sources.get(chat_id)
            if source is None:
                log.warning("Selected source chat %s is no longer available", chat_id)
                continue
            self.sources[chat_id] = source

    async def _toggle_source(self, source: SourceChat) -> str:
        async with self._source_lock:
            if source.chat_id in self.sources:
                if source.chat_id in self._configured_source_ids:
                    return "fixed"
                del self.sources[source.chat_id]
                self._save_selected_sources()
                return "removed"

            if len(self.sources) >= MAX_SELECTED_SOURCES:
                return "limit"
            self.sources[source.chat_id] = source
            self._save_selected_sources()
        self._schedule_backfill((source,))
        return "added"

    def _save_selected_sources(self) -> None:
        selected = sorted(
            chat_id
            for chat_id in self.sources
            if chat_id not in self._configured_source_ids
        )
        self.archive.set_state("selected_source_chats", json.dumps(selected))

    def _schedule_backfill(self, sources: Sequence[SourceChat]) -> None:
        if not sources:
            return
        task = asyncio.create_task(
            self._backfill_and_classify(sources), name="selected-source-backfill"
        )
        self._backfill_tasks.add(task)
        task.add_done_callback(self._finish_backfill_task)

    def _finish_backfill_task(self, task: asyncio.Task[None]) -> None:
        self._backfill_tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            log.exception("Background source backfill failed")

    async def _backfill_and_classify(self, sources: Sequence[SourceChat]) -> None:
        await self._backfill_sources(sources)
        await self._classify_sources(sources)

    def _schedule_classification(
        self, sources: Sequence[SourceChat], force: bool = False
    ) -> None:
        if not sources:
            return
        task = asyncio.create_task(
            self._classify_sources(sources, force=force), name="content-classification"
        )
        self._classification_tasks.add(task)
        task.add_done_callback(self._finish_classification_task)

    def _finish_classification_task(self, task: asyncio.Task[None]) -> None:
        self._classification_tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            log.exception("Background content classification failed")

    def _register_handlers(self) -> None:
        self.user.add_event_handler(self._on_new_message, events.NewMessage)
        self.user.add_event_handler(self._on_edited_message, events.MessageEdited)
        self.user.add_event_handler(self._on_deleted_message, events.MessageDeleted)
        self.bot.add_event_handler(self._on_help, events.NewMessage(pattern=r"^/(start|help)$"))
        self.bot.add_event_handler(self._on_status, events.NewMessage(pattern=r"^/status$"))
        self.bot.add_event_handler(self._on_summary, events.NewMessage(pattern=r"^/(summary|digest)$"))
        self.bot.add_event_handler(self._on_settings, events.NewMessage(pattern=r"^/settings$"))
        self.bot.add_event_handler(self._on_content, events.NewMessage(pattern=r"^/content$"))
        self.bot.add_event_handler(self._on_checkin, events.NewMessage(pattern=r"^/checkin$"))
        self.bot.add_event_handler(self._on_refresh, events.NewMessage(pattern=r"^/refresh$"))
        self.bot.add_event_handler(self._on_alerts, events.NewMessage(pattern=r"^/alerts$"))
        self.bot.add_event_handler(self._on_topics, events.NewMessage(pattern=r"^/topics$"))
        self.bot.add_event_handler(self._on_backup, events.NewMessage(pattern=r"^/backup$"))
        self.bot.add_event_handler(self._on_ask, events.NewMessage(pattern=r"^/ask(?:\s+(.+))?$"))
        self.bot.add_event_handler(self._on_groups, events.NewMessage(pattern=r"^/groups$"))
        self.bot.add_event_handler(self._on_recent, events.NewMessage(pattern=r"^/recent(?:\s+(.+))?$"))
        self.bot.add_event_handler(
            self._on_group_callback,
            events.CallbackQuery(
                pattern=rb"^(?:(?:g|gp|ga|gx|gd|r|rp|c|cp|cr|cra|k|kp|kc|km|kt|ke|kr|kd|kh|ko|kf|ka|kb|ks|ki|k7|kx|tp|td|ti|af|ro)(?::|$)|(?:s|st|sd)$)"
            ),
        )
        self.bot.add_event_handler(self._on_private_text, events.NewMessage(incoming=True))

    async def _on_new_message(self, event: Any) -> None:
        source = self.sources.get(event.chat_id)
        if source is not None:
            await self._store_telegram_message(source, event.message)
        await self._observe_checkin_message(event)
        await self._suggest_checkin_from_message(event)
        await self._schedule_alert_analysis(event)

    async def _on_edited_message(self, event: Any) -> None:
        source = self.sources.get(event.chat_id)
        if source is not None:
            await self._store_telegram_message(source, event.message)

    async def _on_deleted_message(self, event: Any) -> None:
        if event.chat_id in self.sources:
            self.archive.delete(event.chat_id, event.deleted_ids)

    async def _store_telegram_message(self, source: SourceChat, message: Any) -> bool:
        text = message.raw_text or ""
        if not text:
            return False
        sender = message.sender
        if sender is None:
            try:
                sender = await message.get_sender()
            except Exception:
                sender = None
        return self.archive.upsert(
            StoredMessage(
                chat_id=source.chat_id,
                message_id=int(message.id),
                chat_name=source.name,
                chat_username=source.username,
                sender_id=getattr(sender, "id", None),
                sender_name=_sender_name(sender),
                sent_at=message.date,
                text=text,
                reply_to_id=getattr(message, "reply_to_msg_id", None),
            )
        )

    async def _backfill(self) -> None:
        await self._backfill_sources(tuple(self.sources.values()))

    async def _backfill_sources(self, sources: Sequence[SourceChat]) -> None:
        if self.settings.backfill_days <= 0:
            return
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(
            days=self.settings.backfill_days
        )
        for source in sources:
            count = 0
            async for message in self.user.iter_messages(
                source.entity, limit=self.settings.backfill_max_messages
            ):
                if message.date < cutoff:
                    break
                if await self._store_telegram_message(source, message):
                    count += 1
            log.info("Backfilled %d messages from %s", count, source.name)

    async def _on_help(self, event: Any) -> None:
        if not await self._private_authorized(event):
            return
        await event.reply(
            "可用命令：\n"
            "/groups - 分页多选要归档的已加入群组\n"
            "/recent [数量] - 选择群组并读取最近消息\n"
            "/ask <问题> - 查询已归档的群聊历史\n"
            "/summary - 立即生成过去 24 小时摘要\n"
            "/content - 识别内容类型并排除成人群摘要\n"
            "/checkin - 管理自动签到\n"
            "/alerts - 配置重大事件提醒\n"
            "/topics - 管理关键词话题订阅\n"
            "/refresh - 重新扫描群组和机器人\n"
            "/backup - 导出不含凭据的消息数据库\n"
            "/settings - 设置每日推送时间和开关\n"
            "/status - 查看归档和定时任务状态\n\n"
            "也可以直接私聊发送问题。群组选择和最近消息仅在私聊中可用。"
        )

    async def _on_status(self, event: Any) -> None:
        if not await self._private_authorized(event):
            return
        count = self.archive.count(tuple(self.sources))
        overrides, classifications = self._content_state()
        adult_count = sum(
            _summary_excluded(source.chat_id, overrides, classifications)
            for source in self.sources.values()
        )
        hour, minute = self._digest_schedule()
        checkins = self._checkin_configs()
        enabled_checkins = sum(config.enabled for config in checkins.values())
        alert_enabled, _ = self._alert_config()
        schedule = f"{hour:02d}:{minute:02d}"
        names = (
            "\n".join(f"- {source.name}" for source in self.sources.values())
            or "- 暂未选择"
        )
        text = (
            f"已归档消息：{count}\n来源群组（{len(self.sources)}）：\n{names}\n"
            f"可选择群组：{len(self.available_sources)}\n"
            f"后台回填任务：{len(self._backfill_tasks)} 个\n"
            f"摘要排除的成人群：{adult_count} 个\n"
            f"每日推送：{'开启' if self._digest_enabled() else '已关闭'}\n"
            f"每日摘要：{schedule}（{self.settings.timezone}）\n"
            f"自动签到：{enabled_checkins}/{len(checkins)} 个群已开启\n"
            f"重大事件提醒：{'开启' if alert_enabled else '关闭'}\n"
            f"话题订阅：{len(self._topic_subscriptions())} 个\n"
            f"AI 模型：{self.settings.llm_model}"
            + (
                f"（备用：{self.settings.llm_fallback_model}）"
                if self.settings.llm_fallback_model
                else ""
            )
        )
        await _send_long(event, text)

    async def _on_summary(self, event: Any) -> None:
        if not await self._private_authorized(event):
            return
        if not self.sources:
            await event.reply("暂无选择的归档群，使用 /groups 添加。")
            return
        await event.reply("正在生成过去 24 小时摘要...")
        await self._send_digest(event.chat_id)

    async def _on_settings(self, event: Any) -> None:
        if not await self._private_authorized(event):
            return
        self._pending_schedule_users.discard(event.sender_id)
        text, buttons = self._settings_picker()
        await event.reply(text, buttons=buttons)

    async def _on_content(self, event: Any) -> None:
        if not await self._private_authorized(event):
            return
        await self._send_group_picker(event, mode="content", page=0)

    async def _on_checkin(self, event: Any) -> None:
        if not await self._private_authorized(event):
            return
        self._pending_checkin_text_users.pop(event.sender_id, None)
        self._pending_checkin_schedule_users.pop(event.sender_id, None)
        text, buttons = self._checkin_picker()
        await event.reply(text, buttons=buttons)

    async def _on_refresh(self, event: Any) -> None:
        if not await self._private_authorized(event):
            return
        await event.reply("正在重新扫描 Telegram 对话列表...")
        count = await self._refresh_dialogs()
        await event.reply(f"扫描完成：发现 {count} 个可签到群组或机器人。")

    async def _on_alerts(self, event: Any) -> None:
        if not await self._private_authorized(event):
            return
        text, buttons = self._alerts_picker()
        await event.reply(text, buttons=buttons)

    async def _on_topics(self, event: Any) -> None:
        if not await self._private_authorized(event):
            return
        text, buttons = self._topics_picker()
        await event.reply(text, buttons=buttons)

    async def _on_backup(self, event: Any) -> None:
        if not await self._private_authorized(event):
            return
        await event.reply("正在生成一致性数据库备份，请稍候...")
        try:
            path = await asyncio.to_thread(self._create_backup)
            await self.bot.send_file(
                event.chat_id,
                path,
                caption="消息数据库备份。此文件不包含 .env、Telegram 会话或 API 密钥。",
            )
            self._prune_backups()
        except Exception:
            log.exception("Database backup failed")
            await event.reply("备份失败，请检查服务日志。")

    def _settings_picker(self) -> tuple[str, list[list[Any]]]:
        hour, minute = self._digest_schedule()
        enabled = self._digest_enabled()
        text = (
            "每日摘要设置\n"
            f"时间：{hour:02d}:{minute:02d}（UTC+8）\n"
            f"推送：{'开启' if enabled else '已关闭'}\n"
            "推送对象：仅当前配置的本人私聊"
        )
        buttons = [
            [Button.inline("更改推送时间", data=b"st")],
            [Button.inline("关闭每日推送" if enabled else "开启每日推送", data=b"sd")],
        ]
        return text, buttons

    def _alerts_picker(self) -> tuple[str, list[list[Any]]]:
        enabled, keywords = self._alert_config()
        text = (
            "重大事件提醒\n"
            f"状态：{'开启' if enabled else '关闭'}\n"
            f"关键词：{', '.join(keywords)}\n"
            "命中关键词会立即提醒；同时对少量事件线索使用 AI 复核，避免普通聊天打扰。"
        )
        return text, [
            [Button.inline("关闭提醒" if enabled else "开启提醒", data=b"ka")],
            [Button.inline("修改关键词", data=b"kb")],
        ]

    def _topic_subscriptions(self) -> list[dict[str, Any]]:
        archive = getattr(self, "archive", None)
        if archive is None or not hasattr(archive, "get_state"):
            return []
        try:
            raw = json.loads(archive.get_state(TOPIC_SUBSCRIPTIONS_STATE) or "[]")
        except json.JSONDecodeError:
            return []
        if not isinstance(raw, list):
            return []
        result: list[dict[str, Any]] = []
        for item in raw[:MAX_TOPIC_SUBSCRIPTIONS]:
            if not isinstance(item, dict):
                continue
            try:
                chat_id = int(item["chat_id"])
            except (KeyError, TypeError, ValueError):
                continue
            keyword = " ".join(str(item.get("keyword", "")).split())[:80]
            if keyword:
                result.append({"chat_id": chat_id, "keyword": keyword, "last_sent": str(item.get("last_sent", ""))})
        return result

    def _save_topic_subscriptions(self, values: Sequence[dict[str, Any]]) -> None:
        self.archive.set_state(TOPIC_SUBSCRIPTIONS_STATE, json.dumps(list(values)[:MAX_TOPIC_SUBSCRIPTIONS], ensure_ascii=False))

    def _topics_picker(self) -> tuple[str, list[list[Any]]]:
        subscriptions = self._topic_subscriptions()
        lines = ["话题订阅", "命中关键词后由 AI 判断是否为新进展，并私聊提醒。"]
        buttons: list[list[Any]] = [[Button.inline("添加订阅", data=b"tp:0")]]
        for index, item in enumerate(subscriptions):
            source = self.available_sources.get(int(item["chat_id"]))
            name = source.name if source else str(item["chat_id"])
            lines.append(f"- {name}：{item['keyword']}")
            buttons.append([Button.inline("移除：" + _short_name(f"{name} {item['keyword']}", 28), data=f"td:{index}".encode())])
        if len(subscriptions) >= MAX_TOPIC_SUBSCRIPTIONS:
            lines.append(f"已达到 {MAX_TOPIC_SUBSCRIPTIONS} 个订阅上限。")
        return "\n".join(lines), buttons

    def _topic_target_picker(self, page: int) -> tuple[str, list[list[Any]]]:
        groups = sorted(self.available_sources.values(), key=lambda source: source.name.casefold())
        pages = max(1, (len(groups) + GROUPS_PAGE_SIZE - 1) // GROUPS_PAGE_SIZE)
        page = max(0, min(page, pages - 1))
        current = groups[page * GROUPS_PAGE_SIZE : (page + 1) * GROUPS_PAGE_SIZE]
        buttons = [[Button.inline(_short_name(source.name, 34), data=f"ti:{page}:{source.chat_id}".encode())] for source in current]
        navigation: list[Any] = []
        if page > 0:
            navigation.append(Button.inline("上一页", data=f"tp:{page - 1}".encode()))
        if page + 1 < pages:
            navigation.append(Button.inline("下一页", data=f"tp:{page + 1}".encode()))
        if navigation:
            buttons.append(navigation)
        buttons.append([Button.inline("返回订阅", data=b"tp:back")])
        return f"选择要订阅的群组（第 {page + 1}/{pages} 页）：", buttons

    def _checkin_picker(self) -> tuple[str, list[list[Any]]]:
        configs = self._checkin_configs()
        if not configs:
            text = (
                "自动签到\n"
                "尚未配置签到目标。签到会使用你的 Telegram 个人账号，按设定时间发送文本。"
            )
        else:
            lines = ["自动签到"]
            for chat_id, config in sorted(configs.items(), key=lambda item: self._checkin_name(item[0])):
                state = "开启" if config.enabled else "关闭"
                last = config.last_success_day or "尚未签到"
                lines.append(
                    f"- {self._checkin_label(chat_id)}：{state}，"
                    f"{config.hour:02d}:{config.minute:02d}+随机偏移，上次 {last}"
                    + (f"，连续失败 {config.failure_streak} 天" if config.failure_streak else "")
                )
            text = "\n".join(lines)
        buttons: list[list[Any]] = [
            [Button.inline("添加签到目标", data=b"kp:0")],
            [Button.inline("刷新群组和机器人", data=b"kf")],
        ]
        for chat_id in sorted(configs):
            buttons.append(
                [
                    Button.inline(
                        "配置：" + _short_name(self._checkin_label(chat_id), 30),
                        data=f"kc:{chat_id}".encode(),
                    )
                ]
            )
        return text, buttons

    def _checkin_name(self, chat_id: int) -> str:
        target = self.available_checkin_targets.get(chat_id)
        return target.name if target is not None else f"不可用目标 ({chat_id})"

    def _checkin_label(self, chat_id: int) -> str:
        target = self.available_checkin_targets.get(chat_id)
        if target is None:
            return self._checkin_name(chat_id)
        prefix = "机器人" if target.target_kind == "bot" else "群组"
        return f"{prefix}：{target.name}"

    def _checkin_target_picker(self, page: int) -> tuple[str, list[list[Any]] | None]:
        targets = sorted(
            self.available_checkin_targets.values(),
            key=lambda target: (target.target_kind, target.name.casefold()),
        )
        if not targets:
            return "未发现可选择的群组或机器人。", None
        pages = max(1, (len(targets) + GROUPS_PAGE_SIZE - 1) // GROUPS_PAGE_SIZE)
        page = max(0, min(page, pages - 1))
        configs = self._checkin_configs()
        buttons: list[list[Any]] = []
        for target in targets[page * GROUPS_PAGE_SIZE : (page + 1) * GROUPS_PAGE_SIZE]:
            action = "配置" if target.chat_id in configs else "添加"
            buttons.append(
                [
                    Button.inline(
                        f"{action}：" + _short_name(self._checkin_label(target.chat_id), 34),
                        data=f"k:{page}:{target.chat_id}".encode(),
                    )
                ]
            )
        navigation: list[Any] = []
        if page:
            navigation.append(Button.inline("上一页", data=f"kp:{page - 1}".encode()))
        if page + 1 < pages:
            navigation.append(Button.inline("下一页", data=f"kp:{page + 1}".encode()))
        if navigation:
            buttons.append(navigation)
        return f"选择自动签到目标（群组或机器人，第 {page + 1}/{pages} 页）：", buttons

    def _checkin_config_picker(self, chat_id: int) -> tuple[str, list[list[Any]]]:
        config = self._checkin_configs().get(chat_id)
        if config is None:
            return "该签到目标不存在。", [[Button.inline("返回", data=b"kc:0")]]
        state = "开启" if config.enabled else "关闭"
        last = config.last_success_day or "尚未签到"
        topic = str(config.topic_id) if config.topic_id else "主聊天"
        text = (
            f"自动签到：{self._checkin_label(chat_id)}\n"
            f"状态：{state}\n"
            f"时间：{config.hour:02d}:{config.minute:02d}（UTC+8，随机延后 300-800ms）\n"
            f"Topic：{topic}\n"
            f"文本：{config.text}\n"
            f"上次状态：{config.last_status or '尚未执行'}，{last}\n"
            f"详情：{config.last_detail or '无'}\n"
            "立即签到会计入今天，避免定时任务重复发送。"
        )
        return text, [
            [Button.inline("更改签到文本", data=f"km:{chat_id}".encode())],
            [Button.inline("更改签到时间", data=f"kt:{chat_id}".encode())],
            [Button.inline("设置 Topic", data=f"ko:{chat_id}".encode())],
            [
                Button.inline(
                    "关闭自动签到" if config.enabled else "开启自动签到",
                    data=f"ke:{chat_id}".encode(),
                )
            ],
            [Button.inline("立即签到", data=f"kr:{chat_id}".encode())],
            [Button.inline("移除此群", data=f"kd:{chat_id}".encode())],
            [Button.inline("查看签到记录", data=f"kh:{chat_id}".encode())],
            [Button.inline("返回列表", data=b"kc:0")],
        ]

    def _checkin_history_picker(self, chat_id: int) -> tuple[str, list[list[Any]]]:
        config = self._checkin_configs().get(chat_id)
        if config is None:
            return "该签到目标不存在。", [[Button.inline("返回", data=b"kc:0")]]
        lines = [f"签到记录：{self._checkin_label(chat_id)}"]
        if not config.history:
            lines.append("暂无记录。")
        else:
            for record in reversed(config.history[-14:]):
                detail = f"：{record.detail}" if record.detail else ""
                lines.append(f"{record.day} {record.status}{detail}")
        return "\n".join(lines), [[Button.inline("返回配置", data=f"kc:{chat_id}".encode())]]

    async def _on_ask(self, event: Any) -> None:
        if not await self._private_authorized(event):
            return
        match = event.pattern_match
        question = match.group(1).strip() if match and match.group(1) else ""
        if not question:
            await event.reply("用法：/ask <问题>")
            return
        await self._answer_question(event, question)

    async def _on_groups(self, event: Any) -> None:
        if not await self._private_authorized(event):
            return
        await self._send_group_picker(event, mode="groups", page=0)

    async def _on_recent(self, event: Any) -> None:
        if not await self._private_authorized(event):
            return
        raw_count = event.pattern_match.group(1) if event.pattern_match else None
        count = _recent_count(raw_count)
        if count is None:
            await event.reply(f"用法：/recent [1-{MAX_RECENT_MESSAGES}]")
            return
        await self._send_group_picker(event, mode="recent", page=0, count=count)

    async def _send_group_picker(
        self, event: Any, mode: str, page: int, count: int = DEFAULT_RECENT_MESSAGES
    ) -> None:
        text, buttons = self._group_picker(mode, page, count)
        await event.reply(text, buttons=buttons)

    def _group_picker(
        self, mode: str, page: int, count: int
    ) -> tuple[str, list[list[Any]] | None]:
        candidates = self.sources.values() if mode == "content" else self.available_sources.values()
        groups = sorted(candidates, key=lambda source: source.name.casefold())
        if not groups:
            return (
                "暂无可管理的归档群。请先使用 /groups 选择要归档的群组。"
                if mode == "content"
                else "未发现可选择的群组。确认已授权的 Telegram 账号已加入目标群，然后稍后重启服务。",
                None,
            )

        pages = max(1, (len(groups) + GROUPS_PAGE_SIZE - 1) // GROUPS_PAGE_SIZE)
        page = max(0, min(page, pages - 1))
        start = page * GROUPS_PAGE_SIZE
        buttons: list[list[Any]] = []
        overrides, classifications = self._content_state()
        for source in groups[start : start + GROUPS_PAGE_SIZE]:
            if mode == "groups":
                selected = source.chat_id in self.sources
                label = ("取消归档：" if selected else "归档：") + _short_name(source.name)
                data = f"g:{page}:{source.chat_id}".encode()
            elif mode == "recent":
                label = "读取：" + _short_name(source.name)
                data = f"r:{count}:{page}:{source.chat_id}".encode()
            else:
                status = _content_status(source.chat_id, overrides, classifications)
                label = f"{CONTENT_STATUS_LABELS[status]}：" + _short_name(source.name, 32)
                data = f"c:{page}:{source.chat_id}".encode()
            buttons.append([Button.inline(label, data=data)])

        navigation: list[Any] = []
        if page:
            prefix = {"groups": "gp", "recent": "rp", "content": "cp"}[mode]
            data = (
                f"{prefix}:{page - 1}"
                if mode != "recent"
                else f"{prefix}:{count}:{page - 1}"
            )
            navigation.append(Button.inline("上一页", data=data.encode()))
        if page + 1 < pages:
            prefix = {"groups": "gp", "recent": "rp", "content": "cp"}[mode]
            data = (
                f"{prefix}:{page + 1}"
                if mode != "recent"
                else f"{prefix}:{count}:{page + 1}"
            )
            navigation.append(Button.inline("下一页", data=data.encode()))
        if navigation:
            buttons.append(navigation)

        if mode == "groups":
            buttons.append(
                [
                    Button.inline("全选本页", data=f"ga:{page}".encode()),
                    Button.inline("清除本页", data=f"gx:{page}".encode()),
                ]
            )
            buttons.append([Button.inline("完成", data=f"gd:{page}".encode())])
            text = f"选择归档群组（已选 {len(self.sources)} 个，第 {page + 1}/{pages} 页）："
        elif mode == "recent":
            text = f"选择要读取最近 {count} 条消息的群组（第 {page + 1}/{pages} 页）："
        else:
            text = (
                f"内容分类（成人群不参与摘要，第 {page + 1}/{pages} 页）：\n"
                "点击群组可切换“排除成人摘要”与“参与摘要”；自动识别仅依据近期文字。"
            )
            buttons.append(
                [
                    Button.inline("重新分析本页", data=f"cr:{page}".encode()),
                    Button.inline("重新分析所有群", data=b"cra"),
                ]
            )
        return text, buttons

    async def _set_group_page(self, page: int, selected: bool) -> tuple[int, int, bool]:
        groups = sorted(
            self.available_sources.values(), key=lambda source: source.name.casefold()
        )
        pages = max(1, (len(groups) + GROUPS_PAGE_SIZE - 1) // GROUPS_PAGE_SIZE)
        page = max(0, min(page, pages - 1))
        page_groups = groups[page * GROUPS_PAGE_SIZE : (page + 1) * GROUPS_PAGE_SIZE]
        added: list[SourceChat] = []
        removed: list[int] = []
        limited = False
        async with self._source_lock:
            sources = dict(self.sources)
            if selected:
                for source in page_groups:
                    if source.chat_id in sources or source.chat_id in self._configured_source_ids:
                        continue
                    if len(sources) >= MAX_SELECTED_SOURCES:
                        limited = True
                        break
                    sources[source.chat_id] = source
                    added.append(source)
            else:
                for source in page_groups:
                    if source.chat_id in self._configured_source_ids:
                        continue
                    if source.chat_id in sources:
                        del sources[source.chat_id]
                        removed.append(source.chat_id)
            self.sources = {chat_id: sources[chat_id] for chat_id in sorted(sources)}
            self._save_selected_sources()
        self._schedule_backfill(added)
        return len(added), len(removed), limited

    async def _on_group_callback(self, event: Any) -> None:
        if event.sender_id not in self.settings.allowed_user_ids:
            await event.answer("无权访问。", alert=True)
            return
        if not getattr(event, "is_private", False):
            await event.answer("请私聊机器人使用。", alert=True)
            return

        try:
            parts = event.data.decode().split(":")
            action = parts[0]
            if action == "s" and len(parts) == 1:
                text, buttons = self._settings_picker()
                await event.edit(text, buttons=buttons)
                return
            if action == "st" and len(parts) == 1:
                self._pending_schedule_users.add(event.sender_id)
                await event.answer()
                await event.edit(
                    "请直接发送推送时间，格式为 HH:MM，例如 21:30。\n"
                    "时区固定为 UTC+8。"
                )
                return
            if action == "sd" and len(parts) == 1:
                enabled = not self._digest_enabled()
                self.archive.set_state("digest_enabled", "1" if enabled else "0")
                await event.answer("每日推送已开启。" if enabled else "每日推送已关闭。")
                text, buttons = self._settings_picker()
                await event.edit(text, buttons=buttons)
                return
            if action == "kf" and len(parts) == 1:
                await event.answer("正在重新扫描 Telegram 对话...")
                count = await self._refresh_dialogs()
                await event.edit(f"扫描完成：发现 {count} 个可签到群组或机器人。", buttons=[[Button.inline("打开签到", data=b"kc:0")]])
                return
            if action == "ka" and len(parts) == 1:
                enabled, keywords = self._alert_config()
                self.archive.set_state(
                    CHECKIN_ALERT_STATE,
                    json.dumps({"enabled": not enabled, "keywords": list(keywords)}, ensure_ascii=False),
                )
                await event.answer("重大事件提醒已开启。" if not enabled else "重大事件提醒已关闭。")
                text, buttons = self._alerts_picker()
                await event.edit(text, buttons=buttons)
                return
            if action == "kb" and len(parts) == 1:
                self._pending_alert_keywords_users.add(event.sender_id)
                await event.answer()
                await event.edit("请发送逗号分隔的关键词，例如：紧急,故障,截止,涨价")
                return
            if action == "tp" and len(parts) == 2:
                if parts[1] == "back":
                    text, buttons = self._topics_picker()
                    await event.edit(text, buttons=buttons)
                    return
                if len(self._topic_subscriptions()) >= MAX_TOPIC_SUBSCRIPTIONS:
                    await event.answer("已达到订阅上限。", alert=True)
                    return
                text, buttons = self._topic_target_picker(int(parts[1]))
                await event.answer()
                await event.edit(text, buttons=buttons)
                return
            if action == "ti" and len(parts) == 3:
                chat_id = int(parts[2])
                if chat_id not in self.available_sources:
                    raise ValueError("unknown topic target")
                self._pending_topic_users[event.sender_id] = chat_id
                await event.answer()
                await event.edit("请发送要关注的关键词，例如：固件发布、漏洞、价格变化")
                return
            if action == "td" and len(parts) == 2:
                index = int(parts[1])
                subscriptions = self._topic_subscriptions()
                if not 0 <= index < len(subscriptions):
                    raise ValueError("unknown topic subscription")
                subscriptions.pop(index)
                self._save_topic_subscriptions(subscriptions)
                await event.answer("订阅已移除。")
                text, buttons = self._topics_picker()
                await event.edit(text, buttons=buttons)
                return
            if action == "af" and len(parts) == 3:
                feedback, encoded = parts[1], parts[2]
                chat_id, topic = _decode_alert_feedback(encoded)
                self._save_alert_feedback(chat_id, topic, feedback)
                await event.answer("反馈已保存。")
                await event.edit("已记录你的反馈，后续会减少相同主题的重复提醒。")
                return
            if action in {"ks", "ki", "k7", "kx"} and len(parts) == 2:
                chat_id = int(parts[1])
                source = self.available_sources.get(chat_id)
                if source is None:
                    raise ValueError("unknown suggestion target")
                self._checkin_suggestion_days = {
                    item for item in self._checkin_suggestion_days if item[0] != chat_id
                }
                if action in {"ki", "k7", "kx"}:
                    until = {
                        "ki": dt.datetime.now(ZoneInfo(self.settings.timezone)).date(),
                        "k7": dt.datetime.now(ZoneInfo(self.settings.timezone)).date()
                        + dt.timedelta(days=6),
                    }.get(action)
                    self._set_checkin_suggestion_ignore(chat_id, until=until, permanent=action == "kx")
                    label = {"ki": "今天", "k7": "未来 7 天", "kx": "永久"}[action]
                    await event.answer(f"已忽略：{label}不再提示。")
                    await event.edit(f"已忽略“{source.name}”的签到建议：{label}不再提示。")
                    return
                configs = self._checkin_configs()
                config = configs.get(chat_id, CheckinConfig())
                suggestion = _checkin_suggestion_text(
                    self.archive.get_state(CHECKIN_SUGGESTION_STATE), chat_id
                )
                if suggestion:
                    config = _replace_checkin(config, text=suggestion)
                configs[chat_id] = config
                self._save_checkin_configs(configs)
                await event.answer("已同意，自动签到已部署。")
                await event.edit(
                    f"已为“{source.name}”部署自动签到：每天 {config.hour:02d}:{config.minute:02d}，"
                    f"随机延后 {CHECKIN_OFFSET_MIN_MS}-{CHECKIN_OFFSET_MAX_MS}ms。\n"
                    f"文本：{config.text}"
                )
                return
            if action == "kp" and len(parts) == 2:
                text, buttons = self._checkin_target_picker(int(parts[1]))
                await event.edit(text, buttons=buttons)
                return
            if action == "kc" and len(parts) == 2:
                chat_id = int(parts[1])
                if chat_id == 0:
                    text, buttons = self._checkin_picker()
                else:
                    text, buttons = self._checkin_config_picker(chat_id)
                await event.edit(text, buttons=buttons)
                return
            if action == "kh" and len(parts) == 2:
                chat_id = int(parts[1])
                await event.answer()
                text, buttons = self._checkin_history_picker(chat_id)
                await event.edit(text, buttons=buttons)
                return
            if action == "ko" and len(parts) == 2:
                chat_id = int(parts[1])
                if chat_id not in self._checkin_configs():
                    raise ValueError("unknown check-in target")
                self._pending_checkin_topic_users[event.sender_id] = chat_id
                await event.answer()
                await event.edit("请发送 Topic 根消息 ID；发送 0 表示群组主聊天。机器人目标请发送 0。")
                return
            if action == "k" and len(parts) == 3:
                page, chat_id = int(parts[1]), int(parts[2])
                if chat_id not in self.available_checkin_targets:
                    raise ValueError("unknown check-in target")
                await self._ensure_checkin(chat_id)
                text, buttons = self._checkin_config_picker(chat_id)
                await event.answer("签到目标已添加。")
                await event.edit(text, buttons=buttons)
                return
            if action == "km" and len(parts) == 2:
                chat_id = int(parts[1])
                if chat_id not in self._checkin_configs():
                    raise ValueError("unknown check-in target")
                self._pending_checkin_schedule_users.pop(event.sender_id, None)
                self._pending_checkin_text_users[event.sender_id] = chat_id
                await event.answer()
                await event.edit(
                    "请直接发送每天要向该目标发送的签到文本。\n"
                    "可包含 @机器人 和命令；仅支持文字，最多 1000 个字符。"
                )
                return
            if action == "kt" and len(parts) == 2:
                chat_id = int(parts[1])
                if chat_id not in self._checkin_configs():
                    raise ValueError("unknown check-in target")
                self._pending_checkin_text_users.pop(event.sender_id, None)
                self._pending_checkin_schedule_users[event.sender_id] = chat_id
                await event.answer()
                await event.edit("请直接发送签到时间，格式为 HH:MM，例如 08:30。\n时区固定为 UTC+8。")
                return
            if action == "ke" and len(parts) == 2:
                chat_id = int(parts[1])
                configs = self._checkin_configs()
                config = configs.get(chat_id)
                if config is None:
                    raise ValueError("unknown check-in target")
                configs[chat_id] = _replace_checkin(config, enabled=not config.enabled)
                self._save_checkin_configs(configs)
                await event.answer("自动签到已开启。" if not config.enabled else "自动签到已关闭。")
                text, buttons = self._checkin_config_picker(chat_id)
                await event.edit(text, buttons=buttons)
                return
            if action == "kr" and len(parts) == 2:
                chat_id = int(parts[1])
                if chat_id not in self._checkin_configs():
                    raise ValueError("unknown check-in target")
                await event.answer("正在发送签到...")
                try:
                    await self._send_checkin(chat_id, mark_today=True)
                except Exception:
                    log.exception("Manual check-in failed for chat id=%s", chat_id)
                    await event.edit("立即签到失败，请稍后重试。")
                    return
                text, buttons = self._checkin_config_picker(chat_id)
                await event.edit("立即签到已发送。\n\n" + text, buttons=buttons)
                return
            if action == "kd" and len(parts) == 2:
                chat_id = int(parts[1])
                configs = self._checkin_configs()
                if chat_id not in configs:
                    raise ValueError("unknown check-in target")
                del configs[chat_id]
                self._save_checkin_configs(configs)
                await event.answer("签到目标已移除。")
                text, buttons = self._checkin_picker()
                await event.edit(text, buttons=buttons)
                return
            if action == "gp" and len(parts) == 2:
                await self._edit_group_picker(event, "groups", int(parts[1]))
                return
            if action == "rp" and len(parts) == 3:
                await self._edit_group_picker(event, "recent", int(parts[2]), int(parts[1]))
                return
            if action == "cp" and len(parts) == 2:
                await self._edit_group_picker(event, "content", int(parts[1]))
                return
            if action == "cr" and len(parts) == 2:
                page = int(parts[1])
                self._schedule_classification(self._content_page_sources(page), force=True)
                await event.answer("正在重新分析本页群组。")
                await event.edit("本页群组正在后台重新分析。稍后发送 /content 查看结果。")
                return
            if action == "cra" and len(parts) == 1:
                count = len(self.sources)
                self._schedule_classification(tuple(self.sources.values()), force=True)
                await event.answer(f"正在重新分析全部 {count} 个归档群。")
                await event.edit("所有归档群正在后台重新分析。稍后发送 /content 查看结果。")
                return
            if action in {"ga", "gx"} and len(parts) == 2:
                added, removed, limited = await self._set_group_page(
                    int(parts[1]), selected=action == "ga"
                )
                if limited:
                    message = f"已加入 {added} 个；已达到最多 {MAX_SELECTED_SOURCES} 个群组的上限。"
                elif action == "ga":
                    message = f"本页已加入 {added} 个群组。"
                else:
                    message = f"本页已清除 {removed} 个群组。"
                await event.answer(message)
                await self._edit_group_picker(event, "groups", int(parts[1]))
                return
            if action == "gd" and len(parts) == 2:
                await event.answer("选择已保存，后台回填会继续进行。")
                await event.edit(
                    f"已保存 {len(self.sources)} 个归档群组。\n"
                    "可继续使用 /groups 修改选择。"
                )
                return
            if action == "g" and len(parts) == 3:
                page, chat_id = int(parts[1]), int(parts[2])
                source = self.available_sources.get(chat_id)
                if source is None:
                    raise ValueError("unknown source")
                result = await self._toggle_source(source)
                messages = {
                    "added": "已加入归档，后台开始回填。",
                    "removed": "已取消归档。",
                    "fixed": "此群由配置固定，不能在机器人中移除。",
                    "limit": f"最多可选择 {MAX_SELECTED_SOURCES} 个群组。",
                }
                await event.answer(messages[result])
                await self._edit_group_picker(event, "groups", page)
                return
            if action == "r" and len(parts) == 4:
                count, page, chat_id = (int(parts[1]), int(parts[2]), int(parts[3]))
                source = self.available_sources.get(chat_id)
                if source is None or not 1 <= count <= MAX_RECENT_MESSAGES:
                    raise ValueError("invalid source request")
                await event.answer()
                await event.edit(f"正在读取 {source.name} 的最近消息...")
                text = await self._recent_messages(source, count)
                await _send_long(self.bot, text, target=event.chat_id)
                await event.edit("最近消息已发送，可继续用 /recent 选择其他群组。")
                return
            if action == "c" and len(parts) == 3:
                page, chat_id = int(parts[1]), int(parts[2])
                source = self.sources.get(chat_id)
                if source is None:
                    raise ValueError("unknown source")
                overrides, classifications = self._content_state()
                excluded = _summary_excluded(chat_id, overrides, classifications)
                await self._set_content_override(chat_id, "include" if excluded else "adult")
                await event.answer("该群将参与摘要。" if excluded else "已标记成人内容，摘要将排除该群。")
                await self._edit_group_picker(event, "content", page)
                return
        except (UnicodeDecodeError, ValueError, IndexError):
            await event.answer("操作无效，请重新发送命令。", alert=True)
            return
        except Exception:
            log.exception("Group picker action failed")
            await event.answer("操作失败，请检查服务日志。", alert=True)

    async def _edit_group_picker(
        self, event: Any, mode: str, page: int, count: int = DEFAULT_RECENT_MESSAGES
    ) -> None:
        text, buttons = self._group_picker(mode, page, count)
        await event.edit(text, buttons=buttons)

    def _content_page_sources(self, page: int) -> tuple[SourceChat, ...]:
        groups = sorted(self.sources.values(), key=lambda source: source.name.casefold())
        if not groups:
            return ()
        pages = max(1, (len(groups) + GROUPS_PAGE_SIZE - 1) // GROUPS_PAGE_SIZE)
        page = max(0, min(page, pages - 1))
        return tuple(groups[page * GROUPS_PAGE_SIZE : (page + 1) * GROUPS_PAGE_SIZE])

    async def _recent_messages(self, source: SourceChat, count: int) -> str:
        records: list[tuple[dt.datetime, str, str]] = []
        async for message in self.user.iter_messages(source.entity, limit=count):
            text = (message.raw_text or "").strip()
            if not text:
                continue
            sender = message.sender
            if sender is None:
                try:
                    sender = await message.get_sender()
                except Exception:
                    sender = None
            records.append((message.date, _sender_name(sender), text[:700]))
        return _render_recent_messages(source.name, list(reversed(records)))

    async def _on_private_text(self, event: Any) -> None:
        if not event.is_private:
            return
        if not await self._authorized(event, reply_denied=False):
            return
        if event.sender_id in self._pending_alert_keywords_users:
            keywords = _parse_alert_keywords(event.raw_text)
            if not keywords:
                await event.reply("关键词无效，请发送逗号分隔的文字关键词。")
                return
            enabled, _ = self._alert_config()
            self.archive.set_state(
                CHECKIN_ALERT_STATE,
                json.dumps({"enabled": enabled, "keywords": list(keywords)}, ensure_ascii=False),
            )
            self._pending_alert_keywords_users.discard(event.sender_id)
            await event.reply("重大事件提醒关键词已保存。")
            return
        chat_id = self._pending_topic_users.get(event.sender_id)
        if chat_id is not None:
            keyword = " ".join(event.raw_text.split())[:80]
            if not keyword:
                await event.reply("关键词不能为空。")
                return
            subscriptions = self._topic_subscriptions()
            subscriptions = [item for item in subscriptions if not (item["chat_id"] == chat_id and item["keyword"].casefold() == keyword.casefold())]
            subscriptions.append({"chat_id": chat_id, "keyword": keyword, "last_sent": ""})
            self._save_topic_subscriptions(subscriptions)
            self._pending_topic_users.pop(event.sender_id, None)
            await event.reply("话题订阅已保存。发送 /topics 可管理。")
            return
        chat_id = self._pending_checkin_topic_users.get(event.sender_id)
        if chat_id is not None:
            raw_topic = event.raw_text.strip()
            try:
                topic_id = int(raw_topic)
            except ValueError:
                topic_id = -1
            target = self.available_checkin_targets.get(chat_id)
            if topic_id < 0 or (target is not None and target.target_kind == "bot" and topic_id != 0):
                await event.reply("Topic ID 无效；请发送非负整数，机器人目标只能发送 0。")
                return
            configs = self._checkin_configs()
            config = configs.get(chat_id)
            if config is None:
                self._pending_checkin_topic_users.pop(event.sender_id, None)
                await event.reply("签到目标已不存在，请重新发送 /checkin。")
                return
            configs[chat_id] = _replace_checkin(config, topic_id=topic_id or None)
            self._save_checkin_configs(configs)
            self._pending_checkin_topic_users.pop(event.sender_id, None)
            await event.reply("Topic 已保存。发送 0 可恢复主聊天。")
            return
        chat_id = self._pending_checkin_text_users.get(event.sender_id)
        if chat_id is not None:
            text = event.raw_text.strip()
            if not 1 <= len(text) <= MAX_CHECKIN_TEXT_LENGTH:
                await event.reply(
                    f"签到文本需要为 1-{MAX_CHECKIN_TEXT_LENGTH} 个字符，请重新发送。"
                )
                return
            configs = self._checkin_configs()
            config = configs.get(chat_id)
            if config is None:
                self._pending_checkin_text_users.pop(event.sender_id, None)
                await event.reply("签到目标已不存在，请重新发送 /checkin。")
                return
            configs[chat_id] = _replace_checkin(config, text=text)
            self._save_checkin_configs(configs)
            self._pending_checkin_text_users.pop(event.sender_id, None)
            await event.reply("签到文本已保存。发送 /checkin 可继续管理。")
            return
        chat_id = self._pending_checkin_schedule_users.get(event.sender_id)
        if chat_id is not None:
            schedule = _parse_schedule(event.raw_text)
            if schedule is None:
                await event.reply("时间无效，请发送 HH:MM，例如 08:30。")
                return
            configs = self._checkin_configs()
            config = configs.get(chat_id)
            if config is None:
                self._pending_checkin_schedule_users.pop(event.sender_id, None)
                await event.reply("签到目标已不存在，请重新发送 /checkin。")
                return
            hour, minute = schedule
            configs[chat_id] = _replace_checkin(
                config,
                hour=hour,
                minute=minute,
                attempt_day="",
                attempt_count=0,
                retry_at="",
                scheduled_for="",
            )
            self._save_checkin_configs(configs)
            self._pending_checkin_schedule_users.pop(event.sender_id, None)
            await event.reply(f"签到时间已设为 {hour:02d}:{minute:02d}（UTC+8）。")
            return
        if event.sender_id in self._pending_schedule_users:
            schedule = _parse_schedule(event.raw_text)
            if schedule is None:
                await event.reply("时间无效，请发送 HH:MM，例如 21:30。")
                return
            hour, minute = schedule
            self.archive.set_state("digest_hour", str(hour))
            self.archive.set_state("digest_minute", str(minute))
            self._pending_schedule_users.discard(event.sender_id)
            await event.reply(f"每日推送时间已设为 {hour:02d}:{minute:02d}（UTC+8）。")
            return
        if event.raw_text.startswith("/"):
            return
        await self._answer_question(event, event.raw_text.strip())

    async def _answer_question(self, event: Any, question: str) -> None:
        progress = await event.reply("正在检索历史消息...")
        try:
            terms = await self.llm.query_terms(question)
            messages = self.archive.search(
                tuple(self.sources), terms, self.settings.query_max_sources
            )
            if len(messages) < min(8, self.settings.query_max_sources):
                existing = {(m.chat_id, m.message_id) for m in messages}
                for item in self.archive.recent(
                    tuple(self.sources), self.settings.query_max_sources
                ):
                    if (item.chat_id, item.message_id) not in existing:
                        messages.append(item)
                    if len(messages) >= self.settings.query_max_sources:
                        break
            answer = await self.llm.answer(question, _redact_messages(messages))
            await progress.delete()
            await _send_long(event, render_answer(answer))
        except Exception:
            log.exception("Question answering failed")
            await progress.edit("查询失败，请检查服务日志。")

    async def _authorized(self, event: Any, reply_denied: bool = True) -> bool:
        if event.sender_id in self.settings.allowed_user_ids:
            return True
        log.warning("Denied bot access for user id=%s", event.sender_id)
        if reply_denied:
            await event.reply("无权访问。")
        return False

    async def _private_authorized(self, event: Any) -> bool:
        if not await self._authorized(event):
            return False
        if getattr(event, "is_private", False):
            return True
        await event.reply("请私聊机器人使用此命令。")
        return False

    def _digest_schedule(self) -> tuple[int, int]:
        return (
            _state_int_in_range(
                self.archive.get_state("digest_hour"),
                self.settings.summary_hour,
                0,
                23,
            ),
            _state_int_in_range(
                self.archive.get_state("digest_minute"),
                self.settings.summary_minute,
                0,
                59,
            ),
        )

    def _digest_enabled(self) -> bool:
        return self.archive.get_state("digest_enabled") != "0"

    def _content_state(self) -> tuple[dict[int, str], dict[int, tuple[str, dt.date]]]:
        return (
            _content_overrides_from_state(self.archive.get_state(CONTENT_OVERRIDES_STATE)),
            _content_classifications_from_state(
                self.archive.get_state(CONTENT_CLASSIFICATIONS_STATE)
            ),
        )

    def _checkin_configs(self) -> dict[int, CheckinConfig]:
        return _checkin_configs_from_state(self.archive.get_state(CHECKIN_CONFIGS_STATE))

    def _save_checkin_configs(self, configs: dict[int, CheckinConfig]) -> None:
        self.archive.set_state(
            CHECKIN_CONFIGS_STATE,
            json.dumps(
                {
                    str(chat_id): _checkin_config_to_json(config)
                    for chat_id, config in sorted(configs.items())
                },
                ensure_ascii=False,
            ),
        )

    def _alert_config(self) -> tuple[bool, tuple[str, ...]]:
        try:
            raw = json.loads(self.archive.get_state(CHECKIN_ALERT_STATE) or "{}")
        except json.JSONDecodeError:
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        enabled = bool(raw.get("enabled", True))
        keywords = _parse_alert_keywords(raw.get("keywords", ALERT_KEYWORDS_DEFAULT))
        return enabled, keywords or _parse_alert_keywords(ALERT_KEYWORDS_DEFAULT)

    def _alert_topic_ignored(self, chat_id: int, topic: str) -> bool:
        try:
            raw = json.loads(self.archive.get_state(ALERT_IGNORED_TOPICS_STATE) or "{}")
        except json.JSONDecodeError:
            return False
        if not isinstance(raw, dict):
            return False
        topics = raw.get(str(chat_id), [])
        if not isinstance(topics, list):
            return False
        fingerprint = _alert_topic_fingerprint(topic)
        return fingerprint in {str(item)[:100] for item in topics}

    def _save_alert_feedback(self, chat_id: int, topic: str, feedback: str) -> None:
        fingerprint = _alert_topic_fingerprint(topic)
        if not fingerprint or feedback not in {"important", "ignore"}:
            raise ValueError("invalid alert feedback")
        try:
            feedbacks = json.loads(self.archive.get_state(ALERT_FEEDBACK_STATE) or "{}")
        except json.JSONDecodeError:
            feedbacks = {}
        if not isinstance(feedbacks, dict):
            feedbacks = {}
        feedbacks.setdefault(str(chat_id), {})[fingerprint] = feedback
        self.archive.set_state(ALERT_FEEDBACK_STATE, json.dumps(feedbacks, ensure_ascii=False))
        if feedback != "ignore":
            return
        try:
            ignored = json.loads(self.archive.get_state(ALERT_IGNORED_TOPICS_STATE) or "{}")
        except json.JSONDecodeError:
            ignored = {}
        if not isinstance(ignored, dict):
            ignored = {}
        topics = ignored.setdefault(str(chat_id), [])
        if not isinstance(topics, list):
            topics = []
            ignored[str(chat_id)] = topics
        if fingerprint not in topics:
            topics.append(fingerprint)
        ignored[str(chat_id)] = topics[-50:]
        self.archive.set_state(ALERT_IGNORED_TOPICS_STATE, json.dumps(ignored, ensure_ascii=False))

    async def _refresh_dialogs(self) -> int:
        async with self._refresh_lock:
            await self._discover_source_chats()
            await self._resolve_sources()
            return len(self.available_checkin_targets)

    def _create_backup(self) -> Path:
        backup_dir = self.settings.data_dir / "backups"
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = backup_dir / f"messages-{stamp}.db"
        self.archive.backup_to(path)
        return path

    def _prune_backups(self, keep: int = 7) -> None:
        backup_dir = self.settings.data_dir / "backups"
        paths = sorted(backup_dir.glob("messages-*.db"), key=lambda path: path.stat().st_mtime, reverse=True)
        for path in paths[keep:]:
            try:
                path.unlink()
            except OSError:
                log.warning("Unable to prune backup %s", path)

    async def _ensure_checkin(self, chat_id: int) -> None:
        async with self._checkin_lock:
            configs = self._checkin_configs()
            configs.setdefault(chat_id, CheckinConfig())
            self._save_checkin_configs(configs)

    async def _send_checkin(self, chat_id: int, mark_today: bool) -> None:
        async with self._checkin_lock:
            configs = self._checkin_configs()
            config = configs.get(chat_id)
            target = self.available_checkin_targets.get(chat_id)
            if config is None or target is None:
                raise ValueError("check-in target is unavailable")
            sent_at = dt.datetime.now(ZoneInfo(self.settings.timezone))
            day = sent_at.date().isoformat()
            waiter: asyncio.Future[tuple[str, str]] = asyncio.get_running_loop().create_future()
            self._checkin_waiters[(chat_id, day)] = waiter
            try:
                send_kwargs: dict[str, Any] = {"link_preview": False}
                if target.target_kind == "group" and config.topic_id:
                    send_kwargs["reply_to"] = config.topic_id
                sent_message = await self.user.send_message(target.entity, config.text, **send_kwargs)
            except Exception as exc:
                self._checkin_waiters.pop((chat_id, day), None)
                detail = _safe_error(exc)
                configs[chat_id] = _append_checkin_record(config, "send_failed", detail, sent_at)
                self._save_checkin_configs(configs)
                raise

            try:
                status, detail = await asyncio.wait_for(
                    waiter, timeout=CHECKIN_VERIFY_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                status, detail = "sent_unverified", "已发送，等待窗口内未发现明确成功回复"
            finally:
                self._checkin_waiters.pop((chat_id, day), None)

            latest = self._checkin_configs().get(chat_id, config)
            latest = _append_checkin_record(latest, status, detail, sent_at)
            if mark_today and status in {"verified", "sent_unverified"}:
                latest = _replace_checkin(
                    latest,
                    last_success_day=day,
                    attempt_day=day,
                    attempt_count=0,
                    retry_at="",
                    scheduled_for="",
                )
            self._save_checkin_configs({**self._checkin_configs(), chat_id: latest})
            if status == "failed":
                raise CheckinVerificationError(detail)
            if status == "send_failed":
                raise CheckinVerificationError(detail)

    async def _observe_checkin_message(self, event: Any) -> None:
        # The user's own outbound command is not a verification response. This
        # matters for direct Bot targets where the target kind is otherwise
        # trusted before the Bot has had a chance to reply.
        if getattr(event, "out", False):
            return
        chat_id = getattr(event, "chat_id", None)
        if chat_id is None:
            return
        configs = self._checkin_configs()
        if int(chat_id) not in configs:
            return
        key = (int(chat_id), dt.datetime.now(ZoneInfo(self.settings.timezone)).date().isoformat())
        waiter = self._checkin_waiters.get(key)
        if waiter is None or waiter.done():
            return
        text = (getattr(event, "raw_text", "") or "").strip()
        if not text:
            return
        target = self.available_checkin_targets.get(int(chat_id))
        sender = getattr(getattr(event, "message", None), "sender", None)
        if sender is None:
            sender = getattr(event, "sender", None)
        from_bot = bool(getattr(sender, "bot", False))
        status = _checkin_verification_status(
            text,
            from_bot=from_bot,
            target_kind=target.target_kind if target is not None else None,
        )
        if status is not None:
            waiter.set_result((status, text[:300]))

    async def _suggest_checkin_from_message(self, event: Any) -> None:
        chat_id = getattr(event, "chat_id", None)
        source = self.available_sources.get(chat_id)
        text = (getattr(event, "raw_text", "") or "").strip()
        if source is None or not text or len(text) > MAX_CHECKIN_TEXT_LENGTH:
            return
        sender = getattr(getattr(event, "message", None), "sender", None)
        if sender is None:
            sender = getattr(event, "sender", None)
        if getattr(sender, "bot", False):
            self._record_checkin_suggestion_bot_reply(event, text)
            return
        if getattr(sender, "id", None) == self._user_id:
            return
        if not _is_checkin_suggestion_candidate(text):
            return
        day = dt.datetime.now(ZoneInfo(self.settings.timezone)).date().isoformat()
        key = (int(chat_id), day)
        if (
            key in self._checkin_suggestion_days
            or int(chat_id) in self._checkin_configs()
            or self._checkin_suggestion_ignored(int(chat_id), dt.date.fromisoformat(day))
        ):
            return
        message = getattr(event, "message", None)
        message_id = getattr(message, "id", None)
        if not isinstance(message_id, int):
            return
        candidate_key = (int(chat_id), message_id)
        if candidate_key in self._checkin_suggestion_candidates:
            return
        source_time = _event_message_time(event)
        self._checkin_suggestion_candidates[candidate_key] = CheckinSuggestionCandidate(
            chat_id=int(chat_id),
            message_id=message_id,
            source_name=source.name,
            source_username=source.username,
            source_sender=_sender_name(sender),
            source_time=source_time,
            source_text=text,
        )
        task = asyncio.create_task(
            self._analyze_checkin_suggestion(candidate_key), name="checkin-suggestion"
        )
        self._checkin_suggestion_tasks.add(task)
        task.add_done_callback(self._checkin_suggestion_tasks.discard)

    def _record_checkin_suggestion_bot_reply(self, event: Any, text: str) -> None:
        message = getattr(event, "message", None)
        reply_to = getattr(message, "reply_to_msg_id", None)
        chat_id = getattr(event, "chat_id", None)
        if not isinstance(chat_id, int) or not isinstance(reply_to, int):
            return
        candidate = self._checkin_suggestion_candidates.get((chat_id, reply_to))
        if candidate is not None and not candidate.bot_reply:
            candidate.bot_reply = text[:MAX_CHECKIN_TEXT_LENGTH]

    async def _analyze_checkin_suggestion(self, candidate_key: tuple[int, int]) -> None:
        try:
            await asyncio.sleep(CHECKIN_SUGGESTION_REPLY_WAIT_SECONDS)
            attempt = 0
            while True:
                candidate = self._checkin_suggestion_candidates.get(candidate_key)
                if candidate is None or not candidate.bot_reply:
                    self._checkin_suggestion_candidates.pop(candidate_key, None)
                    return
                if _checkin_suggestion_expired(candidate.source_time):
                    self._checkin_suggestion_candidates.pop(candidate_key, None)
                    log.info("Dropping expired check-in suggestion candidate=%s", candidate_key)
                    return
                day = dt.datetime.now(ZoneInfo(self.settings.timezone)).date().isoformat()
                key = (candidate.chat_id, day)
                if (
                    key in self._checkin_suggestion_days
                    or candidate.chat_id in self._checkin_configs()
                    or self._checkin_suggestion_ignored(candidate.chat_id, dt.date.fromisoformat(day))
                ):
                    self._checkin_suggestion_candidates.pop(candidate_key, None)
                    return

                # The public status probe is only a back-pressure signal. An unknown or
                # stale probe must never discard evidence; the real API remains authoritative.
                if await self._checkin_ai_unavailable():
                    delay = _checkin_suggestion_retry_delay(attempt)
                    attempt += 1
                    log.warning(
                        "Configured AI models are unavailable according to status.input.im; "
                        "retrying check-in candidate=%s in %ss",
                        candidate_key,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                try:
                    decision = await self.llm.assess_checkin_suggestion(
                        candidate.source_name,
                        candidate.source_sender,
                        candidate.source_time,
                        _redact_for_llm(candidate.source_text),
                        _redact_for_llm(candidate.bot_reply),
                    )
                except Exception as exc:
                    if not _is_retryable_checkin_ai_error(exc):
                        self._checkin_suggestion_candidates.pop(candidate_key, None)
                        log.exception("Check-in suggestion analysis failed permanently for candidate=%s", candidate_key)
                        return
                    delay = _checkin_suggestion_retry_delay(attempt)
                    attempt += 1
                    log.warning(
                        "Temporary AI failure for check-in candidate=%s; retrying in %ss: %s",
                        candidate_key,
                        delay,
                        _safe_error(exc),
                    )
                    await asyncio.sleep(delay)
                    continue

                self._checkin_suggestion_candidates.pop(candidate_key, None)
                if not decision.should_suggest or not _is_literal_checkin_proposal(
                    decision.proposed_text, candidate.source_text
                ):
                    return
                self._checkin_suggestion_days.add(key)
                self._save_checkin_suggestion(candidate, decision, day)
                source_link = _telegram_message_link(
                    candidate.chat_id, candidate.source_username, candidate.message_id
                )
                stamp = candidate.source_time.astimezone(ZoneInfo(self.settings.timezone)).strftime(
                    "%Y-%m-%d %H:%M %Z"
                )
                await self.bot.send_message(
                    self.settings.summary_target,
                    (
                        "检测到可复用的签到操作，请确认是否部署。\n"
                        f"群组：{candidate.source_name}\n"
                        f"源消息（{stamp}，{candidate.source_sender}）：{candidate.source_text[:300]}\n"
                        f"Bot 回复：{candidate.bot_reply[:500]}\n"
                        f"AI 判断（高置信）：{decision.reason}\n"
                        f"建议自动发送：{decision.proposed_text}"
                        + (f"\n来源链接：{source_link}" if source_link else "")
                    ),
                    buttons=[
                        [
                            Button.inline("同意并部署", data=f"ks:{candidate.chat_id}".encode()),
                            Button.inline("今天忽略", data=f"ki:{candidate.chat_id}".encode()),
                        ],
                        [
                            Button.inline("7 天忽略", data=f"k7:{candidate.chat_id}".encode()),
                            Button.inline("永久忽略", data=f"kx:{candidate.chat_id}".encode()),
                        ],
                    ],
                    link_preview=False,
                )
                return
        except Exception:
            log.exception("Check-in suggestion analysis failed for candidate=%s", candidate_key)

    async def _checkin_ai_unavailable(self) -> bool:
        primary_model = getattr(self.settings, "llm_model", "")
        if not primary_model:
            return False
        models = [primary_model]
        fallback_model = getattr(self.settings, "llm_fallback_model", None)
        if fallback_model:
            models.append(fallback_model)
        try:
            timeout = aiohttp.ClientTimeout(total=CHECKIN_SUGGESTION_STATUS_TIMEOUT_SECONDS)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    CHECKIN_SUGGESTION_STATUS_URL,
                    headers={"Accept": "application/json"},
                ) as response:
                    if response.status != 200:
                        return False
                    payload = await response.json(content_type=None)
        except Exception as exc:
            log.warning("Unable to read AI status source: %s", _safe_error(exc))
            return False
        available = _checkin_status_available(payload, models)
        return available is False

    def _save_checkin_suggestion(
        self, candidate: CheckinSuggestionCandidate, decision: CheckinSuggestionDecision, day: str
    ) -> None:
        try:
            saved = json.loads(self.archive.get_state(CHECKIN_SUGGESTION_STATE) or "{}")
        except json.JSONDecodeError:
            saved = {}
        if not isinstance(saved, dict):
            saved = {}
        saved[str(candidate.chat_id)] = {
            "day": day,
            "text": decision.proposed_text,
        }
        self.archive.set_state(
            CHECKIN_SUGGESTION_STATE,
            json.dumps(saved, ensure_ascii=False),
        )

    def _checkin_suggestion_ignored(self, chat_id: int, today: dt.date) -> bool:
        try:
            values = json.loads(self.archive.get_state(CHECKIN_SUGGESTION_IGNORES_STATE) or "{}")
        except json.JSONDecodeError:
            return False
        if not isinstance(values, dict):
            return False
        item = values.get(str(chat_id))
        if not isinstance(item, dict):
            return False
        if bool(item.get("permanent")):
            return True
        try:
            return dt.date.fromisoformat(str(item.get("until", ""))) >= today
        except ValueError:
            return False

    def _set_checkin_suggestion_ignore(
        self, chat_id: int, *, until: dt.date | None, permanent: bool
    ) -> None:
        try:
            values = json.loads(self.archive.get_state(CHECKIN_SUGGESTION_IGNORES_STATE) or "{}")
        except json.JSONDecodeError:
            values = {}
        if not isinstance(values, dict):
            values = {}
        values[str(chat_id)] = {
            "permanent": permanent,
            "until": "" if permanent or until is None else until.isoformat(),
        }
        self.archive.set_state(
            CHECKIN_SUGGESTION_IGNORES_STATE, json.dumps(values, ensure_ascii=False)
        )

    async def _notify_checkin_failure(self, chat_id: int, detail: str) -> None:
        config = self._checkin_configs().get(chat_id)
        escalation = ""
        if config is not None and config.failure_streak >= CHECKIN_FAILURE_ESCALATION_DAYS:
            escalation = (
                f"\n升级提醒：该目标已连续失败 {config.failure_streak} 天，"
                "请检查目标是否改名、命令是否失效或 Bot 是否异常。"
            )
        await self.bot.send_message(
            self.settings.summary_target,
            f"自动签到失败：{self._checkin_label(chat_id)}\n{detail[:500]}{escalation}",
            link_preview=False,
        )

    async def _schedule_alert_analysis(self, event: Any) -> None:
        enabled, keywords = self._alert_config()
        subscriptions = self._topic_subscriptions()
        if not enabled and not subscriptions:
            return
        if not getattr(event, "is_group", False):
            return
        source = self.sources.get(getattr(event, "chat_id", None))
        text = (getattr(event, "raw_text", "") or "").strip()
        if source is None or len(text) < 8:
            return
        message_time = _event_message_time(event)
        now = dt.datetime.now(dt.timezone.utc)
        if now - message_time > ALERT_MAX_MESSAGE_AGE:
            return
        lowered = text.casefold()
        signals = (
            tuple(
                signal
                for signal in (*keywords, *ALERT_EVENT_HINTS)
                if signal.casefold() in lowered
            )
            if enabled
            else ()
        )
        topic_match = any(
            int(item["chat_id"]) == source.chat_id
            and str(item["keyword"]).casefold() in lowered
            for item in subscriptions
        )
        if not signals and not topic_match:
            return
        task = asyncio.create_task(
            self._analyze_alert(source, text, event, signals), name="ai-event-alert"
        )
        self._alert_tasks.add(task)
        task.add_done_callback(self._alert_tasks.discard)

    async def _analyze_alert(
        self, source: SourceChat, text: str, event: Any, signals: Sequence[str]
    ) -> None:
        try:
            now = dt.datetime.now(dt.timezone.utc)
            message_time = _event_message_time(event)
            context = self.archive.recent((source.chat_id,), ALERT_CONTEXT_MESSAGES)
            decision = await self.llm.detect_event(
                source.name, _redact_for_llm(text), _redact_messages(context), message_time, now
            )
            if decision.alert:
                async with self._alert_lock:
                    if self._is_duplicate_alert(source.chat_id, decision, now):
                        log.info(
                            "Suppressed repeated event alert for chat id=%s topic=%s signals=%s",
                            source.chat_id,
                            decision.topic,
                            ",".join(signals),
                        )
                    elif not self._alert_topic_ignored(source.chat_id, decision.topic):
                        await self._send_alert(source, text, decision, event)
            await self._notify_topic_subscriptions(source, text, event, message_time, now)
        except Exception:
            log.exception("Event alert analysis failed for chat id=%s", source.chat_id)

    def _is_duplicate_alert(
        self, chat_id: int, decision: EventDecision, now: dt.datetime
    ) -> bool:
        topic = _alert_topic_fingerprint(decision.topic)
        if not topic:
            return True
        previous = self._alert_last_sent.get((chat_id, topic))
        if previous is None:
            return False
        if now - previous.sent_at < ALERT_TOPIC_COOLDOWN:
            return True
        if not decision.is_update:
            return True
        return _alert_detail_fingerprint(decision.new_information) == _alert_detail_fingerprint(
            previous.new_information
        )

    async def _send_alert(
        self, source: SourceChat, text: str, decision: EventDecision, event: Any
    ) -> None:
        link = getattr(getattr(event, "message", None), "id", None)
        if link and source.username:
            link_text = f"\nhttps://t.me/{source.username.lstrip('@')}/{link}"
        else:
            link_text = ""
        encoded = _encode_alert_feedback(source.chat_id, decision.topic)
        await self.bot.send_message(
            self.settings.summary_target,
            (
                f"重大事件提醒 [{_alert_priority_label(decision.priority)}]：{source.name}\n"
                f"主题：{decision.topic}\n"
                f"原因：{decision.reason}\n"
                f"新增信息：{decision.new_information}\n"
                f"原消息：{text[:1200]}{link_text}"
            ),
            buttons=[
                [
                    Button.inline("重要", data=f"af:important:{encoded}".encode()),
                    Button.inline("忽略此主题", data=f"af:ignore:{encoded}".encode()),
                ]
            ],
            link_preview=False,
        )
        topic = _alert_topic_fingerprint(decision.topic)
        self._alert_last_sent[(source.chat_id, topic)] = AlertRecord(
            dt.datetime.now(dt.timezone.utc), decision.new_information
        )

    async def _notify_topic_subscriptions(
        self, source: SourceChat, text: str, event: Any, message_time: dt.datetime, now: dt.datetime
    ) -> None:
        subscriptions = self._topic_subscriptions()
        matched = [item for item in subscriptions if int(item["chat_id"]) == source.chat_id and str(item["keyword"]).casefold() in text.casefold()]
        if not matched:
            return
        changed = False
        for item in matched:
            last = _state_datetime(item.get("last_sent"), dt.timezone.utc)
            if last is not None and now - last < TOPIC_SUBSCRIPTION_COOLDOWN:
                continue
            context = _redact_messages(self.archive.recent((source.chat_id,), ALERT_CONTEXT_MESSAGES))
            decision = await self.llm.detect_event(source.name, _redact_for_llm(text), context, message_time, now)
            if not decision.alert or self._alert_topic_ignored(source.chat_id, decision.topic):
                continue
            item["last_sent"] = now.isoformat()
            changed = True
            link = getattr(getattr(event, "message", None), "id", None)
            link_text = f"\nhttps://t.me/{source.username.lstrip('@')}/{link}" if link and source.username else ""
            await self.bot.send_message(
                self.settings.summary_target,
                f"话题订阅提醒：{source.name}\n关键词：{item['keyword']}\n{decision.new_information}\n原消息：{text[:1000]}{link_text}",
                link_preview=False,
            )
        if changed:
            self._save_topic_subscriptions(subscriptions)

    async def _run_due_checkins(self, now: dt.datetime) -> None:
        today = now.date().isoformat()
        configs = self._checkin_configs()
        for chat_id, config in configs.items():
            if not config.enabled or config.last_success_day == today:
                continue
            scheduled_for = self._scheduled_checkin_at(chat_id, now, config, configs)
            if now < scheduled_for:
                continue
            attempt_count = config.attempt_count if config.attempt_day == today else 0
            retry_at = _state_datetime(config.retry_at, now.tzinfo) if config.attempt_day == today else None
            if attempt_count >= CHECKIN_MAX_ATTEMPTS or (retry_at is not None and now < retry_at):
                continue
            attempt = attempt_count + 1
            configs[chat_id] = _replace_checkin(
                config,
                attempt_day=today,
                attempt_count=attempt,
                retry_at="",
            )
            self._save_checkin_configs(configs)
            try:
                await self._send_checkin(chat_id, mark_today=True)
                configs = self._checkin_configs()
            except Exception:
                log.exception("Scheduled check-in failed for chat id=%s", chat_id)
                configs = self._checkin_configs()
                latest = configs.get(chat_id)
                if latest is not None:
                    configs[chat_id] = _replace_checkin(
                        latest,
                        attempt_day=today,
                        attempt_count=attempt,
                        retry_at=(now + _digest_retry_delay(attempt)).isoformat(),
                    )
                    self._save_checkin_configs(configs)
                    if attempt >= CHECKIN_MAX_ATTEMPTS:
                        await self._notify_checkin_failure(chat_id, latest.last_detail or "达到最大重试次数")
            else:
                latest = configs.get(chat_id)
                if latest is not None and latest.last_status == "sent_unverified":
                    await self._notify_checkin_failure(chat_id, latest.last_detail)

    def _scheduled_checkin_at(
        self,
        chat_id: int,
        now: dt.datetime,
        config: CheckinConfig,
        configs: dict[int, CheckinConfig],
    ) -> dt.datetime:
        zone = ZoneInfo(self.settings.timezone)
        base = dt.datetime.combine(
            now.date(), dt.time(config.hour, config.minute), tzinfo=zone
        )
        scheduled_for = _state_datetime(config.scheduled_for, zone)
        if scheduled_for is not None and scheduled_for.date() == now.date() and scheduled_for >= base:
            return scheduled_for
        if now > base + dt.timedelta(milliseconds=CHECKIN_OFFSET_MAX_MS):
            base += dt.timedelta(days=1)
        scheduled_for = base + dt.timedelta(
            milliseconds=random.randint(CHECKIN_OFFSET_MIN_MS, CHECKIN_OFFSET_MAX_MS)
        )
        configs[chat_id] = _replace_checkin(config, scheduled_for=scheduled_for.isoformat())
        self._save_checkin_configs(configs)
        return scheduled_for

    def _next_checkin_delay(self, now: dt.datetime) -> float:
        zone = ZoneInfo(self.settings.timezone)
        today = now.date().isoformat()
        delays: list[float] = []
        for config in self._checkin_configs().values():
            if not config.enabled or config.last_success_day == today:
                continue
            base = dt.datetime.combine(
                now.date(), dt.time(config.hour, config.minute), tzinfo=zone
            )
            if now < base:
                delays.append((base - now).total_seconds())
                continue
            scheduled_for = _state_datetime(config.scheduled_for, zone)
            if scheduled_for is None:
                delays.append(30.0)
                continue
            if scheduled_for.date() != now.date():
                continue
            if now < scheduled_for:
                delays.append((scheduled_for - now).total_seconds())
        return min(30.0, *delays) if delays else 30.0

    async def _set_content_override(self, chat_id: int, value: str) -> None:
        async with self._content_lock:
            overrides, _ = self._content_state()
            overrides[chat_id] = value
            self.archive.set_state(
                CONTENT_OVERRIDES_STATE,
                json.dumps({str(key): item for key, item in sorted(overrides.items())}),
            )

    async def _classify_sources(
        self, sources: Sequence[SourceChat], force: bool = False
    ) -> None:
        today = dt.datetime.now(ZoneInfo(self.settings.timezone)).date()
        _, classifications = self._content_state()
        candidates = [
            source
            for source in sources
            if force or not _classification_is_current(classifications.get(source.chat_id), today)
        ]
        if not candidates:
            return

        async def classify(source: SourceChat) -> tuple[int, str]:
            messages = limit_summary_messages(
                self.archive.recent((source.chat_id,), CONTENT_CLASSIFICATION_MESSAGES),
                CONTENT_CLASSIFICATION_MAX_CHARS,
            )
            if not messages:
                return source.chat_id, "uncertain"
            return source.chat_id, await self.llm.classify_content(_redact_messages(messages))

        results = await asyncio.gather(
            *(classify(source) for source in candidates), return_exceptions=True
        )
        updates: dict[int, str] = {}
        for source, result in zip(candidates, results, strict=True):
            if isinstance(result, Exception):
                log.warning("Content classification failed for chat id=%s", source.chat_id)
                continue
            chat_id, category = result
            updates[chat_id] = category if category in CONTENT_CATEGORIES else "uncertain"
        if not updates:
            return
        async with self._content_lock:
            _, latest = self._content_state()
            latest.update({chat_id: (category, today) for chat_id, category in updates.items()})
            self.archive.set_state(
                CONTENT_CLASSIFICATIONS_STATE,
                json.dumps(
                    {
                        str(chat_id): {"category": category, "checked_on": checked_on.isoformat()}
                        for chat_id, (category, checked_on) in sorted(latest.items())
                    }
                ),
            )

    async def _summary_source_ids(self) -> tuple[int, ...]:
        await self._classify_sources(tuple(self.sources.values()))
        overrides, classifications = self._content_state()
        return tuple(
            source.chat_id
            for source in self.sources.values()
            if not _summary_excluded(source.chat_id, overrides, classifications)
        )

    async def _scheduler(self) -> None:
        zone = ZoneInfo(self.settings.timezone)
        while True:
            now = dt.datetime.now(zone)
            await self._run_due_checkins(now)
            await self._maybe_send_checkin_report(now)
            day_key = now.date().isoformat()
            attempt_day = self.archive.get_state("digest_attempt_day")
            if attempt_day != day_key:
                self.archive.set_state("digest_attempt_day", day_key)
                self.archive.set_state("digest_attempt_count", "0")
                self.archive.set_state("digest_retry_at", "")
            hour, minute = self._digest_schedule()
            due = (now.hour, now.minute) >= (hour, minute)
            attempts = _state_int(self.archive.get_state("digest_attempt_count"))
            retry_at = _state_datetime(self.archive.get_state("digest_retry_at"), zone)
            retry_ready = retry_at is None or now >= retry_at
            if (
                due
                and self._digest_enabled()
                and self.sources
                and self.archive.get_state("last_digest_day") != day_key
                and attempts < self.settings.summary_max_attempts
                and retry_ready
            ):
                attempt = attempts + 1
                self.archive.set_state("digest_attempt_count", str(attempt))
                try:
                    await self._send_digest_with_mode(self.settings.summary_target, advance_cursors=True)
                    self.archive.set_state("last_digest_day", day_key)
                    self.archive.set_state("digest_retry_at", "")
                    self.archive.prune(self.settings.retention_days)
                except Exception:
                    log.exception("Scheduled digest failed")
                    self.archive.set_state(
                        "digest_retry_at",
                        (now + _digest_retry_delay(attempt)).isoformat(),
                    )
            await asyncio.sleep(max(0.05, self._next_checkin_delay(now)))

    async def _maybe_send_checkin_report(self, now: dt.datetime) -> None:
        if self.archive.get_state(CHECKIN_REPORT_ENABLED_STATE) == "0":
            return
        day = now.date().isoformat()
        if self.archive.get_state(CHECKIN_REPORT_STATE) == day:
            return
        configs = self._checkin_configs()
        enabled = {chat_id: config for chat_id, config in configs.items() if config.enabled}
        if not enabled:
            return
        zone = ZoneInfo(self.settings.timezone)
        for config in enabled.values():
            base = dt.datetime.combine(now.date(), dt.time(config.hour, config.minute), tzinfo=zone)
            if now < base:
                return
            if config.last_success_day == day:
                continue
            scheduled = _state_datetime(config.scheduled_for, zone)
            if scheduled is None or scheduled > now:
                return
            attempts = config.attempt_count if config.attempt_day == day else 0
            if attempts < CHECKIN_MAX_ATTEMPTS:
                return
        lines = [f"签到日报（{day}）"]
        for chat_id, config in sorted(enabled.items(), key=lambda item: self._checkin_label(item[0])):
            detail = f"：{config.last_detail}" if config.last_detail else ""
            lines.append(f"- {self._checkin_label(chat_id)}：{config.last_status or '未执行'}{detail}")
        await self.bot.send_message(self.settings.summary_target, "\n".join(lines), link_preview=False)
        self.archive.set_state(CHECKIN_REPORT_STATE, day)

    async def _send_digest(self, target: int | str) -> None:
        await self._send_digest_with_mode(target, advance_cursors=False)

    async def _send_digest_with_mode(self, target: int | str, *, advance_cursors: bool) -> None:
        async with self._digest_lock:
            now = dt.datetime.now(dt.timezone.utc)
            source_ids = await self._summary_source_ids()
            cursors = self._digest_cursors()
            records: dict[tuple[int, int], StoredMessage] = {}
            for chat_id in source_ids:
                cursor = cursors.get(str(chat_id))
                start = _state_datetime(cursor, dt.timezone.utc) if cursor else None
                if start is None:
                    start = now - dt.timedelta(hours=24)
                else:
                    # Small overlap preserves context for an ongoing topic without
                    # turning the digest back into a full-history report.
                    start -= dt.timedelta(hours=2)
                for item in self.archive.range((chat_id,), start, now, self.settings.summary_max_messages):
                    records[(item.chat_id, item.message_id)] = item
            messages = sorted(records.values(), key=lambda item: item.sent_at)
            messages = messages[-self.settings.summary_max_messages :]
            messages = limit_summary_messages(
                _redact_messages(messages), self.settings.summary_max_chars
            )
            if not messages:
                await self.bot.send_message(target, "过去 24 小时没有可参与摘要的归档消息。")
                if advance_cursors:
                    self._set_digest_cursors(source_ids, now)
                return
            now = dt.datetime.now(ZoneInfo(self.settings.timezone))
            digest = await self.llm.daily_digest(messages, now.date(), as_of=now)
            await _send_long(self.bot, digest, target=target)
            if advance_cursors:
                self._set_digest_cursors(source_ids, dt.datetime.now(dt.timezone.utc))

    def _digest_cursors(self) -> dict[str, str]:
        try:
            raw = json.loads(self.archive.get_state(DIGEST_CURSORS_STATE) or "{}")
        except json.JSONDecodeError:
            return {}
        return raw if isinstance(raw, dict) else {}

    def _set_digest_cursors(self, chat_ids: Sequence[int], value: dt.datetime) -> None:
        cursors = self._digest_cursors()
        stamp = value.astimezone(dt.timezone.utc).isoformat()
        for chat_id in chat_ids:
            cursors[str(chat_id)] = stamp
        self.archive.set_state(DIGEST_CURSORS_STATE, json.dumps(cursors, ensure_ascii=False))

    async def _set_bot_commands(self) -> None:
        await self.bot(
            functions.bots.SetBotCommandsRequest(
                scope=types.BotCommandScopeDefault(),
                lang_code="en",
                commands=[
                    types.BotCommand(command="groups", description="选择要归档的群组"),
                    types.BotCommand(command="recent", description="读取指定群组的最近消息"),
                    types.BotCommand(command="ask", description="查询群聊历史"),
                    types.BotCommand(command="summary", description="生成过去 24 小时摘要"),
                    types.BotCommand(command="content", description="管理成人内容摘要排除"),
                    types.BotCommand(command="checkin", description="管理自动签到"),
                    types.BotCommand(command="alerts", description="配置重大事件提醒"),
                    types.BotCommand(command="topics", description="管理关键词话题订阅"),
                    types.BotCommand(command="refresh", description="刷新群组和机器人列表"),
                    types.BotCommand(command="backup", description="导出消息数据库备份"),
                    types.BotCommand(command="settings", description="设置每日推送"),
                    types.BotCommand(command="status", description="查看归档状态"),
                    types.BotCommand(command="help", description="查看帮助"),
                ],
            )
        )


async def _send_long(event_or_client: Any, text: str, target: int | str | None = None) -> None:
    for chunk in split_message(text):
        if target is None:
            await event_or_client.reply(chunk, link_preview=False)
        else:
            await event_or_client.send_message(target, chunk, link_preview=False)


def split_message(text: str, limit: int = 3800) -> list[str]:
    text = text.strip()
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while remaining:
        boundary = remaining.rfind("\n", 0, limit)
        if boundary < limit // 2:
            boundary = limit
        chunks.append(remaining[:boundary].strip())
        remaining = remaining[boundary:].strip()
    return [chunk for chunk in chunks if chunk]


def _source_ids_from_state(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    try:
        raw_ids = json.loads(value)
    except json.JSONDecodeError:
        return ()
    if not isinstance(raw_ids, list):
        return ()
    ids: list[int] = []
    for raw_id in raw_ids:
        try:
            chat_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if chat_id not in ids:
            ids.append(chat_id)
    return tuple(ids[:MAX_SELECTED_SOURCES])


def _checkin_configs_from_state(value: str | None) -> dict[int, CheckinConfig]:
    try:
        raw = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    configs: dict[int, CheckinConfig] = {}
    for raw_id, raw_config in raw.items():
        try:
            chat_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if not isinstance(raw_config, dict):
            continue
        text = raw_config.get("text", DEFAULT_CHECKIN_TEXT)
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not 1 <= len(text) <= MAX_CHECKIN_TEXT_LENGTH:
            continue
        hour = raw_config.get("hour", DEFAULT_CHECKIN_HOUR)
        minute = raw_config.get("minute", DEFAULT_CHECKIN_MINUTE)
        if not isinstance(hour, int) or not isinstance(minute, int):
            continue
        if not 0 <= hour <= 23 or not 0 <= minute <= 59:
            continue
        attempt_count = raw_config.get("attempt_count", 0)
        if not isinstance(attempt_count, int) or not 0 <= attempt_count <= CHECKIN_MAX_ATTEMPTS:
            attempt_count = 0
        topic_id = raw_config.get("topic_id")
        if topic_id is not None:
            try:
                topic_id = int(topic_id)
            except (TypeError, ValueError):
                topic_id = None
            if topic_id is not None and topic_id <= 0:
                topic_id = None
        history = _checkin_history_from_state(raw_config.get("history"))
        configs[chat_id] = CheckinConfig(
            enabled=bool(raw_config.get("enabled", True)),
            text=text,
            hour=hour,
            minute=minute,
            last_success_day=_safe_day(raw_config.get("last_success_day")),
            attempt_day=_safe_day(raw_config.get("attempt_day")),
            attempt_count=attempt_count,
            retry_at=_safe_datetime(raw_config.get("retry_at")),
            scheduled_for=_safe_datetime(raw_config.get("scheduled_for")),
            topic_id=topic_id,
            last_status=str(raw_config.get("last_status", ""))[:40],
            last_detail=str(raw_config.get("last_detail", ""))[:300],
            failure_streak=(
                int(raw_config.get("failure_streak", 0))
                if isinstance(raw_config.get("failure_streak", 0), int)
                and 0 <= int(raw_config.get("failure_streak", 0)) <= 365
                else 0
            ),
            history=history,
        )
        if len(configs) >= MAX_CHECKIN_TARGETS:
            break
    return configs


def _checkin_config_to_json(config: CheckinConfig) -> dict[str, str | int | bool]:
    return {
        "enabled": config.enabled,
        "text": config.text,
        "hour": config.hour,
        "minute": config.minute,
        "last_success_day": config.last_success_day,
        "attempt_day": config.attempt_day,
        "attempt_count": config.attempt_count,
        "retry_at": config.retry_at,
        "scheduled_for": config.scheduled_for,
        "topic_id": config.topic_id,
        "last_status": config.last_status,
        "last_detail": config.last_detail,
        "failure_streak": config.failure_streak,
        "history": [
            {"day": record.day, "at": record.at, "status": record.status, "detail": record.detail}
            for record in config.history[-CHECKIN_HISTORY_LIMIT:]
        ],
    }


def _replace_checkin(config: CheckinConfig, **changes: Any) -> CheckinConfig:
    values = _checkin_config_to_json(config)
    values.update(changes)
    if isinstance(values.get("history"), list):
        values["history"] = _checkin_history_from_state(values["history"])
    return CheckinConfig(**values)


def _checkin_history_from_state(value: Any) -> tuple[CheckinRecord, ...]:
    if not isinstance(value, list):
        return ()
    records: list[CheckinRecord] = []
    for item in value[-CHECKIN_HISTORY_LIMIT:]:
        if not isinstance(item, dict):
            continue
        day = _safe_day(item.get("day"))
        if not day:
            continue
        records.append(
            CheckinRecord(
                day=day,
                at=_safe_datetime(item.get("at")),
                status=str(item.get("status", "unknown"))[:40],
                detail=str(item.get("detail", ""))[:300],
            )
        )
    return tuple(records)


def _append_checkin_record(config: CheckinConfig, status: str, detail: str, now: dt.datetime) -> CheckinConfig:
    record = CheckinRecord(
        day=now.date().isoformat(),
        at=now.isoformat(),
        status=status,
        detail=detail[:300],
    )
    history = tuple((*config.history, record)[-CHECKIN_HISTORY_LIMIT:])
    failure_streak = config.failure_streak + 1 if status in {"failed", "send_failed"} else 0
    return _replace_checkin(
        config,
        last_status=status,
        last_detail=detail[:300],
        failure_streak=min(failure_streak, 365),
        history=history,
    )


def _checkin_verification_status(
    text: str,
    *,
    from_bot: bool,
    target_kind: str | None,
) -> str | None:
    """Classify a likely check-in response without trusting ordinary group members."""
    normalized = re.sub(r"\s+", " ", text.casefold()).strip()
    if not normalized:
        return None
    if any(keyword.casefold() in normalized for keyword in CHECKIN_FAILURE_KEYWORDS):
        return "failed"

    # A bot target is already the direct recipient. For group targets, only a
    # message authored by a Telegram bot is trusted as the automated response.
    trusted_sender = from_bot or target_kind == "bot"
    if not trusted_sender:
        return None
    command = bool(re.search(r"(?<![a-z0-9_])/(?:qd|checkin)(?![a-z0-9_])", normalized))
    explicit_success = any(keyword.casefold() in normalized for keyword in CHECKIN_SUCCESS_KEYWORDS)
    plain_checkin = "签到" in normalized and not any(
        marker in normalized for marker in ("请发送", "请输入", "回复", "输入")
    )
    if command or explicit_success or plain_checkin:
        return "verified"
    return None


def _event_message_time(event: Any) -> dt.datetime:
    value = getattr(getattr(event, "message", None), "date", None)
    if not isinstance(value, dt.datetime):
        return dt.datetime.now(dt.timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def _alert_topic_fingerprint(topic: str) -> str:
    normalized = re.sub(r"[^\w\u3400-\u9fff]+", " ", topic.casefold(), flags=re.UNICODE)
    return " ".join(normalized.split())[:100]


def _alert_detail_fingerprint(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.casefold()).strip()
    return normalized[:500]


def _alert_priority_label(priority: str) -> str:
    return {"critical": "紧急", "high": "重要"}.get(priority, "重要")


def _is_checkin_suggestion_candidate(text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    if not normalized or any(
        marker in normalized
        for marker in (
            "签到成功",
            "已签到",
            "签到完成",
            "签到失败",
            "怎么签到",
            "如何签到",
            "签到教程",
            "签到规则",
        )
    ):
        return False
    if CHECKIN_COMMAND_PATTERN.search(normalized):
        return True
    has_bot_mention = bool(re.search(r"@[a-z][a-z0-9_]{4,}", normalized, re.IGNORECASE))
    return has_bot_mention and any(word in normalized for word in ("签到", "打卡"))


def _is_literal_checkin_proposal(proposed_text: str, source_text: str) -> bool:
    proposal = " ".join(proposed_text.split())
    source = " ".join(source_text.split())
    if not proposal or len(proposal) > MAX_CHECKIN_TEXT_LENGTH:
        return False
    return proposal.casefold() in source.casefold()


def _checkin_suggestion_retry_delay(attempt: int) -> int:
    """Return a bounded delay so an unavailable provider cannot cause a busy loop."""
    return min(
        CHECKIN_SUGGESTION_RETRY_MAX_SECONDS,
        CHECKIN_SUGGESTION_RETRY_INITIAL_SECONDS * (2 ** min(max(attempt, 0), 10)),
    )


def _checkin_status_available(payload: Any, models: Sequence[str]) -> bool | None:
    """Parse status.input.im without treating unknown/malformed data as an outage."""
    if not isinstance(payload, dict):
        return None
    generated_at = payload.get("generated_at")
    if not isinstance(generated_at, (int, float)):
        return None
    age = dt.datetime.now(dt.timezone.utc).timestamp() - generated_at
    if age < -300 or age > CHECKIN_SUGGESTION_STATUS_MAX_AGE_SECONDS:
        return None
    services = payload.get("services") if isinstance(payload, dict) else None
    if not isinstance(services, list):
        return None
    by_model = {
        str(item.get("model")): item
        for item in services
        if isinstance(item, dict) and item.get("model")
    }
    known = [by_model[model] for model in models if model in by_model]
    if not known:
        return None
    return any(
        isinstance(item.get("last"), dict) and item["last"].get("ok") is True
        for item in known
    )


def _checkin_suggestion_expired(source_time: dt.datetime) -> bool:
    now = dt.datetime.now(dt.timezone.utc)
    timestamp = source_time
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
    else:
        timestamp = timestamp.astimezone(dt.timezone.utc)
    return now - timestamp > CHECKIN_SUGGESTION_MAX_AGE


def _is_retryable_checkin_ai_error(exc: Exception) -> bool:
    """Retry only transient provider failures; auth and malformed responses fail closed."""
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code in {408, 409, 425, 429} or status_code >= 500
    name = type(exc).__name__.lower()
    return any(
        marker in name
        for marker in ("connection", "timeout", "temporarilyunavailable", "ratelimit")
    )


def _telegram_message_link(chat_id: int, username: str | None, message_id: int) -> str | None:
    if username:
        return f"https://t.me/{username.lstrip('@')}/{message_id}"
    raw = str(abs(chat_id))
    if raw.startswith("100"):
        return f"https://t.me/c/{raw[3:]}/{message_id}"
    return None


def _parse_alert_keywords(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values = re.split(r"[,，\\n]", value)
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        return ()
    result: list[str] = []
    for item in values:
        text = " ".join(str(item).split())[:40]
        if text and text.casefold() not in {value.casefold() for value in result}:
            result.append(text)
    return tuple(result[:20])


def _checkin_suggestion_text(value: str | None, chat_id: int) -> str:
    try:
        raw = json.loads(value or "{}")
    except json.JSONDecodeError:
        return DEFAULT_CHECKIN_TEXT
    if not isinstance(raw, dict):
        return DEFAULT_CHECKIN_TEXT
    item = raw.get(str(chat_id), {})
    if not isinstance(item, dict):
        return DEFAULT_CHECKIN_TEXT
    text = str(item.get("text", "")).strip()
    return text[:MAX_CHECKIN_TEXT_LENGTH] or DEFAULT_CHECKIN_TEXT


def _safe_day(value: Any) -> str:
    try:
        return dt.date.fromisoformat(str(value)).isoformat()
    except (TypeError, ValueError):
        return ""


def _safe_datetime(value: Any) -> str:
    try:
        return dt.datetime.fromisoformat(str(value)).isoformat()
    except (TypeError, ValueError):
        return ""


def _safe_error(exc: Exception) -> str:
    text = " ".join(str(exc).split())
    return text[:300] or type(exc).__name__


_SENSITIVE_PATTERNS = (
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{20,}\b"), "[REDACTED_BOT_TOKEN]"),
    (re.compile(r"\b[A-Fa-f0-9]{32}\b"), "[REDACTED_HASH]"),
    (re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"(?<!\d)(?:\+?\d[\d -]{7,}\d)(?!\d)"), "[REDACTED_PHONE]"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[REDACTED_IP]"),
)


def _redact_for_llm(value: str) -> str:
    redacted = value
    for pattern, replacement in _SENSITIVE_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _redact_messages(messages: Sequence[StoredMessage]) -> list[StoredMessage]:
    return [
        StoredMessage(
            chat_id=item.chat_id,
            message_id=item.message_id,
            chat_name=_redact_for_llm(item.chat_name),
            chat_username=item.chat_username,
            sender_id=item.sender_id,
            sender_name=_redact_for_llm(item.sender_name),
            sent_at=item.sent_at,
            text=_redact_for_llm(item.text),
            reply_to_id=item.reply_to_id,
        )
        for item in messages
    ]


def _encode_alert_feedback(chat_id: int, topic: str) -> str:
    value = f"{chat_id}|{_alert_topic_fingerprint(topic)}".encode()
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def _decode_alert_feedback(value: str) -> tuple[int, str]:
    padded = value + "=" * (-len(value) % 4)
    raw = base64.urlsafe_b64decode(padded.encode()).decode()
    chat_id, topic = raw.split("|", 1)
    return int(chat_id), topic[:100]


CONTENT_STATUS_LABELS = {
    "manual_adult": "成人(手动)",
    "auto_adult": "成人(自动)",
    "manual_include": "参与(手动)",
    "general": "普通",
    "unknown": "待分析",
}


def _content_overrides_from_state(value: str | None) -> dict[int, str]:
    try:
        raw = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    overrides: dict[int, str] = {}
    for raw_id, raw_value in raw.items():
        try:
            chat_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        value = str(raw_value).strip().lower()
        if value in {"adult", "include"}:
            overrides[chat_id] = value
    return overrides


def _content_classifications_from_state(
    value: str | None,
) -> dict[int, tuple[str, dt.date]]:
    try:
        raw = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    classifications: dict[int, tuple[str, dt.date]] = {}
    for raw_id, raw_item in raw.items():
        try:
            chat_id = int(raw_id)
            category = str(raw_item["category"]).strip().lower()
            checked_on = dt.date.fromisoformat(str(raw_item["checked_on"]))
        except (KeyError, TypeError, ValueError):
            continue
        if category in CONTENT_CATEGORIES:
            classifications[chat_id] = category, checked_on
    return classifications


def _classification_is_current(
    value: tuple[str, dt.date] | None, today: dt.date
) -> bool:
    return value is not None and 0 <= (today - value[1]).days < CONTENT_CLASSIFICATION_TTL_DAYS


def _content_status(
    chat_id: int,
    overrides: dict[int, str],
    classifications: dict[int, tuple[str, dt.date]],
) -> str:
    override = overrides.get(chat_id)
    if override == "adult":
        return "manual_adult"
    if override == "include":
        return "manual_include"
    category = classifications.get(chat_id, ("uncertain", dt.date.min))[0]
    return "auto_adult" if category == "adult" else category if category == "general" else "unknown"


def _summary_excluded(
    chat_id: int,
    overrides: dict[int, str],
    classifications: dict[int, tuple[str, dt.date]],
) -> bool:
    return _content_status(chat_id, overrides, classifications) in {
        "manual_adult",
        "auto_adult",
    }


def _recent_count(value: str | None) -> int | None:
    if value is None or not value.strip():
        return DEFAULT_RECENT_MESSAGES
    try:
        count = int(value.strip())
    except ValueError:
        return None
    return count if 1 <= count <= MAX_RECENT_MESSAGES else None


def _short_name(value: str, limit: int = 48) -> str:
    value = " ".join(value.split()) or "未命名群组"
    return value if len(value) <= limit else value[: limit - 1] + "..."


def _render_recent_messages(
    source_name: str, records: Sequence[tuple[dt.datetime, str, str]]
) -> str:
    if not records:
        return f"{source_name} 最近没有可读取的文字消息。"
    lines = [f"{source_name} 最近消息："]
    for sent_at, sender, text in records:
        stamp = sent_at.astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines.extend(("", f"{stamp} {sender}", text))
    return "\n".join(lines)


def _sender_name(sender: Any) -> str:
    if sender is None:
        return "unknown"
    title = getattr(sender, "title", None)
    if title:
        return str(title)
    parts = [getattr(sender, "first_name", None), getattr(sender, "last_name", None)]
    name = " ".join(str(part) for part in parts if part)
    return name or getattr(sender, "username", None) or str(getattr(sender, "id", "unknown"))


def _state_int(value: str | None) -> int:
    try:
        return max(0, int(value or "0"))
    except ValueError:
        return 0


def _state_int_in_range(
    value: str | None, default: int, minimum: int, maximum: int
) -> int:
    try:
        parsed = int(value) if value is not None else default
    except ValueError:
        return default
    return parsed if minimum <= parsed <= maximum else default


def _parse_schedule(value: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", value)
    if match is None:
        return None
    hour, minute = (int(part) for part in match.groups())
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        return None
    return hour, minute


def _state_datetime(value: str | None, zone: ZoneInfo) -> dt.datetime | None:
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=zone)
    return parsed.astimezone(zone)


def _digest_retry_delay(attempt: int) -> dt.timedelta:
    return dt.timedelta(minutes=min(60, 5 * (2 ** max(0, attempt - 1))))
