from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import random
import re
from dataclasses import dataclass
from typing import Any, Sequence
from zoneinfo import ZoneInfo

from telethon import Button, TelegramClient, events, functions, types

from .config import Settings
from .database import Archive, StoredMessage
from .llm import CONTENT_CATEGORIES, InsightLLM, limit_summary_messages, render_answer

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


@dataclass(frozen=True)
class SourceChat:
    entity: Any
    chat_id: int
    name: str
    username: str | None
    target_kind: str = "group"


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

    async def run(self) -> None:
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.archive.initialize()
        await self.user.connect()
        if not await self.user.is_user_authorized():
            raise RuntimeError("Telegram user session is not authorized; run auth first")

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
            await asyncio.gather(*self._backfill_tasks, return_exceptions=True)
            await asyncio.gather(*self._classification_tasks, return_exceptions=True)
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
        self.bot.add_event_handler(self._on_ask, events.NewMessage(pattern=r"^/ask(?:\s+(.+))?$"))
        self.bot.add_event_handler(self._on_groups, events.NewMessage(pattern=r"^/groups$"))
        self.bot.add_event_handler(self._on_recent, events.NewMessage(pattern=r"^/recent(?:\s+(.+))?$"))
        self.bot.add_event_handler(
            self._on_group_callback,
            events.CallbackQuery(
                pattern=rb"^(?:(?:g|gp|ga|gx|gd|r|rp|c|cp|cr|cra|k|kp|kc|km|kt|ke|kr|kd)(?::|$)|(?:s|st|sd)$)"
            ),
        )
        self.bot.add_event_handler(self._on_private_text, events.NewMessage(incoming=True))

    async def _on_new_message(self, event: Any) -> None:
        source = self.sources.get(event.chat_id)
        if source is not None:
            await self._store_telegram_message(source, event.message)

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
                )
            text = "\n".join(lines)
        buttons: list[list[Any]] = [
            [Button.inline("添加签到目标", data=b"kp:0")],
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
        text = (
            f"自动签到：{self._checkin_label(chat_id)}\n"
            f"状态：{state}\n"
            f"时间：{config.hour:02d}:{config.minute:02d}（UTC+8，随机延后 300-800ms）\n"
            f"文本：{config.text}\n"
            f"上次成功：{last}\n"
            "立即签到会计入今天，避免定时任务重复发送。"
        )
        return text, [
            [Button.inline("更改签到文本", data=f"km:{chat_id}".encode())],
            [Button.inline("更改签到时间", data=f"kt:{chat_id}".encode())],
            [
                Button.inline(
                    "关闭自动签到" if config.enabled else "开启自动签到",
                    data=f"ke:{chat_id}".encode(),
                )
            ],
            [Button.inline("立即签到", data=f"kr:{chat_id}".encode())],
            [Button.inline("移除此群", data=f"kd:{chat_id}".encode())],
            [Button.inline("返回列表", data=b"kc:0")],
        ]

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
            answer = await self.llm.answer(question, messages)
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
            await self.user.send_message(target.entity, config.text, link_preview=False)
            if mark_today:
                today = dt.datetime.now(ZoneInfo(self.settings.timezone)).date().isoformat()
                configs[chat_id] = _replace_checkin(
                    config,
                    last_success_day=today,
                    attempt_day=today,
                    attempt_count=0,
                    retry_at="",
                    scheduled_for="",
                )
                self._save_checkin_configs(configs)

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
            return source.chat_id, await self.llm.classify_content(messages)

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
                    await self._send_digest(self.settings.summary_target)
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

    async def _send_digest(self, target: int | str) -> None:
        async with self._digest_lock:
            now = dt.datetime.now(dt.timezone.utc)
            source_ids = await self._summary_source_ids()
            messages = self.archive.range(
                source_ids,
                now - dt.timedelta(hours=24),
                now,
                self.settings.summary_max_messages,
            )
            messages = limit_summary_messages(
                messages, self.settings.summary_max_chars
            )
            if not messages:
                await self.bot.send_message(target, "过去 24 小时没有可参与摘要的归档消息。")
                return
            day = dt.datetime.now(ZoneInfo(self.settings.timezone)).date()
            digest = await self.llm.daily_digest(messages, day)
            await _send_long(self.bot, digest, target=target)

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
    }


def _replace_checkin(config: CheckinConfig, **changes: Any) -> CheckinConfig:
    values = _checkin_config_to_json(config)
    values.update(changes)
    return CheckinConfig(**values)


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
