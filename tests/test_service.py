import asyncio
import datetime as dt
import json
from types import SimpleNamespace

import pytest
from zoneinfo import ZoneInfo

from tg_insight.service import (
    ALERT_TOPIC_COOLDOWN,
    AlertRecord,
    CHECKIN_SUGGESTION_REPLY_WAIT_SECONDS,
    CHECKIN_SUGGESTION_IGNORES_STATE,
    CheckinSuggestionCandidate,
    CheckinSuggestionDecision,
    CHECKIN_OFFSET_MAX_MS,
    CHECKIN_OFFSET_MIN_MS,
    CheckinConfig,
    EventDecision,
    DEFAULT_RECENT_MESSAGES,
    SourceChat,
    TelegramInsightService,
    _checkin_configs_from_state,
    _alert_topic_fingerprint,
    _classification_is_current,
    _checkin_verification_status,
    _content_classifications_from_state,
    _content_overrides_from_state,
    _content_status,
    _digest_retry_delay,
    _recent_count,
    _render_recent_messages,
    _parse_schedule,
    _short_name,
    _source_ids_from_state,
    _state_datetime,
    _state_int,
    _state_int_in_range,
    _summary_excluded,
    _is_checkin_suggestion_candidate,
    split_message,
)


def test_split_message_preserves_content() -> None:
    source = "first line\n" + "x" * 5000 + "\nlast line"
    chunks = split_message(source, limit=1000)
    assert all(len(chunk) <= 1000 for chunk in chunks)
    assert "".join(chunks).replace("\n", "") == source.replace("\n", "")


def test_split_message_leaves_short_text_alone() -> None:
    assert split_message("short") == ["short"]


def test_digest_retry_delay_is_bounded() -> None:
    assert _digest_retry_delay(1) == dt.timedelta(minutes=5)
    assert _digest_retry_delay(3) == dt.timedelta(minutes=20)
    assert _digest_retry_delay(20) == dt.timedelta(minutes=60)


def test_scheduler_state_parsing_fails_closed() -> None:
    zone = ZoneInfo("Asia/Shanghai")
    assert _state_int("invalid") == 0
    assert _state_datetime("invalid", zone) is None
    assert _state_datetime("2026-08-18T21:00:00+08:00", zone) is not None


def test_digest_schedule_parsing_and_state_bounds() -> None:
    assert _parse_schedule(" 9:05 ") == (9, 5)
    assert _parse_schedule("24:00") is None
    assert _parse_schedule("09:60") is None
    assert _parse_schedule("9.05") is None
    assert _state_int_in_range("22", 21, 0, 23) == 22
    assert _state_int_in_range("invalid", 21, 0, 23) == 21
    assert _state_int_in_range("24", 21, 0, 23) == 21


def test_persisted_digest_settings_override_defaults() -> None:
    state = {"digest_hour": "8", "digest_minute": "30", "digest_enabled": "0"}
    service = object.__new__(TelegramInsightService)
    service.settings = SimpleNamespace(summary_hour=21, summary_minute=0)
    service.archive = SimpleNamespace(get_state=state.get)

    assert service._digest_schedule() == (8, 30)
    assert service._digest_enabled() is False


def test_content_classification_state_and_overrides_control_summary_exclusion() -> None:
    today = dt.date(2026, 8, 18)
    classifications = _content_classifications_from_state(
        '{"1":{"category":"adult","checked_on":"2026-08-18"},'
        '"2":{"category":"general","checked_on":"2026-08-10"}}'
    )
    overrides = _content_overrides_from_state('{"2":"adult","3":"include"}')

    assert _classification_is_current(classifications[1], today)
    assert not _classification_is_current(classifications[2], today)
    assert _content_status(1, overrides, classifications) == "auto_adult"
    assert _summary_excluded(1, overrides, classifications)
    assert _content_status(2, overrides, classifications) == "manual_adult"
    assert _summary_excluded(2, overrides, classifications)
    assert _content_status(3, overrides, classifications) == "manual_include"
    assert not _summary_excluded(3, overrides, classifications)


@pytest.mark.asyncio
async def test_summary_source_selection_excludes_manual_and_auto_adult_groups() -> None:
    state = {
        "content_classifications": (
            '{"1":{"category":"adult","checked_on":"2026-08-18"},'
            '"2":{"category":"general","checked_on":"2026-08-18"}}'
        ),
        "content_overrides": '{"2":"adult","3":"include"}',
    }
    service = object.__new__(TelegramInsightService)
    service.settings = SimpleNamespace(timezone="Asia/Shanghai")
    service.sources = {
        index: SourceChat(entity=object(), chat_id=index, name=f"群组 {index}", username=None)
        for index in range(1, 4)
    }
    service.archive = SimpleNamespace(get_state=state.get)

    async def classify_sources(_sources):
        return None

    service._classify_sources = classify_sources

    assert await service._summary_source_ids() == (3,)


def test_selected_source_state_rejects_invalid_values_and_deduplicates() -> None:
    assert _source_ids_from_state(None) == ()
    assert _source_ids_from_state("not-json") == ()
    assert _source_ids_from_state('{"chat_id": 1}') == ()
    assert _source_ids_from_state('[1, "2", "bad", 1]') == (1, 2)


def test_checkin_config_state_is_validated_and_defaults_are_stable() -> None:
    configs = _checkin_configs_from_state(
        '{"-1001":{"enabled":false,"text":"@bot /checkin","hour":8,"minute":30,'
        '"last_success_day":"2026-08-18","attempt_day":"bad","attempt_count":2,'
        '"retry_at":"2026-08-18T08:35:00+08:00"},'
        '"bad":{"text":"ignored"},"-1002":{"text":"","hour":8,"minute":0}}'
    )

    assert configs == {
        -1001: CheckinConfig(
            enabled=False,
            text="@bot /checkin",
            hour=8,
            minute=30,
            last_success_day="2026-08-18",
            attempt_count=2,
            retry_at="2026-08-18T08:35:00+08:00",
        )
    }

    defaults = _checkin_configs_from_state('{"-1003":{"text":"签到"}}')
    assert defaults[-1003].hour == 0
    assert defaults[-1003].minute == 30


def test_checkin_picker_labels_bot_targets() -> None:
    service = object.__new__(TelegramInsightService)
    service.available_checkin_targets = {
        42: SourceChat(
            entity=object(),
            chat_id=42,
            name="签到机器人",
            username="checkin_bot",
            target_kind="bot",
        )
    }
    service.archive = SimpleNamespace(
        get_state=lambda key: '{"42":{"text":"/checkin"}}' if key == "checkin_configs" else None
    )

    text, _buttons = service._checkin_picker()

    assert "机器人：签到机器人" in text


@pytest.mark.asyncio
async def test_due_checkin_sends_once_and_persists_today() -> None:
    state = {
        "checkin_configs": (
            '{"-1001":{"enabled":true,"text":"@bot /checkin","hour":8,"minute":30,'
            '"last_success_day":"","attempt_day":"","attempt_count":0,"retry_at":""}}'
        )
    }
    sent: list[tuple[object, str]] = []

    class Archive:
        def get_state(self, key):
            return state.get(key)

        def set_state(self, key, value):
            state[key] = value

    service_ref: dict[str, TelegramInsightService] = {}

    class User:
        async def send_message(self, entity, text, **_kwargs):
            sent.append((entity, text))
            waiter = next(iter(service_ref["service"]._checkin_waiters.values()))
            waiter.set_result(("verified", "签到"))

    service = object.__new__(TelegramInsightService)
    service.settings = SimpleNamespace(timezone="Asia/Shanghai")
    service.archive = Archive()
    service.user = User()
    service.available_sources = {
        -1001: SourceChat(entity="target", chat_id=-1001, name="签到群", username=None)
    }
    service.available_checkin_targets = service.available_sources
    service._checkin_lock = asyncio.Lock()
    service._checkin_waiters = {}
    service_ref["service"] = service

    now = dt.datetime(2026, 8, 18, 8, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    await service._run_due_checkins(now)

    planned = _checkin_configs_from_state(state["checkin_configs"])[-1001].scheduled_for
    planned_at = dt.datetime.fromisoformat(planned)
    assert CHECKIN_OFFSET_MIN_MS <= (planned_at - now).total_seconds() * 1000 <= CHECKIN_OFFSET_MAX_MS

    await service._run_due_checkins(now + dt.timedelta(seconds=1))
    await service._run_due_checkins(now + dt.timedelta(seconds=1))

    assert sent == [("target", "@bot /checkin")]
    assert _checkin_configs_from_state(state["checkin_configs"])[-1001].last_success_day == dt.datetime.now(
        ZoneInfo("Asia/Shanghai")
    ).date().isoformat()


@pytest.mark.asyncio
async def test_outbound_bot_command_is_not_checkin_verification() -> None:
    state = {"checkin_configs": '{"42":{"text":"/checkin"}}'}

    class Archive:
        def get_state(self, key):
            return state.get(key)

    service = object.__new__(TelegramInsightService)
    service.settings = SimpleNamespace(timezone="Asia/Shanghai")
    service.archive = Archive()
    service.available_checkin_targets = {
        42: SourceChat(entity="bot", chat_id=42, name="签到机器人", username=None, target_kind="bot")
    }
    service._checkin_waiters = {}
    key = (42, dt.datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat())
    waiter = asyncio.get_running_loop().create_future()
    service._checkin_waiters[key] = waiter

    await service._observe_checkin_message(
        SimpleNamespace(
            out=True,
            chat_id=42,
            raw_text="/checkin",
            message=SimpleNamespace(sender=SimpleNamespace(bot=False)),
        )
    )
    assert not waiter.done()

    await service._observe_checkin_message(
        SimpleNamespace(
            out=False,
            chat_id=42,
            raw_text="签到",
            message=SimpleNamespace(sender=SimpleNamespace(bot=True)),
        )
    )
    assert waiter.result() == ("verified", "签到")


@pytest.mark.parametrize(
    ("text", "from_bot", "target_kind", "expected"),
    [
        ("签到", True, "group", "verified"),
        ("签到成功啦！", True, "group", "verified"),
        ("请回复 /qd 完成签到", True, "group", "verified"),
        ("@user /qd", False, "group", None),
        ("签到失败，请稍后重试", True, "group", "failed"),
        ("/qd", False, "bot", "verified"),
    ],
)
def test_checkin_verification_accepts_flexible_bot_responses(
    text: str, from_bot: bool, target_kind: str, expected: str | None
) -> None:
    assert _checkin_verification_status(
        text, from_bot=from_bot, target_kind=target_kind
    ) == expected


@pytest.mark.asyncio
async def test_missed_checkin_is_scheduled_for_next_day() -> None:
    state = {
        "checkin_configs": (
            '{"-1001":{"enabled":true,"text":"签到","hour":0,"minute":30,'
            '"last_success_day":"2026-08-17","attempt_day":"","attempt_count":0,"retry_at":""}}'
        )
    }

    class Archive:
        def get_state(self, key):
            return state.get(key)

        def set_state(self, key, value):
            state[key] = value

    class User:
        async def send_message(self, *_args, **_kwargs):
            raise AssertionError("a missed check-in must not send immediately")

    service = object.__new__(TelegramInsightService)
    service.settings = SimpleNamespace(timezone="Asia/Shanghai")
    service.archive = Archive()
    service.user = User()
    service.available_sources = {
        -1001: SourceChat(entity="target", chat_id=-1001, name="签到群", username=None)
    }
    service.available_checkin_targets = service.available_sources
    service._checkin_lock = asyncio.Lock()

    now = dt.datetime(2026, 8, 18, 21, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    await service._run_due_checkins(now)

    scheduled = dt.datetime.fromisoformat(
        _checkin_configs_from_state(state["checkin_configs"])[-1001].scheduled_for
    )
    assert scheduled.date() == dt.date(2026, 8, 19)


def test_recent_count_is_bounded() -> None:
    assert _recent_count(None) == DEFAULT_RECENT_MESSAGES
    assert _recent_count("1") == 1
    assert _recent_count("50") == 50
    assert _recent_count("0") is None
    assert _recent_count("51") is None
    assert _recent_count("many") is None


def test_recent_renderer_and_group_label_keep_bot_output_bounded() -> None:
    sent_at = dt.datetime(2026, 8, 18, 8, tzinfo=dt.timezone.utc)
    rendered = _render_recent_messages("群组", [(sent_at, "Alice", "hello")])
    assert "群组 最近消息" in rendered
    assert "Alice" in rendered
    assert _short_name("x" * 80) == "x" * 47 + "..."


@pytest.mark.asyncio
async def test_group_page_bulk_selection_persists_and_skips_fixed_sources() -> None:
    service = object.__new__(TelegramInsightService)
    service.available_sources = {
        index: SourceChat(entity=object(), chat_id=index, name=f"群组 {index}", username=None)
        for index in range(1, 11)
    }
    service.sources = {1: service.available_sources[1]}
    service._configured_source_ids = {1}
    service._source_lock = asyncio.Lock()
    service._backfill_tasks = set()
    state: dict[str, str] = {}
    service.archive = SimpleNamespace(set_state=lambda key, value: state.__setitem__(key, value))

    async def backfill(_sources):
        return None

    service._backfill_sources = backfill
    added, removed, limited = await service._set_group_page(0, selected=True)
    await asyncio.sleep(0)

    assert (added, removed, limited) == (7, 0, False)
    assert set(service.sources) == {1, 2, 3, 4, 5, 6, 7, 10}
    assert state["selected_source_chats"] == "[2, 3, 4, 5, 6, 7, 10]"

    added, removed, limited = await service._set_group_page(0, selected=False)
    assert (added, removed, limited) == (0, 7, False)
    assert set(service.sources) == {1}


@pytest.mark.asyncio
async def test_keyword_alert_candidate_is_reviewed_by_ai_before_sending() -> None:
    class Archive:
        def recent(self, _chat_ids, _limit):
            return []

    class LLM:
        def __init__(self) -> None:
            self.calls = []

        async def detect_event(self, *args):
            self.calls.append(args)
            return EventDecision(False)

    class Bot:
        async def send_message(self, *_args, **_kwargs):
            raise AssertionError("keyword candidates must not send before AI review")

    service = object.__new__(TelegramInsightService)
    service.archive = Archive()
    service.llm = LLM()
    service.bot = Bot()
    service.settings = SimpleNamespace(summary_target=5361150559)
    service.sources = {1: SourceChat(entity=object(), chat_id=1, name="群组", username=None)}
    service._alert_tasks = set()
    service._alert_last_sent = {}
    service._alert_lock = asyncio.Lock()
    service._alert_config = lambda: (True, ("发布",))
    event = SimpleNamespace(
        is_group=True,
        chat_id=1,
        raw_text="明天发布例行周报，请大家关注。",
        message=SimpleNamespace(id=5, date=dt.datetime.now(dt.timezone.utc)),
    )

    await service._schedule_alert_analysis(event)
    await asyncio.gather(*tuple(service._alert_tasks))

    assert len(service.llm.calls) == 1
    assert service.llm.calls[0][0:2] == ("群组", "明天发布例行周报，请大家关注。")


@pytest.mark.asyncio
async def test_old_alert_candidate_is_not_sent_for_ai_analysis() -> None:
    service = object.__new__(TelegramInsightService)
    service.sources = {1: SourceChat(entity=object(), chat_id=1, name="群组", username=None)}
    service._alert_config = lambda: (True, ("故障",))
    service._alert_tasks = set()
    event = SimpleNamespace(
        is_group=True,
        chat_id=1,
        raw_text="故障已在昨天恢复。",
        message=SimpleNamespace(id=5, date=dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=2)),
    )

    await service._schedule_alert_analysis(event)

    assert not service._alert_tasks


def test_alert_deduplication_is_per_group_topic_and_requires_a_real_update() -> None:
    service = object.__new__(TelegramInsightService)
    now = dt.datetime(2026, 8, 19, tzinfo=dt.timezone.utc)
    service._alert_last_sent = {
        (1, _alert_topic_fingerprint("支付服务中断")): AlertRecord(now, "正在抢修")
    }
    original = EventDecision(True, "high", "服务中断", "支付服务中断", "仍在抢修", False)
    update = EventDecision(True, "high", "服务中断", "支付服务中断", "官方公布恢复时间", True)

    assert service._is_duplicate_alert(1, original, now + dt.timedelta(minutes=1))
    assert service._is_duplicate_alert(1, original, now + ALERT_TOPIC_COOLDOWN + dt.timedelta(minutes=1))
    assert not service._is_duplicate_alert(2, original, now + dt.timedelta(minutes=1))
    assert not service._is_duplicate_alert(1, update, now + ALERT_TOPIC_COOLDOWN + dt.timedelta(minutes=1))
    assert service._is_duplicate_alert(
        1,
        EventDecision(True, "high", "服务中断", "支付服务中断", "正在抢修", True),
        now + ALERT_TOPIC_COOLDOWN + dt.timedelta(minutes=1),
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("@daily_bot /qd", True),
        ("/checkin", True),
        ("@daily_bot 签到", True),
        ("怎么签到？", False),
        ("签到成功，获得积分", False),
        ("这是签到教程", False),
        ("今天群里有人签到", False),
    ],
)
def test_checkin_suggestion_candidates_require_an_actual_action(text: str, expected: bool) -> None:
    assert _is_checkin_suggestion_candidate(text) is expected


@pytest.mark.asyncio
async def test_checkin_suggestion_includes_source_bot_reply_and_ai_reason(monkeypatch) -> None:
    state: dict[str, str] = {}
    sent: list[tuple[int, str, dict]] = []

    class Archive:
        def get_state(self, key):
            return state.get(key)

        def set_state(self, key, value):
            state[key] = value

    class LLM:
        async def assess_checkin_suggestion(self, *args):
            assert args[3:] == ("@daily_bot /qd", "签到成功，获得 1 积分")
            return CheckinSuggestionDecision(
                True, "high", "用户命令已获 Bot 成功确认", "@daily_bot /qd"
            )

    class Bot:
        async def send_message(self, target, text, **kwargs):
            sent.append((target, text, kwargs))

    service = object.__new__(TelegramInsightService)
    service.settings = SimpleNamespace(summary_target=5361150559, timezone="Asia/Shanghai")
    service.archive = Archive()
    service.llm = LLM()
    service.bot = Bot()
    service._checkin_suggestion_days = set()
    service._checkin_suggestion_candidates = {
        (-1001, 7): CheckinSuggestionCandidate(
            chat_id=-1001,
            message_id=7,
            source_name="签到群",
            source_username="daily_group",
            source_sender="Alice",
            source_time=dt.datetime(2026, 8, 19, tzinfo=dt.timezone.utc),
            source_text="@daily_bot /qd",
            bot_reply="签到成功，获得 1 积分",
        )
    }
    service._checkin_configs = lambda: {}
    monkeypatch.setattr("tg_insight.service.CHECKIN_SUGGESTION_REPLY_WAIT_SECONDS", 0)

    await service._analyze_checkin_suggestion((-1001, 7))

    assert len(sent) == 1
    _, notification, kwargs = sent[0]
    assert "源消息" in notification and "@daily_bot /qd" in notification
    assert "Bot 回复：签到成功，获得 1 积分" in notification
    assert "AI 判断（高置信）：用户命令已获 Bot 成功确认" in notification
    assert "https://t.me/daily_group/7" in notification
    assert kwargs["buttons"]
    saved = json.loads(state["checkin_suggestions"])
    assert saved["-1001"]["text"] == "@daily_bot /qd"


@pytest.mark.asyncio
async def test_checkin_suggestion_requires_a_direct_bot_reply(monkeypatch) -> None:
    class LLM:
        async def assess_checkin_suggestion(self, *_args):
            raise AssertionError("AI must not run without bot evidence")

    service = object.__new__(TelegramInsightService)
    service.llm = LLM()
    service._checkin_suggestion_candidates = {
        (-1001, 7): CheckinSuggestionCandidate(
            -1001, 7, "签到群", None, "Alice", dt.datetime.now(dt.timezone.utc), "/qd"
        )
    }
    monkeypatch.setattr("tg_insight.service.CHECKIN_SUGGESTION_REPLY_WAIT_SECONDS", 0)

    await service._analyze_checkin_suggestion((-1001, 7))


def test_checkin_suggestion_only_associates_a_direct_bot_reply() -> None:
    candidate = CheckinSuggestionCandidate(
        -1001, 7, "签到群", None, "Alice", dt.datetime.now(dt.timezone.utc), "/qd"
    )
    service = object.__new__(TelegramInsightService)
    service._checkin_suggestion_candidates = {(-1001, 7): candidate}

    service._record_checkin_suggestion_bot_reply(
        SimpleNamespace(chat_id=-1001, message=SimpleNamespace(reply_to_msg_id=6)), "签到成功"
    )
    assert not candidate.bot_reply

    service._record_checkin_suggestion_bot_reply(
        SimpleNamespace(chat_id=-1001, message=SimpleNamespace(reply_to_msg_id=7)), "签到成功"
    )
    assert candidate.bot_reply == "签到成功"


def test_checkin_suggestion_ignores_are_persisted_for_today_seven_days_or_forever() -> None:
    state: dict[str, str] = {}
    service = object.__new__(TelegramInsightService)
    service.archive = SimpleNamespace(get_state=state.get, set_state=state.__setitem__)
    today = dt.date(2026, 8, 20)

    service._set_checkin_suggestion_ignore(1, until=today, permanent=False)
    assert service._checkin_suggestion_ignored(1, today)
    assert not service._checkin_suggestion_ignored(1, today + dt.timedelta(days=1))

    service._set_checkin_suggestion_ignore(1, until=today + dt.timedelta(days=6), permanent=False)
    assert service._checkin_suggestion_ignored(1, today + dt.timedelta(days=6))
    assert not service._checkin_suggestion_ignored(1, today + dt.timedelta(days=7))

    service._set_checkin_suggestion_ignore(1, until=None, permanent=True)
    assert service._checkin_suggestion_ignored(1, today + dt.timedelta(days=10_000))
    assert CHECKIN_SUGGESTION_IGNORES_STATE in state


@pytest.mark.asyncio
async def test_unarchived_group_is_listened_for_checkin_without_writing_messages() -> None:
    state: dict[str, str] = {}
    service = object.__new__(TelegramInsightService)
    service.settings = SimpleNamespace(timezone="Asia/Shanghai")
    service.sources = {}
    service.available_sources = {
        -1001: SourceChat(entity=object(), chat_id=-1001, name="未归档群", username="untracked")
    }
    service.archive = SimpleNamespace(get_state=state.get, set_state=state.__setitem__)
    service._checkin_suggestion_days = set()
    service._checkin_suggestion_candidates = {}
    service._checkin_suggestion_tasks = set()
    service._checkin_configs = lambda: {}
    service._user_id = 5361150559

    await service._suggest_checkin_from_message(
        SimpleNamespace(
            chat_id=-1001,
            raw_text="@daily_bot /qd",
            message=SimpleNamespace(
                id=7,
                date=dt.datetime.now(dt.timezone.utc),
                sender=SimpleNamespace(id=111, bot=False, first_name="Alice"),
            ),
        )
    )

    assert (-1001, 7) in service._checkin_suggestion_candidates
    assert not state
    for task in service._checkin_suggestion_tasks:
        task.cancel()
    await asyncio.gather(*service._checkin_suggestion_tasks, return_exceptions=True)
