import asyncio
import datetime as dt
from types import SimpleNamespace

import pytest
from zoneinfo import ZoneInfo

from tg_insight.service import (
    CHECKIN_OFFSET_MAX_MS,
    CHECKIN_OFFSET_MIN_MS,
    CheckinConfig,
    DEFAULT_RECENT_MESSAGES,
    SourceChat,
    TelegramInsightService,
    _checkin_configs_from_state,
    _classification_is_current,
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

    class User:
        async def send_message(self, entity, text, **_kwargs):
            sent.append((entity, text))

    service = object.__new__(TelegramInsightService)
    service.settings = SimpleNamespace(timezone="Asia/Shanghai")
    service.archive = Archive()
    service.user = User()
    service.available_sources = {
        -1001: SourceChat(entity="target", chat_id=-1001, name="签到群", username=None)
    }
    service.available_checkin_targets = service.available_sources
    service._checkin_lock = asyncio.Lock()

    now = dt.datetime(2026, 8, 18, 8, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    await service._run_due_checkins(now)

    planned = _checkin_configs_from_state(state["checkin_configs"])[-1001].scheduled_for
    planned_at = dt.datetime.fromisoformat(planned)
    assert CHECKIN_OFFSET_MIN_MS <= (planned_at - now).total_seconds() * 1000 <= CHECKIN_OFFSET_MAX_MS

    await service._run_due_checkins(now + dt.timedelta(seconds=1))
    await service._run_due_checkins(now + dt.timedelta(seconds=1))

    assert sent == [("target", "@bot /checkin")]
    assert _checkin_configs_from_state(state["checkin_configs"])[-1001].last_success_day == "2026-08-18"


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
