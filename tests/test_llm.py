import datetime as dt
import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tg_insight.database import StoredMessage
from tg_insight.llm import (
    Answer,
    EventDecision,
    InsightLLM,
    _batch_messages,
    _content_category,
    _fallback_terms,
    _group_messages,
    _records_jsonl,
    limit_summary_messages,
    render_answer,
)
import tg_insight.llm as llm_module


def sample(text: str = "hello") -> StoredMessage:
    return StoredMessage(
        chat_id=-1001234567890,
        message_id=8,
        chat_name="group",
        chat_username="public_group",
        sender_id=1,
        sender_name="Alice",
        sent_at=dt.datetime(2026, 8, 18, 12, 0, tzinfo=dt.timezone.utc),
        text=text,
    )


def test_render_answer_includes_source_link() -> None:
    rendered = render_answer(Answer("Grounded [S1]", (sample(),)))
    assert "Grounded [S1]" in rendered
    assert "https://t.me/public_group/8" in rendered


def test_render_answer_includes_every_cited_source() -> None:
    messages = tuple(
        StoredMessage(
            chat_id=-1001,
            message_id=index,
            chat_name="test",
            chat_username="test",
            sender_id=index,
            sender_name=f"user {index}",
            sent_at=dt.datetime(2026, 8, 18, tzinfo=dt.timezone.utc),
            text=f"message {index}",
        )
        for index in range(1, 21)
    )
    rendered = render_answer(Answer("结论来自 [S3] 和 [S18]。", messages))

    assert "[S3]" in rendered
    assert "[S18]" in rendered
    assert "[S1]" not in rendered


def test_batch_messages_respects_rough_character_limit() -> None:
    batches = _batch_messages([sample("a" * 900), sample("b" * 900)], max_chars=1500)
    assert len(batches) == 2


def test_fallback_terms_handles_chinese() -> None:
    assert "固件版本什么时候发布" in _fallback_terms("固件版本什么时候发布？")


def test_content_category_parser_fails_closed_to_uncertain() -> None:
    assert _content_category('{"category":"adult"}') == "adult"
    assert _content_category('```json\n{"category":"general"}\n```') == "general"
    assert _content_category('{"category":"unexpected"}') == "uncertain"
    assert _content_category("not json") == "uncertain"


class EventLLMStub(InsightLLM):
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[list[dict[str, str]]] = []

    async def _complete(self, messages, temperature):
        self.calls.append(messages)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


@pytest.mark.asyncio
async def test_event_detection_requires_a_structured_high_priority_decision() -> None:
    stub = EventLLMStub(
        '{"alert":true,"priority":"high","reason":"官方确认服务中断",'
        '"topic":"服务中断","new_information":"已确认影响支付且正在抢修",'
        '"is_update":false}'
    )
    cutoff = dt.datetime(2026, 8, 19, 0, 0, tzinfo=dt.timezone.utc)

    decision = await stub.detect_event(
        "群组", "支付服务已中断", [sample()], cutoff - dt.timedelta(minutes=2), cutoff
    )

    assert decision == EventDecision(
        True, "high", "官方确认服务中断", "服务中断", "已确认影响支付且正在抢修", False
    )
    prompt = stub.calls[0][0]["content"]
    records = stub.calls[0][1]["content"]
    assert "never alert merely because it contains a keyword" in prompt
    assert "is_update" in prompt
    assert "Target message time: 2026-08-18T23:58:00+00:00" in records
    assert "Recent same-group context" in records


@pytest.mark.asyncio
async def test_event_detection_fails_closed_for_incomplete_or_invalid_results() -> None:
    stub = EventLLMStub('{"alert":true,"priority":"medium","reason":"可能有事"}')
    now = dt.datetime(2026, 8, 19, tzinfo=dt.timezone.utc)

    decision = await stub.detect_event("群组", "可能有重要通知", [], now, now)

    assert decision == EventDecision(False)


class FallbackCompletions:
    def __init__(self) -> None:
        self.models: list[str] = []

    async def create(self, *, model, messages, temperature):
        self.models.append(model)
        if model == "primary":
            raise ConnectionError("primary offline")
        return "fallback response"


def test_completion_uses_fallback_only_for_temporary_primary_errors(monkeypatch) -> None:
    completions = FallbackCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    llm = object.__new__(InsightLLM)
    llm.client = client
    llm.model = "primary"
    llm.fallback_model = "fallback"
    llm._request_slots = asyncio.Semaphore(1)
    monkeypatch.setattr(llm_module, "_should_use_fallback", lambda exc: isinstance(exc, ConnectionError))

    response = asyncio.run(llm._complete([], 0))

    assert response == "fallback response"
    assert completions.models == ["primary", "fallback"]


def test_summary_message_budget_keeps_newest_records_within_limit() -> None:
    messages = [
        replace(sample("x" * 900), message_id=index, sent_at=sample().sent_at + dt.timedelta(minutes=index))
        for index in range(1, 5)
    ]

    selected = limit_summary_messages(messages, max_chars=1_300)

    assert [item.message_id for item in selected] == [4]
    assert len(_records_jsonl(selected)) <= 1_300


def test_summary_records_include_precise_timestamp_and_message_age() -> None:
    reference = dt.datetime(2026, 8, 19, 0, 0, tzinfo=dt.timezone.utc)

    record = _records_jsonl([sample()], reference)

    assert '"sent_at": "2026-08-18T12:00:00+00:00"' in record
    assert '"age_hours": 12.0' in record


def test_group_messages_keeps_chats_separate() -> None:
    first = sample("first")
    second = replace(first, chat_id=-1002, chat_name="other group", message_id=9)
    third = replace(first, message_id=10, text="third")

    groups = _group_messages([first, second, third])

    assert [(name, [item.message_id for item in items]) for name, items in groups] == [
        ("group", [8, 10]),
        ("other group", [9]),
    ]


class DigestLLMStub(InsightLLM):
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []
        self.responses = iter(("[A] group one", "[S] group two", "final briefing"))

    async def _complete(self, messages, temperature):
        self.calls.append(messages)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=next(self.responses)))]
        )


async def test_daily_digest_prioritizes_then_keeps_group_sections() -> None:
    first = sample("first")
    second = replace(first, chat_id=-1002, chat_name="other group", message_id=9)
    stub = DigestLLMStub()

    cutoff = dt.datetime(2026, 8, 18, 16, 0, tzinfo=dt.timezone.utc)
    result = await InsightLLM.daily_digest(
        stub, [first, second], dt.date(2026, 8, 18), as_of=cutoff
    )

    assert result == "final briefing"
    assert len(stub.calls) == 3
    final_prompt = stub.calls[-1][0]["content"]
    final_records = stub.calls[-1][1]["content"]
    assert "今日最重要信息" in final_prompt
    assert "分群情报" in final_prompt
    assert "old item as new" in final_prompt
    assert "Analysis cutoff: 2026-08-18T16:00:00+00:00" in final_prompt
    assert "Group: group" in final_records
    assert "Group: other group" in final_records
    first_group_records = stub.calls[0][1]["content"]
    assert '"age_hours": 4.0' in first_group_records
    assert "持续跟进" in stub.calls[0][0]["content"]
