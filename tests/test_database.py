import datetime as dt
from types import SimpleNamespace

from tg_insight.database import (
    AIRetryJob,
    Archive,
    DeferredCheckinSuggestion,
    MessageWriteResult,
    StoredMessage,
)


NOW = dt.datetime(2026, 8, 18, 12, 0, tzinfo=dt.timezone.utc)


def make_archive(tmp_path, max_messages: int = 1_000) -> Archive:
    return Archive(tmp_path / "messages.db", max_messages=max_messages, min_free_mb=0)


def message(message_id: int, text: str, *, minutes: int = 0) -> StoredMessage:
    return StoredMessage(
        chat_id=-1001234567890,
        message_id=message_id,
        chat_name="test group",
        chat_username=None,
        sender_id=100 + message_id,
        sender_name=f"user {message_id}",
        sent_at=NOW + dt.timedelta(minutes=minutes),
        text=text,
    )


def test_upsert_range_and_private_group_link(tmp_path) -> None:
    archive = make_archive(tmp_path)
    archive.initialize()
    assert archive.upsert(message(1, "first value")) is MessageWriteResult.INSERTED
    assert archive.upsert(message(1, "first value")) is MessageWriteResult.UNCHANGED
    assert archive.upsert(message(1, "updated value")) is MessageWriteResult.UPDATED

    rows = archive.range(
        [-1001234567890], NOW - dt.timedelta(hours=1), NOW + dt.timedelta(hours=1), 10
    )
    assert len(rows) == 1
    assert rows[0].text == "updated value"
    assert rows[0].link == "https://t.me/c/1234567890/1"


def test_search_supports_chinese_trigram_and_short_like(tmp_path) -> None:
    archive = make_archive(tmp_path)
    archive.initialize()
    archive.upsert(message(1, "大家决定下周发布新的固件版本"))
    archive.upsert(message(2, "今天讨论了网络代理设置"))

    long_result = archive.search([-1001234567890], ["固件版本"], 10)
    short_result = archive.search([-1001234567890], ["代理"], 10)

    assert [item.message_id for item in long_result] == [1]
    assert [item.message_id for item in short_result] == [2]


def test_search_is_scoped_to_allowed_chats(tmp_path) -> None:
    archive = make_archive(tmp_path)
    archive.initialize()
    archive.upsert(message(1, "secret launch date"))
    other = StoredMessage(
        chat_id=-100999,
        message_id=2,
        chat_name="other",
        chat_username="other",
        sender_id=1,
        sender_name="other user",
        sent_at=NOW,
        text="secret launch date",
    )
    archive.upsert(other)

    rows = archive.search([-1001234567890], ["launch date"], 10)
    assert {(row.chat_id, row.message_id) for row in rows} == {(-1001234567890, 1)}


def test_delete_and_state(tmp_path) -> None:
    archive = make_archive(tmp_path)
    archive.initialize()
    archive.upsert(message(1, "remove me"))
    archive.delete(-1001234567890, [1])
    assert archive.count([-1001234567890]) == 0

    assert archive.get_state("last_digest_day") is None
    archive.set_state("last_digest_day", "2026-08-18")
    assert archive.get_state("last_digest_day") == "2026-08-18"


def test_deferred_checkin_suggestions_are_durable_and_rescheduled(tmp_path) -> None:
    archive = make_archive(tmp_path)
    archive.initialize()
    suggestion = DeferredCheckinSuggestion(
        chat_id=-1001234567890,
        message_id=77,
        source_name="签到群",
        source_username="daily_group",
        source_sender="Alice",
        source_time=NOW,
        source_text="/qd",
        bot_reply="签到成功",
        retry_after=NOW + dt.timedelta(hours=1),
        retry_count=0,
        last_error="HTTP 404",
    )
    archive.defer_checkin_suggestion(suggestion)

    assert archive.due_deferred_checkin_suggestions(NOW + dt.timedelta(minutes=59)) == []
    due = archive.due_deferred_checkin_suggestions(NOW + dt.timedelta(hours=1))
    assert len(due) == 1
    assert due[0].source_text == "/qd"
    assert due[0].retry_count == 0

    next_retry = NOW + dt.timedelta(days=1)
    archive.reschedule_deferred_checkin_suggestion(-1001234567890, 77, next_retry)
    rescheduled = archive.due_deferred_checkin_suggestions(next_retry)
    assert rescheduled[0].retry_count == 1
    assert rescheduled[0].retry_after == next_retry

    archive.delete_deferred_checkin_suggestion(-1001234567890, 77)
    assert archive.due_deferred_checkin_suggestions(next_retry + dt.timedelta(days=1)) == []


def test_ai_retry_jobs_are_durable_and_deletable(tmp_path) -> None:
    archive = make_archive(tmp_path)
    archive.initialize()
    job = AIRetryJob(
        job_key="alert:-1001:42",
        kind="alert",
        payload='{"chat_id": -1001}',
        retry_after=NOW,
        retry_count=6,
        last_error="HTTP 503",
        created_at=NOW,
    )
    archive.enqueue_ai_retry_job(job)
    jobs = archive.ai_retry_jobs()
    assert len(jobs) == 1
    assert jobs[0] == job
    archive.delete_ai_retry_job(job.job_key)
    assert archive.ai_retry_jobs() == []


def test_archive_evicts_oldest_messages_at_hard_limit(tmp_path) -> None:
    archive = make_archive(tmp_path, max_messages=2)
    archive.initialize()

    archive.upsert(message(1, "oldest", minutes=1))
    archive.upsert(message(2, "middle", minutes=2))
    archive.upsert(message(3, "newest", minutes=3))

    rows = archive.range(
        [-1001234567890], NOW, NOW + dt.timedelta(hours=1), limit=10
    )
    assert [item.message_id for item in rows] == [2, 3]


def test_archive_rejects_write_below_free_space_reserve(tmp_path, monkeypatch) -> None:
    archive = Archive(tmp_path / "messages.db", max_messages=1_000, min_free_mb=1)
    archive.initialize()
    monkeypatch.setattr(
        "tg_insight.database.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=0),
    )

    assert archive.upsert(message(1, "must not be stored")) is MessageWriteResult.SKIPPED
    assert archive.count([-1001234567890]) == 0


def test_archive_rejects_writes_after_reaching_byte_limit(tmp_path) -> None:
    archive = Archive(
        tmp_path / "messages.db", max_messages=1_000, min_free_mb=0, max_bytes=1
    )
    archive.initialize()

    assert archive.upsert(message(1, "must not be stored")) is MessageWriteResult.SKIPPED
    assert archive.count([-1001234567890]) == 0
