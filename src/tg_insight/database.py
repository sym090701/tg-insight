from __future__ import annotations

import datetime as dt
import logging
import re
import shutil
import sqlite3
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence


log = logging.getLogger("tg_insight.database")


class MessageWriteResult(Enum):
    """Outcome of storing one Telegram message."""

    INSERTED = "inserted"
    UPDATED = "updated"
    UNCHANGED = "unchanged"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class StoredMessage:
    chat_id: int
    message_id: int
    chat_name: str
    chat_username: str | None
    sender_id: int | None
    sender_name: str
    sent_at: dt.datetime
    text: str
    reply_to_id: int | None = None

    @property
    def link(self) -> str | None:
        if self.chat_username:
            return f"https://t.me/{self.chat_username.lstrip('@')}/{self.message_id}"
        raw = str(abs(self.chat_id))
        if raw.startswith("100"):
            return f"https://t.me/c/{raw[3:]}/{self.message_id}"
        return None


@dataclass(frozen=True)
class DeferredCheckinSuggestion:
    chat_id: int
    message_id: int
    source_name: str
    source_username: str | None
    source_sender: str
    source_time: dt.datetime
    source_text: str
    bot_reply: str
    retry_after: dt.datetime
    retry_count: int
    last_error: str


@dataclass(frozen=True)
class AIRetryJob:
    job_key: str
    kind: str
    payload: str
    retry_after: dt.datetime
    retry_count: int
    last_error: str
    created_at: dt.datetime


class Archive:
    def __init__(
        self, path: Path, max_messages: int, min_free_mb: int, max_bytes: int | None = None
    ):
        self.path = path
        self.max_messages = max_messages
        self.max_bytes = max_bytes
        self.min_free_bytes = min_free_mb * 1024 * 1024
        self._message_count: int | None = None
        self._last_space_warning = 0.0

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    chat_id INTEGER NOT NULL,
                    message_id INTEGER NOT NULL,
                    chat_name TEXT NOT NULL,
                    chat_username TEXT,
                    sender_id INTEGER,
                    sender_name TEXT NOT NULL,
                    sent_at TEXT NOT NULL,
                    text TEXT NOT NULL,
                    reply_to_id INTEGER,
                    PRIMARY KEY (chat_id, message_id)
                );
                CREATE INDEX IF NOT EXISTS idx_messages_chat_date
                    ON messages(chat_id, sent_at);
                CREATE INDEX IF NOT EXISTS idx_messages_date
                    ON messages(sent_at);

                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                    text,
                    content='messages',
                    content_rowid='rowid',
                    tokenize='trigram'
                );

                CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
                    INSERT INTO messages_fts(rowid, text) VALUES (new.rowid, new.text);
                END;
                CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
                    INSERT INTO messages_fts(messages_fts, rowid, text)
                    VALUES ('delete', old.rowid, old.text);
                END;
                CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
                    INSERT INTO messages_fts(messages_fts, rowid, text)
                    VALUES ('delete', old.rowid, old.text);
                    INSERT INTO messages_fts(rowid, text) VALUES (new.rowid, new.text);
                END;

                CREATE TABLE IF NOT EXISTS state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS deferred_checkin_suggestions (
                    chat_id INTEGER NOT NULL,
                    message_id INTEGER NOT NULL,
                    source_name TEXT NOT NULL,
                    source_username TEXT,
                    source_sender TEXT NOT NULL,
                    source_time TEXT NOT NULL,
                    source_text TEXT NOT NULL,
                    bot_reply TEXT NOT NULL,
                    retry_after TEXT NOT NULL,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT NOT NULL,
                    PRIMARY KEY (chat_id, message_id)
                );
                CREATE INDEX IF NOT EXISTS idx_deferred_checkin_retry_after
                    ON deferred_checkin_suggestions(retry_after);

                CREATE TABLE IF NOT EXISTS ai_retry_jobs (
                    job_key TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    retry_after TEXT NOT NULL,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_ai_retry_jobs_retry_after
                    ON ai_retry_jobs(retry_after);
                """
            )
            self._message_count = int(
                conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            )
            self._enforce_message_limit(conn)

    def upsert(self, message: StoredMessage) -> MessageWriteResult:
        text = _clean_text(message.text)
        if not text:
            return MessageWriteResult.SKIPPED
        sent_at = _utc_iso(message.sent_at)
        values = (
            message.chat_name,
            message.chat_username,
            message.sender_id,
            message.sender_name,
            sent_at,
            text,
            message.reply_to_id,
        )
        with self.connect() as conn:
            existing = conn.execute(
                """
                SELECT chat_name, chat_username, sender_id, sender_name, sent_at, text, reply_to_id
                FROM messages WHERE chat_id=? AND message_id=?
                """,
                (message.chat_id, message.message_id),
            ).fetchone()
            if existing is not None:
                if tuple(existing) == values:
                    return MessageWriteResult.UNCHANGED
                conn.execute(
                    """
                    UPDATE messages
                    SET chat_name=?, chat_username=?, sender_id=?, sender_name=?, sent_at=?, text=?, reply_to_id=?
                    WHERE chat_id=? AND message_id=?
                    """,
                    (*values, message.chat_id, message.message_id),
                )
                return MessageWriteResult.UPDATED

            if not self._has_free_space() or not self._has_capacity():
                return MessageWriteResult.SKIPPED
            conn.execute(
                """
                INSERT INTO messages (
                    chat_id, message_id, chat_name, chat_username, sender_id,
                    sender_name, sent_at, text, reply_to_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.chat_id,
                    message.message_id,
                    *values,
                ),
            )
            self._message_count = self._current_count(conn) + 1
            self._enforce_message_limit(conn)
        return MessageWriteResult.INSERTED

    def delete(self, chat_id: int, message_ids: Iterable[int]) -> None:
        ids = tuple(message_ids)
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        with self.connect() as conn:
            cursor = conn.execute(
                f"DELETE FROM messages WHERE chat_id=? AND message_id IN ({placeholders})",
                (chat_id, *ids),
            )
            self._message_count = max(0, self._current_count(conn) - cursor.rowcount)

    def range(
        self,
        chat_ids: Sequence[int],
        start: dt.datetime,
        end: dt.datetime,
        limit: int,
    ) -> list[StoredMessage]:
        if not chat_ids or limit <= 0:
            return []
        placeholders = ",".join("?" for _ in chat_ids)
        sql = f"""
            SELECT * FROM messages
            WHERE chat_id IN ({placeholders}) AND sent_at BETWEEN ? AND ?
            ORDER BY sent_at DESC LIMIT ?
        """
        params = (*chat_ids, _utc_iso(start), _utc_iso(end), limit)
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_message(row) for row in reversed(rows)]

    def search(
        self,
        chat_ids: Sequence[int],
        terms: Sequence[str],
        limit: int,
    ) -> list[StoredMessage]:
        if not chat_ids or limit <= 0:
            return []

        normalized = _normalize_terms(terms)
        results: dict[tuple[int, int], StoredMessage] = {}
        placeholders = ",".join("?" for _ in chat_ids)

        long_terms = [term for term in normalized if len(term) >= 3]
        if long_terms:
            query = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in long_terms)
            sql = f"""
                SELECT m.*, bm25(messages_fts) AS rank
                FROM messages_fts
                JOIN messages m ON m.rowid = messages_fts.rowid
                WHERE messages_fts MATCH ? AND m.chat_id IN ({placeholders})
                ORDER BY rank, m.sent_at DESC LIMIT ?
            """
            try:
                with self.connect() as conn:
                    rows = conn.execute(sql, (query, *chat_ids, limit)).fetchall()
                for row in rows:
                    message = _row_to_message(row)
                    results[(message.chat_id, message.message_id)] = message
            except sqlite3.OperationalError:
                pass

        if len(results) < limit and normalized:
            clauses = " OR ".join("text LIKE ?" for _ in normalized)
            sql = f"""
                SELECT * FROM messages
                WHERE chat_id IN ({placeholders}) AND ({clauses})
                ORDER BY sent_at DESC LIMIT ?
            """
            params = (*chat_ids, *(f"%{term}%" for term in normalized), limit)
            with self.connect() as conn:
                rows = conn.execute(sql, params).fetchall()
            for row in rows:
                message = _row_to_message(row)
                results.setdefault((message.chat_id, message.message_id), message)

        ordered = sorted(results.values(), key=lambda item: item.sent_at, reverse=True)
        return ordered[:limit]

    def recent(self, chat_ids: Sequence[int], limit: int) -> list[StoredMessage]:
        now = dt.datetime.now(dt.timezone.utc)
        return self.range(chat_ids, now - dt.timedelta(days=30), now, limit)

    def count(self, chat_ids: Sequence[int]) -> int:
        if not chat_ids:
            return 0
        placeholders = ",".join("?" for _ in chat_ids)
        with self.connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) AS count FROM messages WHERE chat_id IN ({placeholders})",
                tuple(chat_ids),
            ).fetchone()
        return int(row["count"])

    def prune(self, retention_days: int) -> int:
        if retention_days <= 0:
            return 0
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=retention_days)
        with self.connect() as conn:
            cursor = conn.execute(
                "DELETE FROM messages WHERE sent_at < ?", (_utc_iso(cutoff),)
            )
            self._message_count = max(0, self._current_count(conn) - cursor.rowcount)
            if cursor.rowcount:
                conn.execute("PRAGMA incremental_vacuum(1000)")
            return cursor.rowcount

    def _current_count(self, conn: sqlite3.Connection) -> int:
        if self._message_count is None:
            self._message_count = int(
                conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            )
        return self._message_count

    def _enforce_message_limit(self, conn: sqlite3.Connection) -> int:
        count = self._current_count(conn)
        excess = count - self.max_messages
        if excess <= 0:
            return 0
        cursor = conn.execute(
            """
            DELETE FROM messages WHERE rowid IN (
                SELECT rowid FROM messages
                ORDER BY sent_at ASC, rowid ASC LIMIT ?
            )
            """,
            (excess,),
        )
        self._message_count = max(0, count - cursor.rowcount)
        return cursor.rowcount

    def _has_free_space(self) -> bool:
        try:
            enough = shutil.disk_usage(self.path.parent).free >= self.min_free_bytes
        except OSError:
            enough = False
        if not enough and time.monotonic() - self._last_space_warning >= 300:
            log.error(
                "Archive write paused to preserve %d MB of free disk space",
                self.min_free_bytes // (1024 * 1024),
            )
            self._last_space_warning = time.monotonic()
        return enough

    def _has_capacity(self) -> bool:
        if self.max_bytes is None:
            return True
        size = sum(
            candidate.stat().st_size
            for candidate in (
                self.path,
                self.path.with_name(self.path.name + "-wal"),
                self.path.with_name(self.path.name + "-shm"),
            )
            if candidate.exists()
        )
        enough = size < self.max_bytes
        if not enough and time.monotonic() - self._last_space_warning >= 300:
            log.error(
                "Archive write paused at the %d GiB storage limit",
                self.max_bytes // (1024 * 1024 * 1024),
            )
            self._last_space_warning = time.monotonic()
        return enough

    def get_state(self, key: str) -> str | None:
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def set_state(self, key: str, value: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO state(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def defer_checkin_suggestion(self, suggestion: DeferredCheckinSuggestion) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO deferred_checkin_suggestions (
                    chat_id, message_id, source_name, source_username, source_sender,
                    source_time, source_text, bot_reply, retry_after, retry_count, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id, message_id) DO UPDATE SET
                    source_name=excluded.source_name,
                    source_username=excluded.source_username,
                    source_sender=excluded.source_sender,
                    source_time=excluded.source_time,
                    source_text=excluded.source_text,
                    bot_reply=excluded.bot_reply,
                    retry_after=excluded.retry_after,
                    last_error=excluded.last_error
                """,
                (
                    suggestion.chat_id,
                    suggestion.message_id,
                    suggestion.source_name,
                    suggestion.source_username,
                    suggestion.source_sender,
                    _utc_iso(suggestion.source_time),
                    suggestion.source_text,
                    suggestion.bot_reply,
                    _utc_iso(suggestion.retry_after),
                    suggestion.retry_count,
                    suggestion.last_error[:500],
                ),
            )

    def due_deferred_checkin_suggestions(
        self, now: dt.datetime
    ) -> list[DeferredCheckinSuggestion]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM deferred_checkin_suggestions
                WHERE retry_after <= ?
                ORDER BY retry_after ASC, source_time ASC
                """,
                (_utc_iso(now),),
            ).fetchall()
        return [_row_to_deferred_checkin_suggestion(row) for row in rows]

    def deferred_checkin_suggestions(self) -> list[DeferredCheckinSuggestion]:
        """Return all candidates waiting for an AI provider to recover."""
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM deferred_checkin_suggestions
                ORDER BY retry_after ASC, source_time ASC
                """
            ).fetchall()
        return [_row_to_deferred_checkin_suggestion(row) for row in rows]

    def reschedule_deferred_checkin_suggestion(
        self, chat_id: int, message_id: int, retry_after: dt.datetime
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE deferred_checkin_suggestions
                SET retry_after=?, retry_count=retry_count + 1
                WHERE chat_id=? AND message_id=?
                """,
                (_utc_iso(retry_after), chat_id, message_id),
            )

    def delete_deferred_checkin_suggestion(self, chat_id: int, message_id: int) -> None:
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM deferred_checkin_suggestions WHERE chat_id=? AND message_id=?",
                (chat_id, message_id),
            )

    def enqueue_ai_retry_job(self, job: AIRetryJob) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO ai_retry_jobs (
                    job_key, kind, payload, retry_after, retry_count, last_error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_key) DO UPDATE SET
                    kind=excluded.kind,
                    payload=excluded.payload,
                    retry_after=excluded.retry_after,
                    retry_count=excluded.retry_count,
                    last_error=excluded.last_error
                """,
                (
                    job.job_key,
                    job.kind,
                    job.payload,
                    _utc_iso(job.retry_after),
                    job.retry_count,
                    job.last_error[:500],
                    _utc_iso(job.created_at),
                ),
            )

    def ai_retry_jobs(self) -> list[AIRetryJob]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM ai_retry_jobs ORDER BY retry_after ASC, created_at ASC"
            ).fetchall()
        return [_row_to_ai_retry_job(row) for row in rows]

    def delete_ai_retry_job(self, job_key: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM ai_retry_jobs WHERE job_key=?", (job_key,))

    def backup_to(self, destination: Path) -> None:
        """Create a consistent SQLite backup without copying a live WAL file."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = self.connect()
        try:
            target = sqlite3.connect(destination)
            try:
                source.backup(target)
            finally:
                target.close()
        finally:
            source.close()


def _clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    return " ".join(text.split())


def _normalize_terms(terms: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in terms:
        term = " ".join(raw.strip().split())[:80]
        if not term or term.casefold() in seen:
            continue
        seen.add(term.casefold())
        result.append(term)
    return result[:12]


def _utc_iso(value: dt.datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc).isoformat()


def _row_to_message(row: sqlite3.Row) -> StoredMessage:
    return StoredMessage(
        chat_id=int(row["chat_id"]),
        message_id=int(row["message_id"]),
        chat_name=str(row["chat_name"]),
        chat_username=row["chat_username"],
        sender_id=row["sender_id"],
        sender_name=str(row["sender_name"]),
        sent_at=dt.datetime.fromisoformat(str(row["sent_at"])),
        text=str(row["text"]),
        reply_to_id=row["reply_to_id"],
    )


def _row_to_deferred_checkin_suggestion(row: sqlite3.Row) -> DeferredCheckinSuggestion:
    return DeferredCheckinSuggestion(
        chat_id=int(row["chat_id"]),
        message_id=int(row["message_id"]),
        source_name=str(row["source_name"]),
        source_username=row["source_username"],
        source_sender=str(row["source_sender"]),
        source_time=dt.datetime.fromisoformat(str(row["source_time"])),
        source_text=str(row["source_text"]),
        bot_reply=str(row["bot_reply"]),
        retry_after=dt.datetime.fromisoformat(str(row["retry_after"])),
        retry_count=int(row["retry_count"]),
        last_error=str(row["last_error"]),
    )


def _row_to_ai_retry_job(row: sqlite3.Row) -> AIRetryJob:
    return AIRetryJob(
        job_key=str(row["job_key"]),
        kind=str(row["kind"]),
        payload=str(row["payload"]),
        retry_after=dt.datetime.fromisoformat(str(row["retry_after"])),
        retry_count=int(row["retry_count"]),
        last_error=str(row["last_error"]),
        created_at=dt.datetime.fromisoformat(str(row["created_at"])),
    )
