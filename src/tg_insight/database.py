from __future__ import annotations

import datetime as dt
import logging
import re
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


log = logging.getLogger("tg_insight.database")


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
                """
            )
            self._message_count = int(
                conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            )
            self._enforce_message_limit(conn)

    def upsert(self, message: StoredMessage) -> bool:
        text = _clean_text(message.text)
        if not text:
            return False
        if not self._has_free_space():
            return False
        if not self._has_capacity():
            return False
        with self.connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM messages WHERE chat_id=? AND message_id=?",
                (message.chat_id, message.message_id),
            ).fetchone()
            conn.execute(
                """
                INSERT INTO messages (
                    chat_id, message_id, chat_name, chat_username, sender_id,
                    sender_name, sent_at, text, reply_to_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id, message_id) DO UPDATE SET
                    chat_name=excluded.chat_name,
                    chat_username=excluded.chat_username,
                    sender_id=excluded.sender_id,
                    sender_name=excluded.sender_name,
                    sent_at=excluded.sent_at,
                    text=excluded.text,
                    reply_to_id=excluded.reply_to_id
                """,
                (
                    message.chat_id,
                    message.message_id,
                    message.chat_name,
                    message.chat_username,
                    message.sender_id,
                    message.sender_name,
                    _utc_iso(message.sent_at),
                    text,
                    message.reply_to_id,
                ),
            )
            if exists is None:
                self._message_count = self._current_count(conn) + 1
            self._enforce_message_limit(conn)
        return True

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

    def prune_chat(self, chat_id: int, retention_days: int) -> int:
        if retention_days <= 0:
            return 0
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=retention_days)
        with self.connect() as conn:
            cursor = conn.execute(
                "DELETE FROM messages WHERE chat_id=? AND sent_at < ?",
                (chat_id, _utc_iso(cutoff)),
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
