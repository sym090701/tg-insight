from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo


class ConfigError(ValueError):
    pass


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _int_csv(value: str) -> frozenset[int]:
    try:
        return frozenset(int(item) for item in _csv(value))
    except ValueError as exc:
        raise ConfigError("TG_ALLOWED_USER_IDS must contain numeric IDs") from exc


def parse_peer(value: str) -> int | str:
    value = value.strip()
    if value.lstrip("-").isdigit():
        return int(value)
    return value


@dataclass(frozen=True)
class Settings:
    api_id: int
    api_hash: str
    bot_token: str
    source_chats: tuple[int | str, ...]
    summary_target: int | str
    allowed_user_ids: frozenset[int]
    llm_api_key: str
    llm_model: str
    llm_fallback_model: str | None
    llm_base_url: str | None
    data_dir: Path
    timezone: str = "Asia/Shanghai"
    summary_hour: int = 21
    summary_minute: int = 0
    backfill_days: int = 30
    backfill_max_messages: int = 20_000
    retention_days: int = 730
    archive_max_messages: int = 10_000_000
    archive_max_gb: int = 10
    archive_min_free_mb: int = 1_024
    query_max_sources: int = 40
    summary_max_messages: int = 500
    summary_max_chars: int = 120_000
    summary_max_attempts: int = 3

    @property
    def user_session(self) -> Path:
        return self.data_dir / "user"

    @property
    def bot_session(self) -> Path:
        return self.data_dir / "bot"

    @property
    def database_path(self) -> Path:
        return self.data_dir / "messages.db"

    @property
    def archive_max_bytes(self) -> int:
        return self.archive_max_gb * 1024 * 1024 * 1024

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        values = os.environ if env is None else env

        def required(name: str) -> str:
            value = values.get(name, "").strip()
            if not value:
                raise ConfigError(f"{name} is required")
            return value

        def integer(name: str, default: int) -> int:
            try:
                return int(values.get(name, str(default)))
            except ValueError as exc:
                raise ConfigError(f"{name} must be numeric") from exc

        try:
            api_id = int(required("TG_API_ID"))
        except ValueError as exc:
            raise ConfigError("TG_API_ID must be numeric") from exc

        source_chats = tuple(
            parse_peer(v) for v in _csv(values.get("TG_SOURCE_CHATS", ""))
        )
        allowed_user_ids = _int_csv(required("TG_ALLOWED_USER_IDS"))
        if not allowed_user_ids:
            raise ConfigError("TG_ALLOWED_USER_IDS must not be empty")

        settings = cls(
            api_id=api_id,
            api_hash=required("TG_API_HASH"),
            bot_token=required("TG_BOT_TOKEN"),
            source_chats=source_chats,
            summary_target=parse_peer(required("TG_SUMMARY_TARGET")),
            allowed_user_ids=allowed_user_ids,
            llm_api_key=required("LLM_API_KEY"),
            llm_model=required("LLM_MODEL"),
            llm_fallback_model=values.get("LLM_FALLBACK_MODEL", "").strip() or None,
            llm_base_url=values.get("LLM_BASE_URL", "").strip() or None,
            data_dir=Path(values.get("DATA_DIR", "/data")),
            timezone=values.get("TZ", "Asia/Shanghai").strip() or "Asia/Shanghai",
            summary_hour=integer("SUMMARY_HOUR", 21),
            summary_minute=integer("SUMMARY_MINUTE", 0),
            backfill_days=integer("BACKFILL_DAYS", 30),
            backfill_max_messages=integer("BACKFILL_MAX_MESSAGES", 20_000),
            retention_days=integer("RETENTION_DAYS", 730),
            archive_max_messages=integer("ARCHIVE_MAX_MESSAGES", 10_000_000),
            archive_max_gb=integer("ARCHIVE_MAX_GB", 10),
            archive_min_free_mb=integer("ARCHIVE_MIN_FREE_MB", 1_024),
            query_max_sources=integer("QUERY_MAX_SOURCES", 40),
            summary_max_messages=integer("SUMMARY_MAX_MESSAGES", 500),
            summary_max_chars=integer("SUMMARY_MAX_CHARS", 120_000),
            summary_max_attempts=integer("SUMMARY_MAX_ATTEMPTS", 3),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        try:
            ZoneInfo(self.timezone)
        except Exception as exc:
            raise ConfigError(f"Invalid timezone: {self.timezone}") from exc
        if not 0 <= self.summary_hour <= 23:
            raise ConfigError("SUMMARY_HOUR must be between 0 and 23")
        if not 0 <= self.summary_minute <= 59:
            raise ConfigError("SUMMARY_MINUTE must be between 0 and 59")
        if self.llm_base_url:
            parsed = urlsplit(self.llm_base_url)
            if (
                parsed.scheme.lower() != "https"
                or not parsed.hostname
                or parsed.username
                or parsed.password
            ):
                raise ConfigError("LLM_BASE_URL must be an HTTPS URL without credentials")
        if self.llm_fallback_model == self.llm_model:
            raise ConfigError("LLM_FALLBACK_MODEL must differ from LLM_MODEL")
        limits = (
            ("BACKFILL_DAYS", self.backfill_days, 0, 3_650),
            ("BACKFILL_MAX_MESSAGES", self.backfill_max_messages, 1, 100_000),
            ("RETENTION_DAYS", self.retention_days, 1, 3_650),
            ("ARCHIVE_MAX_MESSAGES", self.archive_max_messages, 1_000, 10_000_000),
            ("ARCHIVE_MAX_GB", self.archive_max_gb, 1, 100),
            ("ARCHIVE_MIN_FREE_MB", self.archive_min_free_mb, 256, 8_192),
            ("QUERY_MAX_SOURCES", self.query_max_sources, 1, 100),
            ("SUMMARY_MAX_MESSAGES", self.summary_max_messages, 1, 2_000),
            ("SUMMARY_MAX_CHARS", self.summary_max_chars, 10_000, 500_000),
            ("SUMMARY_MAX_ATTEMPTS", self.summary_max_attempts, 1, 5),
        )
        for name, value, minimum, maximum in limits:
            if not minimum <= value <= maximum:
                raise ConfigError(f"{name} must be between {minimum} and {maximum}")
