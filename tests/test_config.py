from pathlib import Path

import pytest

from tg_insight.config import ConfigError, Settings, parse_peer


def valid_env() -> dict[str, str]:
    return {
        "TG_API_ID": "12345",
        "TG_API_HASH": "hash",
        "TG_BOT_TOKEN": "123:token",
        "TG_SOURCE_CHATS": "@group,-1001234567890",
        "TG_SUMMARY_TARGET": "42",
        "TG_ALLOWED_USER_IDS": "42,43",
        "LLM_API_KEY": "key",
        "LLM_MODEL": "model",
        "DATA_DIR": "/tmp/tg-insight-test",
    }


def test_settings_parse_required_fields() -> None:
    settings = Settings.from_env(valid_env())
    assert settings.source_chats == ("@group", -1001234567890)
    assert settings.summary_target == 42
    assert settings.allowed_user_ids == frozenset({42, 43})
    assert settings.data_dir == Path("/tmp/tg-insight-test")
    assert settings.timezone == "Asia/Shanghai"
    assert settings.archive_max_gb == 10
    assert settings.archive_max_bytes == 10 * 1024 * 1024 * 1024
    assert settings.llm_fallback_model is None


def test_settings_parses_distinct_fallback_model() -> None:
    env = valid_env()
    env["LLM_FALLBACK_MODEL"] = "fallback"
    assert Settings.from_env(env).llm_fallback_model == "fallback"


def test_settings_rejects_duplicate_fallback_model() -> None:
    env = valid_env()
    env["LLM_FALLBACK_MODEL"] = env["LLM_MODEL"]
    with pytest.raises(ConfigError, match="LLM_FALLBACK_MODEL"):
        Settings.from_env(env)


def test_settings_parses_archive_size_limit() -> None:
    env = valid_env()
    env["ARCHIVE_MAX_GB"] = "8"
    assert Settings.from_env(env).archive_max_bytes == 8 * 1024 * 1024 * 1024


def test_settings_allows_bot_selected_source_chats() -> None:
    env = valid_env()
    env["TG_SOURCE_CHATS"] = ""
    assert Settings.from_env(env).source_chats == ()


def test_settings_reject_empty_allowlist() -> None:
    env = valid_env()
    env["TG_ALLOWED_USER_IDS"] = ""
    with pytest.raises(ConfigError, match="TG_ALLOWED_USER_IDS"):
        Settings.from_env(env)


def test_settings_reject_bad_schedule() -> None:
    env = valid_env()
    env["SUMMARY_HOUR"] = "24"
    with pytest.raises(ConfigError, match="SUMMARY_HOUR"):
        Settings.from_env(env)


def test_settings_rejects_cleartext_llm_endpoint() -> None:
    env = valid_env()
    env["LLM_BASE_URL"] = "http://llm.example/v1"
    with pytest.raises(ConfigError, match="HTTPS"):
        Settings.from_env(env)


def test_settings_accepts_https_llm_endpoint() -> None:
    env = valid_env()
    env["LLM_BASE_URL"] = "https://llm.example/v1"
    assert Settings.from_env(env).llm_base_url == "https://llm.example/v1"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RETENTION_DAYS", "0"),
        ("ARCHIVE_MAX_MESSAGES", "999"),
        ("ARCHIVE_MAX_GB", "0"),
        ("ARCHIVE_MIN_FREE_MB", "0"),
        ("SUMMARY_MAX_CHARS", "9999"),
        ("SUMMARY_MAX_ATTEMPTS", "6"),
    ],
)
def test_settings_rejects_unsafe_resource_limits(name: str, value: str) -> None:
    env = valid_env()
    env[name] = value
    with pytest.raises(ConfigError, match=name):
        Settings.from_env(env)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("-100123", -100123), ("123", 123), ("@group", "@group")],
)
def test_parse_peer(raw: str, expected: int | str) -> None:
    assert parse_peer(raw) == expected
