from __future__ import annotations

import time
from pathlib import Path


HEARTBEAT_PATH = Path("/tmp/tg-insight.heartbeat")
HEARTBEAT_MAX_AGE_SECONDS = 75


def mark_healthy(path: Path = HEARTBEAT_PATH) -> None:
    path.touch(exist_ok=True)


def clear_health(path: Path = HEARTBEAT_PATH) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def is_healthy(
    path: Path = HEARTBEAT_PATH,
    *,
    now: float | None = None,
    max_age_seconds: int = HEARTBEAT_MAX_AGE_SECONDS,
) -> bool:
    try:
        age = (time.time() if now is None else now) - path.stat().st_mtime
    except OSError:
        return False
    return 0 <= age <= max_age_seconds
