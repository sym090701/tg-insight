from pathlib import Path

from tg_insight.health import clear_health, is_healthy, mark_healthy


def test_health_marker_is_fresh_only_within_maximum_age(tmp_path: Path) -> None:
    marker = tmp_path / "heartbeat"

    assert not is_healthy(marker, now=100, max_age_seconds=10)
    mark_healthy(marker)
    marker.touch()
    modified = marker.stat().st_mtime
    assert is_healthy(marker, now=modified + 10, max_age_seconds=10)
    assert not is_healthy(marker, now=modified + 11, max_age_seconds=10)

    clear_health(marker)
    assert not is_healthy(marker, now=modified, max_age_seconds=10)
