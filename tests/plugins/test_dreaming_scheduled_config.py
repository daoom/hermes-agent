from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from plugins import dreaming
from plugins.dreaming import _schedule, _settings
from plugins.dreaming.scripts import nightly


class RecordingContext:
    def __init__(self, settings: dict[str, object]) -> None:
        self.settings = settings
        self.commands: dict[str, Callable[[str], str]] = {}
        self.hooks: list[tuple[str, object]] = []
        self.llm = object()

    def get_config(self, key: str, default=None):
        current: object = self.settings
        for segment in key.split("."):
            if not isinstance(current, dict) or segment not in current:
                return default
            current = current[segment]
        return current

    def register_command(self, name: str, *, handler, description: str) -> None:
        self.commands[name] = handler

    def register_hook(self, name: str, handler) -> None:
        self.hooks.append((name, handler))


def _write_profile_config(home: Path, settings: object) -> None:
    payload = {
        "plugins": {
            "entries": {
                "dreaming": {
                    "settings": settings,
                }
            }
        }
    }
    (home / "config.yaml").write_text(json.dumps(payload), encoding="utf-8")


def _enabled_settings(**overrides: object) -> dict[str, object]:
    settings: dict[str, object] = {
        "enabled": True,
        "promotion_mode": "off",
        "min_hours": 24.0,
        "min_sessions": 3,
        "quiet_minutes": 30,
        "lookback_days": 30,
        "rem": {
            "mode": "off",
            "provider": "mistral",
            "model": "dreaming-rem",
        },
        "extract": {
            "provider": "mistral",
            "model": "dreaming-extract",
        },
        "verify": {
            "provider": "mistral",
            "model": "dreaming-verify",
        },
        "llm_timeout_seconds": 60.0,
        "memory_char_limit": 12000,
        "min_score": 0.62,
        "max_promotions": 1,
    }
    settings.update(overrides)
    return settings


def test_register_uses_profile_enablement_not_process_environment(monkeypatch):
    monkeypatch.setenv("HERMES_DREAMING", "1")
    ctx = RecordingContext({"enabled": False})

    dreaming.register(ctx)

    assert ctx.hooks == []
    assert "disabled" in ctx.commands["dream"]("").lower()


def test_two_plugin_contexts_keep_settings_isolated(monkeypatch):
    monkeypatch.delenv("HERMES_DREAMING", raising=False)
    enabled = RecordingContext(_enabled_settings(promotion_mode="shadow"))
    disabled = RecordingContext({"enabled": False, "promotion_mode": "off"})
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        _schedule,
        "dream_run",
        lambda **kwargs: calls.append(kwargs)
        or {
            "status": "complete",
            "promotion_mode": "shadow",
            "promoted": 0,
            "would_promote": 0,
            "candidates_scanned": 0,
            "skipped_meta": 0,
        },
    )

    dreaming.register(enabled)
    dreaming.register(disabled)
    enabled.commands["dream"]("run")

    assert [name for name, _handler in enabled.hooks] == ["on_session_end"]
    assert disabled.hooks == []
    assert "disabled" in disabled.commands["dream"]("").lower()
    assert len(calls) == 1
    settings = calls[0]["settings"]
    assert settings.promotion_mode == "shadow"
    assert settings.extract_model == "dreaming-extract"
    assert settings.verify_model == "dreaming-verify"


def test_nightly_disabled_profile_stops_before_scheduler(tmp_path, monkeypatch):
    _write_profile_config(tmp_path, {"enabled": False})
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_DREAMING", "1")
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        nightly._schedule,
        "dream_run",
        lambda **kwargs: calls.append(kwargs) or {"status": "complete"},
    )

    assert nightly.main() == 0
    assert calls == []


def test_nightly_passes_profile_readiness_and_never_forces(tmp_path, monkeypatch):
    _write_profile_config(
        tmp_path,
        _enabled_settings(min_hours=999.0, min_sessions=77, quiet_minutes=45),
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        nightly._schedule,
        "dream_run",
        lambda **kwargs: calls.append(kwargs)
        or {"status": "not_ready", "promoted": 0},
    )

    assert nightly.main() == 0
    assert len(calls) == 1
    assert calls[0]["force"] is False
    assert calls[0]["respect_quiet"] is True
    settings = calls[0]["settings"]
    assert settings.min_hours == 999.0
    assert settings.min_sessions == 77
    assert settings.quiet_minutes == 45


def test_nightly_malformed_settings_fail_before_scheduler(
    tmp_path, monkeypatch, capsys
):
    _write_profile_config(tmp_path, {"enabled": "yes", "promotion_mode": "auto"})
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        nightly._schedule,
        "dream_run",
        lambda **kwargs: calls.append(kwargs) or {"status": "complete"},
    )

    assert nightly.main() != 0
    assert calls == []
    assert "settings" in capsys.readouterr().err.lower()


def test_force_never_bypasses_an_existing_cycle_lock(tmp_path):
    lock = _schedule._lock_path(str(tmp_path))
    lock.write_text("owner", encoding="utf-8")

    with pytest.raises(RuntimeError, match="already running"):
        _schedule.dream_run(
            hermes_home=str(tmp_path),
            force=True,
            preview=True,
            scan_recent=False,
            settings=_settings.from_mapping(_enabled_settings()),
        )

    assert lock.read_text(encoding="utf-8") == "owner"


def test_existing_cycle_lock_blocks_before_light_sleep_scan(tmp_path, monkeypatch):
    lock = _schedule._lock_path(str(tmp_path))
    lock.write_text("owner", encoding="utf-8")
    _schedule._staging_path(str(tmp_path)).write_text("{}\n", encoding="utf-8")
    scans: list[dict[str, Any]] = []
    monkeypatch.setattr(
        _schedule,
        "light_sleep_scan",
        lambda **kwargs: scans.append(kwargs)
        or {"messages_seen": 0, "candidates_staged": 0},
    )

    with pytest.raises(RuntimeError, match="already running"):
        _schedule.dream_run(
            hermes_home=str(tmp_path),
            force=False,
            preview=False,
            scan_recent=True,
            settings=_settings.from_mapping(
                _enabled_settings(min_hours=0, min_sessions=0, quiet_minutes=0)
            ),
        )

    assert scans == []
    assert lock.read_text(encoding="utf-8") == "owner"


def test_existing_cycle_lock_blocks_before_quiet_window_check(tmp_path, monkeypatch):
    lock = _schedule._lock_path(str(tmp_path))
    lock.write_text("owner", encoding="utf-8")
    quiet_checks: list[dict[str, Any]] = []
    monkeypatch.setattr(
        _schedule,
        "is_quiet",
        lambda **kwargs: quiet_checks.append(kwargs) or False,
    )

    with pytest.raises(RuntimeError, match="already running"):
        _schedule.dream_run(
            hermes_home=str(tmp_path),
            force=False,
            preview=False,
            respect_quiet=True,
            scan_recent=True,
            settings=_settings.from_mapping(_enabled_settings()),
        )

    assert quiet_checks == []
    assert lock.read_text(encoding="utf-8") == "owner"


def test_skipped_quiet_does_not_advance_cycle_readiness_state(tmp_path, monkeypatch):
    initial = {
        "last_dream_at": 123.0,
        "sessions_since_dream": 9,
        "last_message_id_seen": 17,
        "last_candidates_scanned": 4,
        "last_promoted": 0,
        "last_skipped_meta": 2,
        "last_error": None,
    }
    _schedule._write_state(initial, str(tmp_path))
    monkeypatch.setattr(_schedule, "is_quiet", lambda **kwargs: False)

    result = _schedule.dream_run(
        hermes_home=str(tmp_path),
        force=False,
        preview=False,
        respect_quiet=True,
        scan_recent=True,
        settings=_settings.from_mapping(_enabled_settings()),
    )

    assert result["status"] == "skipped_quiet"
    assert _schedule._read_state(str(tmp_path)) == initial
    assert not _schedule._lock_path(str(tmp_path)).exists()
