from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from plugins.dreaming import _schedule


class _Result:
    text = "LLM narrative"


class RecordingLlm:
    def __init__(self) -> None:
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return _Result()


def _stage_candidate(tmp_path: Path, text: str, **overrides):
    candidate = {
        "text": text,
        "hash": overrides.pop("hash", "abc"),
        "role": overrides.pop("role", "user"),
        "created_at": overrides.pop("created_at", time.time()),
        "frequency": overrides.pop("frequency", 8),
        "query_count": overrides.pop("query_count", 6),
        "word_count": overrides.pop("word_count", len(text.split())),
        "relevance": overrides.pop("relevance", 0.95),
    }
    candidate.update(overrides)
    staging = _schedule._staging_path(str(tmp_path))
    with staging.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(candidate) + "\n")
    return candidate


def test_rem_narrative_without_llm_uses_deterministic_fallback(monkeypatch):
    monkeypatch.setenv("HERMES_DREAM_REM_MODE", "off")

    text = _schedule._rem_narrative([
        {"text": "Marc prefers durable memory to stay concise and useful."}
    ])

    assert "Themes surfaced this cycle" in text
    assert "Marc prefers durable memory" in text


def test_rem_narrative_off_mode_does_not_call_llm(monkeypatch):
    monkeypatch.setenv("HERMES_DREAM_REM_MODE", "off")
    llm = RecordingLlm()

    text = _schedule._rem_narrative([
        {"text": "The assistant should not promote stale task progress."}
    ], llm=llm)

    assert llm.calls == []
    assert "assistant should not promote" in text


def test_rem_narrative_uses_host_llm_when_supplied(monkeypatch):
    monkeypatch.setenv("HERMES_DREAM_REM_MODE", "auto")
    monkeypatch.setenv("HERMES_DREAM_PROVIDER", "mistral")
    monkeypatch.setenv("HERMES_DREAM_MODEL", "mistral-small-latest")
    llm = RecordingLlm()

    text = _schedule._rem_narrative([
        {"text": "Recurring memory entries should be deduplicated."}
    ], llm=llm)

    assert text == "LLM narrative"
    assert len(llm.calls) == 1
    kwargs = llm.calls[0]["kwargs"]
    assert kwargs["provider"] == "mistral"
    assert kwargs["model"] == "mistral-small-latest"
    assert kwargs["purpose"] == "dream_rem_narrative"


def test_dream_run_passes_llm_and_defaults_to_profile_memory_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_DREAM_REM_MODE", "auto")
    monkeypatch.setenv("HERMES_DREAM_MIN_SCORE", "0.72")
    monkeypatch.setenv("HERMES_DREAM_MAX_PROMOTIONS", "3")
    llm = RecordingLlm()

    _stage_candidate(tmp_path, "Marc prefers memory entries to be concise and durable.")

    result = _schedule.dream_run(hermes_home=str(tmp_path), force=True, scan_recent=False, llm=llm)

    assert result["candidates_scanned"] == 1
    assert len(llm.calls) == 1
    memory_path = tmp_path / "memories" / "MEMORY.md"
    assert memory_path.exists()
    assert "Marc prefers memory entries" in memory_path.read_text(encoding="utf-8")


def test_preview_mode_does_not_write_memory_or_clear_staging(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_DREAM_REM_MODE", "off")
    _stage_candidate(tmp_path, "Marc prefers preview mode to avoid durable memory writes.")

    result = _schedule.dream_run(hermes_home=str(tmp_path), force=True, preview=True, scan_recent=False)

    assert result["status"] == "preview"
    assert result["promoted"] == 0
    assert result["would_promote"] == 1
    assert not (tmp_path / "memories" / "MEMORY.md").exists()
    assert _schedule._staging_path(str(tmp_path)).read_text(encoding="utf-8").strip()


def test_meta_entries_are_skipped_not_promoted(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_DREAM_REM_MODE", "off")
    _stage_candidate(tmp_path, "The assistant should update MEMORY.md when memory capacity is full.", hash="meta")

    result = _schedule.dream_run(hermes_home=str(tmp_path), force=True, scan_recent=False)

    assert result["skipped_meta"] == 1
    assert result["promoted"] == 0
    assert not (tmp_path / "memories" / "MEMORY.md").exists()


def test_quiet_window_skips_nightly_run(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_DREAM_REM_MODE", "off")
    _stage_candidate(tmp_path, "Marc prefers nightly dreaming to respect active use of the gateway.")
    db = tmp_path / "state.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, timestamp REAL, active INTEGER)")
        conn.execute("INSERT INTO messages VALUES (1, 's', 'user', 'hello', ?, 1)", (time.time(),))

    result = _schedule.dream_run(hermes_home=str(tmp_path), force=True, respect_quiet=True, scan_recent=False)

    assert result["status"] == "skipped_quiet"
    assert (tmp_path / "memories" / "MEMORY.md").exists() is False


def test_light_sleep_scans_state_db_and_stages_candidates(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_DREAM_REM_MODE", "off")
    db = tmp_path / "state.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, timestamp REAL, active INTEGER)")
        conn.execute(
            "INSERT INTO messages VALUES (1, 's1', 'user', ?, ?, 1)",
            ("Marc prefers stable profile-local dreaming configuration for Friday.", time.time()),
        )

    result = _schedule.light_sleep_scan(hermes_home=str(tmp_path))

    assert result["messages_seen"] == 1
    assert result["candidates_staged"] == 1
    assert "profile-local dreaming" in _schedule._staging_path(str(tmp_path)).read_text(encoding="utf-8")
