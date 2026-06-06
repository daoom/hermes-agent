from __future__ import annotations

import json
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


def test_rem_narrative_without_llm_uses_deterministic_fallback(monkeypatch):
    monkeypatch.delenv("HERMES_DREAM_REM_MODE", raising=False)
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.delenv("HERMES_DREAM_MODEL", raising=False)

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
    monkeypatch.setenv("HERMES_DREAM_PROVIDER", "openrouter")
    monkeypatch.setenv("HERMES_DREAM_MODEL", "anthropic/claude-sonnet-4")
    llm = RecordingLlm()

    text = _schedule._rem_narrative([
        {"text": "Recurring memory entries should be deduplicated."}
    ], llm=llm)

    assert text == "LLM narrative"
    assert len(llm.calls) == 1
    kwargs = llm.calls[0]["kwargs"]
    assert kwargs["provider"] == "openrouter"
    assert kwargs["model"] == "anthropic/claude-sonnet-4"
    assert kwargs["purpose"] == "dream_rem_narrative"


def test_dream_run_passes_llm_and_defaults_to_profile_memory_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_DREAM_REM_MODE", "auto")
    llm = RecordingLlm()

    staging = _schedule._staging_path(str(tmp_path))
    candidate = {
        "text": "Marc prefers memory entries to be concise and durable.",
        "hash": "abc",
        "role": "user",
        "created_at": time.time(),
        "frequency": 5,
        "query_count": 5,
        "word_count": 9,
    }
    staging.write_text(json.dumps(candidate) + "\n", encoding="utf-8")

    result = _schedule.dream_run(hermes_home=str(tmp_path), force=True, llm=llm)

    assert result["candidates_scanned"] == 1
    assert len(llm.calls) == 1
    memory_path = tmp_path / "memories" / "MEMORY.md"
    assert memory_path.exists()
    assert "Marc prefers memory entries" in memory_path.read_text(encoding="utf-8")
