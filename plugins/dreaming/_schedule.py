"""
Dream scheduler — enqueue, scan, check, and run consolidation cycles.

State lives in {HERMES_HOME}/dreams/:
  staging.jsonl        one candidate per line, written by on_session_end/light sleep
  state.json           last_dream_at, last_message_id_seen, sessions_since_dream
  lock                 present while a cycle is running
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from . import _diary, _score

_DEFAULT_MIN_HOURS = 24
_DEFAULT_MIN_SESSIONS = 5
_DEFAULT_LOOKBACK_DAYS = 7
_DEFAULT_QUIET_MINUTES = 60
_DEFAULT_PROMOTE_THRESHOLD = 0.72
_DEFAULT_MAX_PROMOTIONS = 3


def _hermes_base(hermes_home: str | None = None) -> Path:
    return Path(hermes_home or os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


def _dreams_dir(hermes_home: str | None = None) -> Path:
    d = _hermes_base(hermes_home) / "dreams"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _staging_path(hermes_home: str | None = None) -> Path:
    return _dreams_dir(hermes_home) / "staging.jsonl"


def _state_path(hermes_home: str | None = None) -> Path:
    return _dreams_dir(hermes_home) / "state.json"


def _lock_path(hermes_home: str | None = None) -> Path:
    return _dreams_dir(hermes_home) / "lock"


def _state_db_path(hermes_home: str | None = None) -> Path:
    return _hermes_base(hermes_home) / "state.db"


def _read_state(hermes_home: str | None = None) -> dict:
    p = _state_path(hermes_home)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {
                    "last_dream_at": float(data.get("last_dream_at", 0.0) or 0.0),
                    "sessions_since_dream": int(data.get("sessions_since_dream", 0) or 0),
                    "last_message_id_seen": int(data.get("last_message_id_seen", 0) or 0),
                    "last_candidates_scanned": int(data.get("last_candidates_scanned", 0) or 0),
                    "last_promoted": int(data.get("last_promoted", 0) or 0),
                    "last_skipped_meta": int(data.get("last_skipped_meta", 0) or 0),
                    "last_error": data.get("last_error"),
                }
        except Exception:
            pass
    return {
        "last_dream_at": 0.0,
        "sessions_since_dream": 0,
        "last_message_id_seen": 0,
        "last_candidates_scanned": 0,
        "last_promoted": 0,
        "last_skipped_meta": 0,
        "last_error": None,
    }


def _write_state(state: dict, hermes_home: str | None = None) -> None:
    _state_path(hermes_home).write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def enqueue_session(
    transcript: list[dict[str, str]],
    *,
    hermes_home: str | None = None,
    session_id: str | None = None,
) -> int:
    """
    Extract candidate memories from a session transcript and append to staging.jsonl.
    Returns the number of candidates written.
    """
    now = time.time()
    candidates: list[dict] = []
    seen: set[str] = set()

    for turn in transcript:
        role = turn.get("role", "")
        content = _coerce_content(turn.get("content", "")).strip()
        if not content or role not in ("user", "assistant"):
            continue
        for sentence in _split_sentences(content):
            candidate = _candidate_from_sentence(sentence, role=role, now=now, session_id=session_id)
            if not candidate:
                continue
            key = candidate["hash"]
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)

    if not candidates:
        return 0

    _append_candidates(candidates, hermes_home=hermes_home)
    state = _read_state(hermes_home)
    state["sessions_since_dream"] = state.get("sessions_since_dream", 0) + 1
    _write_state(state, hermes_home)
    return len(candidates)


def light_sleep_scan(
    *,
    hermes_home: str | None = None,
    lookback_days: int | None = None,
) -> dict[str, Any]:
    """Scan the profile state.db for recent messages and stage candidates."""
    db_path = _state_db_path(hermes_home)
    if not db_path.exists():
        return {"messages_seen": 0, "candidates_staged": 0, "last_message_id_seen": _read_state(hermes_home).get("last_message_id_seen", 0)}

    state = _read_state(hermes_home)
    last_seen = int(state.get("last_message_id_seen", 0) or 0)
    lookback = lookback_days if lookback_days is not None else _env_int("HERMES_DREAM_LOOKBACK_DAYS", _DEFAULT_LOOKBACK_DAYS)
    cutoff = time.time() - max(1, lookback) * 86400

    rows: list[tuple[int, str, str, str, float]] = []
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        query = """
            SELECT id, session_id, role, content, timestamp
            FROM messages
            WHERE active = 1
              AND id > ?
              AND timestamp >= ?
              AND role IN ('user', 'assistant')
              AND COALESCE(content, '') != ''
            ORDER BY id ASC
        """
        for row in conn.execute(query, (last_seen, cutoff)):
            rows.append((int(row["id"]), row["session_id"], row["role"], row["content"], float(row["timestamp"])))

    candidates: list[dict] = []
    seen_in_scan: set[str] = set()
    max_id = last_seen
    session_ids: set[str] = set()
    for msg_id, session_id, role, content, ts in rows:
        max_id = max(max_id, msg_id)
        session_ids.add(session_id)
        for sentence in _split_sentences(_coerce_content(content)):
            candidate = _candidate_from_sentence(sentence, role=role, now=ts, session_id=session_id, message_id=msg_id)
            if not candidate:
                continue
            key = candidate["hash"]
            if key in seen_in_scan:
                continue
            seen_in_scan.add(key)
            candidates.append(candidate)

    if candidates:
        _append_candidates(candidates, hermes_home=hermes_home)

    if max_id > last_seen:
        state["last_message_id_seen"] = max_id
        state["sessions_since_dream"] = state.get("sessions_since_dream", 0) + len(session_ids)
        _write_state(state, hermes_home)

    return {"messages_seen": len(rows), "candidates_staged": len(candidates), "last_message_id_seen": max_id}


def latest_activity_at(*, hermes_home: str | None = None) -> float:
    """Return timestamp of the latest active user/assistant message in state.db."""
    db_path = _state_db_path(hermes_home)
    if not db_path.exists():
        return 0.0
    try:
        with sqlite3.connect(str(db_path)) as conn:
            value = conn.execute("SELECT MAX(timestamp) FROM messages WHERE active = 1").fetchone()[0]
        return float(value or 0.0)
    except Exception:
        return 0.0


def is_quiet(*, hermes_home: str | None = None, quiet_minutes: int | None = None, now: float | None = None) -> bool:
    quiet = quiet_minutes if quiet_minutes is not None else _env_int("HERMES_DREAM_QUIET_MINUTES", _DEFAULT_QUIET_MINUTES)
    latest = latest_activity_at(hermes_home=hermes_home)
    if latest <= 0:
        return True
    return ((now or time.time()) - latest) >= max(0, quiet) * 60


def dream_check(
    *,
    hermes_home: str | None = None,
    min_hours: float = _DEFAULT_MIN_HOURS,
    min_sessions: int = _DEFAULT_MIN_SESSIONS,
) -> bool:
    """Return True if conditions are met to run a dream cycle."""
    if _lock_path(hermes_home).exists():
        return False
    staging = _staging_path(hermes_home)
    if not staging.exists() or staging.stat().st_size == 0:
        return False
    state = _read_state(hermes_home)
    hours_since = (time.time() - state["last_dream_at"]) / 3600
    return hours_since >= min_hours and state.get("sessions_since_dream", 0) >= min_sessions


def dream_run(
    *,
    hermes_home: str | None = None,
    memory_path: Path | None = None,
    force: bool = False,
    preview: bool = False,
    respect_quiet: bool = False,
    scan_recent: bool = True,
    llm: Any = None,
) -> dict[str, Any]:
    """
    Run one full dream cycle. Returns a summary dict.

    Light Sleep scans/stages candidates, REM writes DREAMS.md, and Deep Sleep
    promotes high-scoring entries unless ``preview`` is true.
    """
    lock = _lock_path(hermes_home)
    if lock.exists() and not force:
        raise RuntimeError("dream cycle already running")
    lock.touch()
    now = time.time()

    try:
        if respect_quiet and not is_quiet(hermes_home=hermes_home, now=now):
            result = _summary("skipped_quiet")
            result["latest_activity_at"] = latest_activity_at(hermes_home=hermes_home)
            _record_result(result, hermes_home=hermes_home, now=now)
            return result

        light = light_sleep_scan(hermes_home=hermes_home) if scan_recent else {"messages_seen": 0, "candidates_staged": 0}

        staging = _staging_path(hermes_home)
        raw = _load_staged_candidates(staging)
        if not raw:
            result = _summary("no_candidates")
            result.update({"light_sleep": light})
            _record_result(result, hermes_home=hermes_home, now=now)
            return result

        scored = [(c, _score.score(c, now)) for c in raw]
        scored.sort(key=lambda x: x[1], reverse=True)

        narrative = _rem_narrative([c for c, _ in scored[:30]], llm=llm)

        min_score = _env_float("HERMES_DREAM_MIN_SCORE", _DEFAULT_PROMOTE_THRESHOLD)
        max_promotions = _env_int("HERMES_DREAM_MAX_PROMOTIONS", _DEFAULT_MAX_PROMOTIONS)
        promoted: list[str] = []
        skipped_meta: list[str] = []
        skipped_low_score: list[str] = []

        for candidate, score in scored:
            text = candidate["text"]
            if _score.is_meta_entry(text) or _is_volatile_or_sensitive(text):
                skipped_meta.append(text)
                continue
            if score < min_score:
                skipped_low_score.append(text)
                continue
            if len(promoted) < max_promotions:
                promoted.append(text)

        if memory_path is None:
            memory_path = _hermes_base(hermes_home) / "memories" / "MEMORY.md"

        if promoted and not preview:
            _write_to_memory(promoted, memory_path)

        _diary.append_entry(narrative, promoted if not preview else [], skipped_meta, hermes_home=hermes_home)

        if not preview:
            staging.write_text("", encoding="utf-8")

        result = {
            "status": "preview" if preview else "complete",
            "light_sleep": light,
            "candidates_scanned": len(raw),
            "promoted": 0 if preview else len(promoted),
            "would_promote": len(promoted),
            "skipped_meta": len(skipped_meta),
            "skipped_low_score": len(skipped_low_score),
            "narrative_length": len(narrative),
            "memory_path": str(memory_path),
            "diary_path": str(_hermes_base(hermes_home) / "DREAMS.md"),
        }
        _record_result(result, hermes_home=hermes_home, now=now, reset_sessions=not preview)
        return result

    except Exception as exc:
        state = _read_state(hermes_home)
        state["last_error"] = str(exc)
        _write_state(state, hermes_home)
        raise
    finally:
        try:
            lock.unlink(missing_ok=True)
        except Exception:
            pass


def _summary(status: str) -> dict[str, Any]:
    return {
        "status": status,
        "candidates_scanned": 0,
        "promoted": 0,
        "would_promote": 0,
        "skipped_meta": 0,
        "skipped_low_score": 0,
        "narrative_length": 0,
    }


def _record_result(result: dict[str, Any], *, hermes_home: str | None, now: float, reset_sessions: bool = False) -> None:
    state = _read_state(hermes_home)
    state["last_dream_at"] = now
    state["last_candidates_scanned"] = int(result.get("candidates_scanned", 0) or 0)
    state["last_promoted"] = int(result.get("promoted", 0) or 0)
    state["last_skipped_meta"] = int(result.get("skipped_meta", 0) or 0)
    state["last_error"] = None if result.get("status") != "skipped_quiet" else "skipped: user active inside quiet window"
    if reset_sessions:
        state["sessions_since_dream"] = 0
    _write_state(state, hermes_home)


def _load_staged_candidates(staging: Path) -> list[dict]:
    raw: list[dict] = []
    by_hash: dict[str, dict] = {}
    if not staging.exists():
        return raw
    for line in staging.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        h = rec.get("hash") or hashlib.sha1(str(rec.get("text", "")).lower().encode()).hexdigest()
        if h in by_hash:
            existing = by_hash[h]
            existing["frequency"] = int(existing.get("frequency", 1) or 1) + int(rec.get("frequency", 1) or 1)
            sessions = set(existing.get("session_ids", [])) | set(rec.get("session_ids", []))
            if rec.get("session_id"):
                sessions.add(rec["session_id"])
            existing["session_ids"] = sorted(sessions)
            existing["query_count"] = max(int(existing.get("query_count", 1) or 1), len(sessions) or 1)
            continue
        rec["hash"] = h
        sessions = set(rec.get("session_ids", []))
        if rec.get("session_id"):
            sessions.add(rec["session_id"])
        rec["session_ids"] = sorted(sessions)
        rec["query_count"] = max(int(rec.get("query_count", 1) or 1), len(sessions) or 1)
        by_hash[h] = rec
        raw.append(rec)
    return raw


def _append_candidates(candidates: list[dict], *, hermes_home: str | None = None) -> None:
    staging = _staging_path(hermes_home)
    with staging.open("a", encoding="utf-8") as fh:
        for c in candidates:
            fh.write(json.dumps(c, sort_keys=True) + "\n")


def _candidate_from_sentence(sentence: str, *, role: str, now: float, session_id: str | None = None, message_id: int | None = None) -> dict | None:
    sentence = " ".join(sentence.strip().split())
    if not _looks_like_memory_candidate(sentence):
        return None
    key = hashlib.sha1(sentence.lower().encode()).hexdigest()
    candidate = {
        "text": sentence,
        "hash": key,
        "role": role,
        "created_at": now,
        "frequency": 1,
        "query_count": 1,
        "word_count": len(sentence.split()),
        "relevance": _heuristic_relevance(sentence),
    }
    if session_id:
        candidate["session_id"] = session_id
        candidate["session_ids"] = [session_id]
    if message_id is not None:
        candidate["message_id"] = message_id
    return candidate


def _looks_like_memory_candidate(sentence: str) -> bool:
    if len(sentence) < 24 or sentence.endswith("?"):
        return False
    lower = sentence.lower()
    volatile = [
        "commit ", "pr #", "pull request", "issue #", "branch ", "today i ",
        "we fixed", "i fixed", "pushed", "merged", "temporary", "todo:",
    ]
    if any(token in lower for token in volatile):
        return False
    durable = [
        "prefers", "wants", "expects", "uses", "runs", "is installed", "is located",
        "should", "always", "never", "profile", "configured", "api key", "gateway",
        "service", "memory", "skill", "workflow", "convention", "correction",
    ]
    return any(token in lower for token in durable) or len(sentence.split()) >= 12


def _heuristic_relevance(sentence: str) -> float:
    lower = sentence.lower()
    strong = ["prefers", "expects", "wants", "remember", "configured", "installed", "uses", "runs"]
    score = 0.55
    score += 0.10 * sum(1 for token in strong if token in lower)
    if _score.is_meta_entry(sentence):
        score -= 0.25
    if _is_volatile_or_sensitive(sentence):
        score -= 0.35
    return max(0.05, min(0.95, score))


def _is_volatile_or_sensitive(text: str) -> bool:
    lower = text.lower()
    volatile = ["commit ", "sha", "pr #", "issue #", "fixed bug", "pushed branch", "merged branch"]
    sensitive = ["chastity", "therapy session", "therapist said", "raw trauma", "sexual"]
    return any(t in lower for t in volatile + sensitive)


def _rem_narrative(candidates: list[dict], *, llm: Any = None) -> str:
    """Extract themes and produce a narrative for the dream diary."""
    texts = [c["text"] for c in candidates if c.get("text")]
    if not texts:
        return "No candidates surfaced this cycle."

    rem_mode = os.environ.get("HERMES_DREAM_REM_MODE", "auto").strip().lower()
    if rem_mode != "off":
        if llm is not None:
            try:
                return _llm_narrative(texts, llm)
            except Exception:
                pass
        try:
            return _auxiliary_narrative(texts)
        except Exception:
            pass

    lines = ["**Themes surfaced this cycle:**\n"]
    for i, text in enumerate(texts[:10], 1):
        lines.append(f"{i}. {text[:120]}")
    return "\n".join(lines)


def _auxiliary_narrative(texts: list[str]) -> str:
    """Use Hermes' auxiliary LLM resolver for cron/script runs without ctx.llm."""
    from agent.auxiliary_client import call_llm

    prompt = _rem_prompt(texts)
    provider = os.environ.get("HERMES_DREAM_PROVIDER", "mistral").strip() or "mistral"
    model = os.environ.get("HERMES_DREAM_MODEL", "mistral-small-latest").strip() or "mistral-small-latest"
    timeout = _env_float("HERMES_DREAM_LLM_TIMEOUT", 60.0)
    response = call_llm(
        task="dreaming",
        provider=provider,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=512,
        timeout=timeout,
    )
    return response.choices[0].message.content.strip()


def _rem_prompt(texts: list[str]) -> str:
    return (
        "You are the REM Sleep phase of Hermes Dreaming. The following candidate facts "
        "were observed across recent sessions. Identify 3-5 recurring themes and write "
        "a concise dream diary entry. Do not invent facts. Flag meta-memory noise and "
        "volatile task-progress as review-only.\n\nFacts:\n"
        + "\n".join(f"- {t}" for t in texts[:30])
    )


def _llm_narrative(texts: list[str], llm: Any) -> str:
    """Call *llm* to produce a consolidation narrative. Raises on failure."""
    prompt = _rem_prompt(texts)
    provider = os.environ.get("HERMES_DREAM_PROVIDER", "mistral").strip() or None
    model = os.environ.get("HERMES_DREAM_MODEL", "mistral-small-latest").strip() or None
    timeout_raw = os.environ.get("HERMES_DREAM_LLM_TIMEOUT", "").strip()
    try:
        timeout = float(timeout_raw) if timeout_raw else None
    except ValueError:
        timeout = None

    result = llm.complete(
        [{"role": "user", "content": prompt}],
        provider=provider,
        model=model,
        max_tokens=512,
        temperature=0,
        timeout=timeout,
        purpose="dream_rem_narrative",
    )
    return result.text.strip()


def _write_to_memory(entries: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    existing_lower = existing.lower()
    unique = []
    for e in entries:
        compact = e[:200].strip()
        if compact and compact.lower() not in existing_lower:
            unique.append(compact)
    if not unique:
        return
    new_lines = "\n".join(f"- {e}" for e in unique)
    separator = "\n\n<!-- dreaming -->\n"
    with path.open("a", encoding="utf-8") as fh:
        if separator.strip() not in existing:
            fh.write(separator)
        elif not existing.endswith("\n"):
            fh.write("\n")
        fh.write(new_lines + "\n")


def _split_sentences(text: str) -> list[str]:
    import re
    parts = re.split(r"(?<=[.!])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _coerce_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default
