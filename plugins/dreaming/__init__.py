"""
Dreaming — automatic background memory consolidation for Hermes.

Reference implementation from Willow 2.0 (github.com/rudi193-cmd/willow-2.0).
Adapted to the Hermes plugin interface.

3-phase pipeline:
  Light Sleep  — scan recent transcripts, deduplicate, score candidates
  REM          — extract themes via Hermes' plugin LLM facade when available,
                 otherwise write a deterministic dream diary summary
  Deep Sleep   — promote high-signal entries to MEMORY.md; skip meta-entries
                 rather than polluting long-term memory

Opt-in: disabled by default. Enable with HERMES_DREAMING=1.
Optional REM model override: HERMES_DREAM_PROVIDER / HERMES_DREAM_MODEL
(subject to the plugin LLM trust gates in config.yaml).
Override thresholds: HERMES_DREAM_MIN_HOURS (default: 24),
                     HERMES_DREAM_MIN_SESSIONS (default: 5).
"""
from __future__ import annotations

import os
import threading
import time

from . import _diary, _schedule

_POLL_INTERVAL = int(os.environ.get("HERMES_DREAM_POLL_SECONDS", "300"))  # 5 min


# ---------------------------------------------------------------------------
# Background scheduler thread
# ---------------------------------------------------------------------------

class _DreamThread(threading.Thread):
    def __init__(self, *, llm=None) -> None:
        super().__init__(daemon=True, name="hermes-dreaming")
        self._llm = llm
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.wait(timeout=_POLL_INTERVAL):
            try:
                min_hours = float(os.environ.get("HERMES_DREAM_MIN_HOURS", "24"))
                min_sessions = int(os.environ.get("HERMES_DREAM_MIN_SESSIONS", "5"))
                if _schedule.dream_check(min_hours=min_hours, min_sessions=min_sessions):
                    _schedule.dream_run(llm=self._llm)
            except Exception:
                pass  # never crash the daemon thread

    def stop(self) -> None:
        self._stop_event.set()


_thread: _DreamThread | None = None


# ---------------------------------------------------------------------------
# Hook handlers
# ---------------------------------------------------------------------------

def _on_session_end(ctx) -> None:
    """Enqueue the completed session's transcript for consolidation."""
    transcript = getattr(ctx, "transcript", None) or []
    if not transcript:
        return
    try:
        _schedule.enqueue_session(transcript)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Slash command handler
# ---------------------------------------------------------------------------

_HELP = """\
/dream            — show status and last diary entry
/dream run        — force a consolidation cycle now
/dream status     — check conditions (hours since last, sessions queued)
/dream diary      — show the last dream diary entry
"""


def _handle_slash(raw_args: str, ctx=None) -> str:
    argv = raw_args.strip().split() if isinstance(raw_args, str) else list(raw_args or [])
    sub = argv[0] if argv else ""

    if not argv:
        state = _schedule._read_state()
        hours_ago = (time.time() - state["last_dream_at"]) / 3600
        sessions = state.get("sessions_since_dream", 0)
        last = _diary.last_entry()
        lines = [
            f"**Dreaming** | last cycle: {hours_ago:.1f}h ago | "
            f"sessions queued: {sessions}",
        ]
        if last:
            lines += ["", last]
        return "\n".join(lines)

    if sub == "run":
        try:
            result = _schedule.dream_run(force=True, llm=getattr(ctx, "llm", None) if ctx is not None else None)
            return (
                f"Dream cycle complete. "
                f"Scanned {result['candidates_scanned']} candidates, "
                f"promoted {result['promoted']}, "
                f"routed {result['skipped_meta']} meta-entries to SKILL.md."
            )
        except Exception as e:
            return f"Dream cycle failed: {e}"

    if sub == "status":
        state = _schedule._read_state()
        hours_ago = (time.time() - state["last_dream_at"]) / 3600
        sessions = state.get("sessions_since_dream", 0)
        ready = _schedule.dream_check()
        return (
            f"Hours since last dream: {hours_ago:.1f}\n"
            f"Sessions since last dream: {sessions}\n"
            f"Ready to dream: {'yes' if ready else 'no'}"
        )

    if sub == "diary":
        entry = _diary.last_entry()
        return entry if entry else "No dream diary entries yet."

    return _HELP


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    global _thread

    if not os.environ.get("HERMES_DREAMING", "").strip().lower() in ("1", "true", "yes"):
        # Opt-in only. Register the command so users can discover it,
        # but don't start the background thread or hook session_end.
        ctx.register_command(
            "dream",
            handler=lambda raw_args: (
                "Dreaming is disabled. Set HERMES_DREAMING=1 to enable.\n\n" + _HELP
            ),
            description="Automatic memory consolidation (disabled — set HERMES_DREAMING=1).",
        )
        return

    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_command(
        "dream",
        handler=lambda raw_args: _handle_slash(raw_args, ctx),
        description="Automatic background memory consolidation — status, force run, diary.",
    )

    _thread = _DreamThread(llm=getattr(ctx, "llm", None))
    _thread.start()
