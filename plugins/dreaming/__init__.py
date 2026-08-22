"""
Dreaming — automatic nightly memory consolidation for Hermes.

3-phase pipeline:
  Light Sleep  — scan recent profile transcripts, deduplicate, stage candidates
  REM Sleep    — extract themes via Hermes' plugin LLM facade; write DREAMS.md
  Deep Sleep   — promote high-signal entries to MEMORY.md; skip meta-entries
                 rather than polluting long-term memory

Opt-in: disabled by default. Enable in this profile's plugin settings.
Nightly schedule is handled by Hermes cron; the plugin provides the session
hook and /dream commands.
"""
from __future__ import annotations

import time

from . import _diary, _preference_semantics, _schedule, _settings


# ---------------------------------------------------------------------------
# Hook handlers
# ---------------------------------------------------------------------------

def _on_session_end(**_: object) -> None:
    """Session-end hook placeholder.

    Hermes' session_end hook currently provides metadata (session_id,
    completed, platform, etc.) rather than the transcript. The nightly Light
    Sleep phase scans the profile SQLite session store directly, so this hook
    intentionally does no work beyond staying compatible with the hook API.
    """
    return


# ---------------------------------------------------------------------------
# Slash command handler
# ---------------------------------------------------------------------------

_HELP = """\
/dream            — show status and last diary entry
/dream run        — force a consolidation cycle now
/dream preview    — run Light+REM+Deep scoring without writing MEMORY.md
/dream status     — check conditions, quiet window, routes, and last run stats
/dream diary      — show the last dream diary entry

Memory writes are gated by the profile's promotion_mode (off | shadow | auto),
which defaults to off. In shadow the full two-model pipeline runs and reports
would-promote counts without touching MEMORY.md; only auto ever writes.
"""


def _route_display(provider: str, model: str) -> str:
    provider = provider or "(unset)"
    model = model or "(unset)"
    return f"{provider} / {model}"


def _format_result(result: dict) -> str:
    status = result.get("status", "complete")
    mode = result.get("promotion_mode", "off")
    promoted = result.get("promoted", 0)
    would = result.get("would_promote", promoted)
    if status == "preview":
        promoted_text = f"would promote {would}"
    elif mode == "shadow":
        promoted_text = f"would promote {would} (shadow — no memory writes)"
    else:
        promoted_text = f"promoted {promoted}"
    return (
        f"Dream cycle {status}. "
        f"Scanned {result.get('candidates_scanned', 0)} candidates, "
        f"{promoted_text}, "
        f"skipped {result.get('skipped_meta', 0)} meta/sensitive entries."
    )


def _handle_slash(
    raw_args: str,
    ctx=None,
    *,
    settings: _settings.DreamSettings | None = None,
) -> str:
    resolved = _settings.resolve(settings)
    argv = raw_args.strip().split() if isinstance(raw_args, str) else list(raw_args or [])
    sub = argv[0] if argv else ""
    llm = getattr(ctx, "llm", None) if ctx is not None else None

    if not argv:
        state = _schedule._read_state()
        hours_ago = (time.time() - state["last_dream_at"]) / 3600 if state["last_dream_at"] else 9999
        sessions = state.get("sessions_since_dream", 0)
        last = _diary.last_entry()
        lines = [
            f"**Dreaming** | last cycle: {hours_ago:.1f}h ago | sessions queued: {sessions}",
            f"Last promoted: {state.get('last_promoted', 0)} | last skipped: {state.get('last_skipped_meta', 0)}",
        ]
        if last:
            lines += ["", last]
        return "\n".join(lines)

    if sub == "run":
        try:
            result = _schedule.dream_run(
                force=True,
                preview=False,
                respect_quiet=False,
                scan_recent=True,
                llm=llm,
                settings=resolved,
            )
            return _format_result(result)
        except Exception as e:
            return f"Dream cycle failed: {e}"

    if sub == "preview":
        try:
            result = _schedule.dream_run(
                force=True,
                preview=True,
                respect_quiet=False,
                scan_recent=True,
                llm=llm,
                settings=resolved,
            )
            return _format_result(result)
        except Exception as e:
            return f"Dream preview failed: {e}"

    if sub == "status":
        state = _schedule._read_state()
        hours_ago = (time.time() - state["last_dream_at"]) / 3600 if state["last_dream_at"] else 9999
        sessions = state.get("sessions_since_dream", 0)
        ready = _schedule.dream_check(
            min_hours=resolved.min_hours,
            min_sessions=resolved.min_sessions,
        )
        quiet = _schedule.is_quiet(quiet_minutes=resolved.quiet_minutes)
        latest = _schedule.latest_activity_at()
        return (
            f"Enabled: {'yes' if resolved.enabled else 'no'}\n"
            f"Promotion mode: {_preference_semantics.promotion_mode(resolved)}\n"
            f"REM provider/model: {_route_display(resolved.rem_provider, resolved.rem_model)}\n"
            f"Extract provider/model: {_route_display(resolved.extract_provider, resolved.extract_model)}\n"
            f"Verify provider/model: {_route_display(resolved.verify_provider, resolved.verify_model)}\n"
            f"Hours since last dream: {hours_ago:.1f}\n"
            f"Sessions since last dream: {sessions}\n"
            f"Quiet window satisfied: {'yes' if quiet else 'no'}\n"
            f"Latest activity timestamp: {latest:.0f}\n"
            f"Ready to dream: {'yes' if ready else 'no'}\n"
            f"Last candidates: {state.get('last_candidates_scanned', 0)}\n"
            f"Last promoted: {state.get('last_promoted', 0)}\n"
            f"Last error: {state.get('last_error') or 'none'}"
        )

    if sub == "diary":
        entry = _diary.last_entry()
        return entry if entry else "No dream diary entries yet."

    return _HELP


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    try:
        settings = _settings.from_context(ctx)
    except _settings.DreamSettingsError as exc:
        ctx.register_command(
            "dream",
            handler=lambda _raw_args: (
                f"Dreaming disabled: invalid profile settings ({exc})."
            ),
            description="Dreaming is disabled by invalid profile settings.",
        )
        return

    if not settings.enabled:
        ctx.register_command(
            "dream",
            handler=lambda _raw_args: (
                "Dreaming is disabled in this profile's plugin settings.\n\n" + _HELP
            ),
            description="Automatic nightly memory consolidation (disabled).",
        )
        return

    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_command(
        "dream",
        handler=lambda raw_args: _handle_slash(raw_args, ctx, settings=settings),
        description="Automatic nightly memory consolidation — status, force run, preview, diary.",
    )
