#!/usr/bin/env python3
"""Cron-safe entrypoint for the Dreaming plugin."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
PROFILE_PLUGINS_DIR = PLUGIN_DIR.parent
if str(PROFILE_PLUGINS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(PROFILE_PLUGINS_DIR.parent))
if str(PROFILE_PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(PROFILE_PLUGINS_DIR))

from dreaming import _schedule, _settings  # noqa: E402


def main() -> int:
    hermes_home = os.environ.get("HERMES_HOME")
    try:
        settings = _settings.load_active_profile_settings()
    except _settings.DreamSettingsError as exc:
        print(f"Dreaming settings error: {exc}", file=sys.stderr)
        return 2

    if not settings.enabled:
        return 0

    result = _schedule.dream_run(
        hermes_home=hermes_home,
        force=False,
        preview=False,
        respect_quiet=True,
        scan_recent=True,
        llm=None,
        settings=settings,
    )
    status = result.get("status")
    promoted = int(result.get("promoted", 0) or 0)
    if status in {"skipped_quiet", "not_ready", "no_candidates"}:
        # Silent success: ordinary readiness outcomes should not spam cron.
        return 0
    if promoted > 0 or status not in {"complete", "no_candidates"}:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
