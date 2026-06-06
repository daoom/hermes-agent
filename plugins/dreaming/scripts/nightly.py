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

from dreaming import _schedule  # noqa: E402


def main() -> int:
    hermes_home = os.environ.get("HERMES_HOME")
    result = _schedule.dream_run(
        hermes_home=hermes_home,
        force=True,
        preview=False,
        respect_quiet=True,
        scan_recent=True,
        llm=None,
    )
    status = result.get("status")
    promoted = int(result.get("promoted", 0) or 0)
    if status == "skipped_quiet":
        # Silent success: cron should not spam while Marc is active.
        return 0
    if promoted > 0 or status not in {"complete", "no_candidates"}:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
