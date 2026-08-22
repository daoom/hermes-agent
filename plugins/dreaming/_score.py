"""Candidate scoring for the Deep Sleep promotion gate.

The score ranks candidates only after deterministic source and policy gates pass.
Every component is normalized to [0.0, 1.0]. Session support measures actual
independent observations; generated query counts are not treated as evidence.
"""
from __future__ import annotations

import math
import time
from typing import Any

_WEIGHTS = {
    "relevance": 0.28,
    "source_quality": 0.24,
    "durability": 0.18,
    "session_support": 0.12,
    "recency": 0.08,
    "novelty": 0.07,
    "conceptual_richness": 0.03,
}

_META_KEYWORDS = frozenset([
    "memory is full", "memory capacity", "memory management",
    "memory.md", "skill.md", "store in skill", "update skill",
    "memory limit", "memory overflow",
])


def is_meta_entry(text: str) -> bool:
    """Return True if this looks like a memory-management meta-entry."""
    lower = text.lower()
    return any(kw in lower for kw in _META_KEYWORDS)


def _unit(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def score(candidate: dict[str, Any], now: float | None = None) -> float:
    """Return a calibrated rank for a policy-eligible structured candidate."""
    now = now or time.time()
    relevance = _unit(candidate.get("relevance"), 0.5)
    source_default = 1.0 if candidate.get("role") == "user" else 0.0
    source_quality = _unit(candidate.get("source_quality"), source_default)
    durability = _unit(candidate.get("durability"), 0.8)

    sessions = int(candidate.get("session_count", 0) or 0)
    if sessions <= 0:
        sessions = len(set(candidate.get("session_ids", [])))
    independent_observations = max(1, sessions)
    session_support = min(
        1.0,
        0.4 + 0.6 * math.log1p(independent_observations - 1) / math.log1p(4),
    )

    age_seconds = max(0.0, now - float(candidate.get("created_at", now)))
    recency = math.exp(-(age_seconds / 86400) / 14)

    existing_similarity = _unit(candidate.get("consolidation"), 0.0)
    novelty = 1.0 - existing_similarity

    text = str(candidate.get("canonical_text") or candidate.get("text") or "")
    raw_wc = int(candidate.get("word_count", len(text.split())) or 0)
    conceptual_richness = min(1.0, math.log1p(max(0, raw_wc)) / math.log1p(80))

    components = {
        "relevance": relevance,
        "source_quality": source_quality,
        "durability": durability,
        "session_support": session_support,
        "recency": recency,
        "novelty": novelty,
        "conceptual_richness": conceptual_richness,
    }
    return sum(_WEIGHTS[name] * value for name, value in components.items())
