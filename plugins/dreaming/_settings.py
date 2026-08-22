"""Typed, profile-scoped configuration for the Dreaming plugin."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


class DreamSettingsError(ValueError):
    """Raised when Dreaming profile settings are malformed."""


@dataclass(frozen=True, slots=True)
class DreamSettings:
    enabled: bool = False
    promotion_mode: str = "off"
    min_hours: float = 24.0
    min_sessions: int = 5
    quiet_minutes: int = 60
    lookback_days: int = 7
    rem_mode: str = "auto"
    rem_provider: str = "mistral"
    rem_model: str = "mistral-small-latest"
    extract_provider: str = ""
    extract_model: str = ""
    verify_provider: str = ""
    verify_model: str = ""
    llm_timeout_seconds: float = 60.0
    memory_char_limit: int = 12000
    min_score: float = 0.72
    max_promotions: int = 1


_TOP_LEVEL_KEYS = frozenset(
    {
        "enabled",
        "promotion_mode",
        "min_hours",
        "min_sessions",
        "quiet_minutes",
        "lookback_days",
        "rem",
        "extract",
        "verify",
        "llm_timeout_seconds",
        "memory_char_limit",
        "min_score",
        "max_promotions",
    }
)
_ROUTE_KEYS = frozenset({"provider", "model"})
_REM_KEYS = frozenset({"mode", "provider", "model"})


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DreamSettingsError(f"{path} must be a mapping")
    return value


def _known_keys(value: Mapping[str, Any], allowed: frozenset[str], path: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        rendered = ", ".join(sorted(str(key) for key in unknown))
        raise DreamSettingsError(f"unknown {path} setting(s): {rendered}")


def _strict_bool(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise DreamSettingsError(f"{path} must be a boolean")
    return value


def _string(value: Any, path: str, *, allow_blank: bool = False) -> str:
    if not isinstance(value, str):
        raise DreamSettingsError(f"{path} must be a string")
    normalized = value.strip()
    if not normalized and not allow_blank:
        raise DreamSettingsError(f"{path} must not be blank")
    return normalized


def _choice(value: Any, path: str, choices: frozenset[str]) -> str:
    normalized = _string(value, path).lower()
    if normalized not in choices:
        rendered = ", ".join(sorted(choices))
        raise DreamSettingsError(f"{path} must be one of: {rendered}")
    return normalized


def _finite_float(
    value: Any,
    path: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DreamSettingsError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise DreamSettingsError(
            f"{path} must be finite and between {minimum} and {maximum}"
        )
    return result


def _bounded_int(value: Any, path: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DreamSettingsError(f"{path} must be an integer")
    if not minimum <= value <= maximum:
        raise DreamSettingsError(f"{path} must be between {minimum} and {maximum}")
    return value


def from_mapping(raw: Mapping[str, Any] | None) -> DreamSettings:
    """Validate one ``plugins.entries.dreaming.settings`` mapping."""
    data = _mapping(raw if raw is not None else {}, "dreaming settings")
    _known_keys(data, _TOP_LEVEL_KEYS, "dreaming")

    rem = _mapping(data.get("rem", {}), "dreaming.rem")
    extract = _mapping(data.get("extract", {}), "dreaming.extract")
    verify = _mapping(data.get("verify", {}), "dreaming.verify")
    _known_keys(rem, _REM_KEYS, "dreaming.rem")
    _known_keys(extract, _ROUTE_KEYS, "dreaming.extract")
    _known_keys(verify, _ROUTE_KEYS, "dreaming.verify")

    defaults = DreamSettings()
    return DreamSettings(
        enabled=_strict_bool(data.get("enabled", defaults.enabled), "dreaming.enabled"),
        promotion_mode=_choice(
            data.get("promotion_mode", defaults.promotion_mode),
            "dreaming.promotion_mode",
            frozenset({"off", "shadow", "auto"}),
        ),
        min_hours=_finite_float(
            data.get("min_hours", defaults.min_hours),
            "dreaming.min_hours",
            minimum=0.0,
            maximum=8760.0,
        ),
        min_sessions=_bounded_int(
            data.get("min_sessions", defaults.min_sessions),
            "dreaming.min_sessions",
            minimum=0,
            maximum=100000,
        ),
        quiet_minutes=_bounded_int(
            data.get("quiet_minutes", defaults.quiet_minutes),
            "dreaming.quiet_minutes",
            minimum=0,
            maximum=10080,
        ),
        lookback_days=_bounded_int(
            data.get("lookback_days", defaults.lookback_days),
            "dreaming.lookback_days",
            minimum=1,
            maximum=3650,
        ),
        rem_mode=_choice(
            rem.get("mode", defaults.rem_mode),
            "dreaming.rem.mode",
            frozenset({"auto", "off"}),
        ),
        rem_provider=_string(
            rem.get("provider", defaults.rem_provider), "dreaming.rem.provider"
        ),
        rem_model=_string(rem.get("model", defaults.rem_model), "dreaming.rem.model"),
        extract_provider=_string(
            extract.get("provider", defaults.extract_provider),
            "dreaming.extract.provider",
            allow_blank=True,
        ),
        extract_model=_string(
            extract.get("model", defaults.extract_model),
            "dreaming.extract.model",
            allow_blank=True,
        ),
        verify_provider=_string(
            verify.get("provider", defaults.verify_provider),
            "dreaming.verify.provider",
            allow_blank=True,
        ),
        verify_model=_string(
            verify.get("model", defaults.verify_model),
            "dreaming.verify.model",
            allow_blank=True,
        ),
        llm_timeout_seconds=_finite_float(
            data.get("llm_timeout_seconds", defaults.llm_timeout_seconds),
            "dreaming.llm_timeout_seconds",
            minimum=0.001,
            maximum=300.0,
        ),
        memory_char_limit=_bounded_int(
            data.get("memory_char_limit", defaults.memory_char_limit),
            "dreaming.memory_char_limit",
            minimum=0,
            maximum=10_000_000,
        ),
        min_score=_finite_float(
            data.get("min_score", defaults.min_score),
            "dreaming.min_score",
            minimum=0.0,
            maximum=1.0,
        ),
        max_promotions=_bounded_int(
            data.get("max_promotions", defaults.max_promotions),
            "dreaming.max_promotions",
            minimum=0,
            maximum=1,
        ),
    )


def from_context(ctx: Any) -> DreamSettings:
    """Load settings from the current plugin context only."""
    values: dict[str, Any] = {}
    for key in _TOP_LEVEL_KEYS:
        marker = object()
        value = ctx.get_config(key, marker)
        if value is not marker:
            values[key] = value
    return from_mapping(values)


def load_active_profile_settings() -> DreamSettings:
    """Load settings from the active profile's resolved ``config.yaml``."""
    from hermes_cli.config import load_config_readonly

    config = load_config_readonly()
    plugins = config.get("plugins", {})
    entries = plugins.get("entries", {}) if isinstance(plugins, Mapping) else {}
    entry = entries.get("dreaming", {}) if isinstance(entries, Mapping) else {}
    raw = entry.get("settings", {}) if isinstance(entry, Mapping) else {}
    return from_mapping(raw)


def resolve(settings: DreamSettings | None) -> DreamSettings:
    """Use an explicit immutable snapshot or load the active profile once."""
    return settings if settings is not None else load_active_profile_settings()
