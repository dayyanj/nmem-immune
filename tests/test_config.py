"""Config-alignment guardrails for nmem-immune's pydantic Settings.

Covers the three acceptance criteria from docs/config-alignment-plan.md §5:
  1. defaults-parity   — empty env reproduces today's effective values
  2. invariants        — clamps and ge=1 guards are preserved
  3. round-trip        — every field resolves from NMEM_IMMUNE_<FIELD>, no aliases
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nmem_immune.config import Settings


# 1. ── defaults-parity: the effective values before the migration ──────────
EXPECTED_DEFAULTS = {
    "db_dsn": None,
    "skeptic_enabled": True,            # the one default-ON flag
    "skeptic_text_overlap_threshold": 0.7,
    "skeptic_vector_divergence_threshold": 0.4,
    "skeptic_trust_delta_threshold": 0.3,
    "skeptic_poison_pattern_threshold": 0.6,
    "skeptic_scan_limit": 20,
    "quarantine_aging_days": 7,
    "quarantine_corroboration_min_sources": 2,
    "quarantine_expiry_days": 90,
    "drift_sample_size": 10,
    "drift_stale_days": 30,
    "drift_compatibility_threshold": 0.6,
    "antidote_max_depth": 5,
    "antidote_confidence_slash": 0.5,
    "immunity_decay_rate": 0.05,
    "immunity_min_confidence": 0.1,
    "immunity_retirement_threshold": 0.1,
}


def test_defaults_match_pre_migration(monkeypatch):
    for key in list(EXPECTED_DEFAULTS) + ["NMEM_IMMUNE_" + k.upper() for k in EXPECTED_DEFAULTS]:
        monkeypatch.delenv(key, raising=False)
    s = Settings()
    for field, expected in EXPECTED_DEFAULTS.items():
        assert getattr(s, field) == expected, f"{field} default drifted"


def test_default_on_flag_stays_on(monkeypatch):
    monkeypatch.delenv("NMEM_IMMUNE_SKEPTIC_ENABLED", raising=False)
    assert Settings().skeptic_enabled is True


# 2. ── invariants: clamps preserved, ge=1 guards preserved ─────────────────
def test_scan_limit_clamps_not_rejects(monkeypatch):
    monkeypatch.setenv("NMEM_IMMUNE_SKEPTIC_SCAN_LIMIT", "500")
    assert Settings().skeptic_scan_limit == 200          # clamped, NOT a ValidationError


def test_sample_size_clamps_not_rejects(monkeypatch):
    monkeypatch.setenv("NMEM_IMMUNE_DRIFT_SAMPLE_SIZE", "999")
    assert Settings().drift_sample_size == 100


@pytest.mark.parametrize("env", [
    "NMEM_IMMUNE_QUARANTINE_AGING_DAYS",
    "NMEM_IMMUNE_QUARANTINE_CORROBORATION_MIN_SOURCES",
    "NMEM_IMMUNE_QUARANTINE_EXPIRY_DAYS",
])
def test_pos_int_guards_reject_below_one(monkeypatch, env):
    monkeypatch.setenv(env, "0")
    with pytest.raises(ValidationError):
        Settings()


# 3. ── round-trip: canonical env name resolves to the field (no aliases) ───
SAMPLE = {
    "bool": ("skeptic_enabled", "false", False),
    "int": ("antidote_max_depth", "9", 9),
    "float": ("immunity_min_confidence", "0.42", 0.42),
    "str": ("db_dsn", "postgres://x", "postgres://x"),
}


@pytest.mark.parametrize("field,raw,parsed", list(SAMPLE.values()), ids=list(SAMPLE))
def test_env_name_resolves_to_field(monkeypatch, field, raw, parsed):
    env = "NMEM_IMMUNE_" + field.upper()
    monkeypatch.setenv(env, raw)
    assert getattr(Settings(), field) == parsed


def test_bool_accepts_true_false_wire_format(monkeypatch):
    monkeypatch.setenv("NMEM_IMMUNE_SKEPTIC_ENABLED", "false")
    assert Settings().skeptic_enabled is False
    monkeypatch.setenv("NMEM_IMMUNE_SKEPTIC_ENABLED", "true")
    assert Settings().skeptic_enabled is True
