"""Central configuration for nmem-immune.

All settings are env-var driven with the ``NMEM_IMMUNE_`` prefix. Each flag has ONE
canonical name: the env var is always ``NMEM_IMMUNE_ + FIELD.upper()`` (no aliases).
Access via the module singleton: ``from nmem_immune import config; config.settings.skeptic_enabled``.
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NMEM_IMMUNE_", extra="ignore")

    # ── Database ──────────────────────────────────────────────────
    db_dsn: str | None = Field(
        None,
        description="Postgres DSN. ⚠ UNWIRED — read nowhere in-repo today; verify how immune gets its DSN.",
    )

    # ── Layer 1: Skeptic ─────────────────────────────────────────
    skeptic_enabled: bool = Field(
        True, description="Enable the skeptic layer (contradiction/poison screen). Default ON.")
    skeptic_text_overlap_threshold: float = Field(
        0.7, description="Text-overlap ratio above which two memories are near-duplicate.")
    skeptic_vector_divergence_threshold: float = Field(
        0.4, description="Embedding divergence above which overlapping content is treated as conflicting.")
    skeptic_trust_delta_threshold: float = Field(
        0.3, description="Trust level below which a source disagreement is flagged suspicious.")
    skeptic_poison_pattern_threshold: float = Field(
        0.6, description="Immunity score above which a memory matches a known poison pattern.")
    skeptic_scan_limit: int = Field(
        20, description="Max memories scanned per skeptic pass (silently clamped to ≤200).")

    # ── Layer 2: Quarantine ──────────────────────────────────────
    quarantine_aging_days: int = Field(
        7, ge=1, description="Days a quarantined item ages before re-evaluation.")
    quarantine_corroboration_min_sources: int = Field(
        2, ge=1, description="Independent sources required to release from quarantine. ⚠ UNWIRED.")
    quarantine_expiry_days: int = Field(
        90, ge=1, description="Days after which a quarantined item expires.")

    # ── Layer 3: Drift ───────────────────────────────────────────
    drift_sample_size: int = Field(
        10, description="Memories sampled per drift check (silently clamped to ≤100).")
    drift_stale_days: int = Field(
        30, description="Age at which a memory becomes drift-eligible. ⚠ UNWIRED.")
    drift_compatibility_threshold: float = Field(
        0.6, description="Similarity below which a memory is considered drifted from its cluster.")

    # ── Layer 4: Antidote ────────────────────────────────────────
    antidote_max_depth: int = Field(
        5, description="Max provenance-propagation depth when neutralizing a poisoned memory.")
    antidote_confidence_slash: float = Field(
        0.5, description="Confidence multiplier applied to antidoted memories. ⚠ UNWIRED.")

    # ── Layer 5: Immunity ────────────────────────────────────────
    immunity_decay_rate: float = Field(
        0.05, description="Per-cycle immunity decay. ⚠ UNWIRED.")
    immunity_min_confidence: float = Field(
        0.1, description="Confidence floor below which a memory is retired.")
    immunity_retirement_threshold: float = Field(
        0.1, description="Confidence at/below which a memory retires.")

    # Preserve today's silent-clamp behaviour exactly (min(x, N)), not a hard le=N reject.
    @field_validator("skeptic_scan_limit")
    @classmethod
    def _cap_scan_limit(cls, v: int) -> int:
        return min(v, 200)

    @field_validator("drift_sample_size")
    @classmethod
    def _cap_sample_size(cls, v: int) -> int:
        return min(v, 100)


settings = Settings()

# ── Non-field constants (not env-driven; stay module-level) ──────────────

# Trust by grounding value (matches nmem's grounding column:
# source_material, confirmed, inferred, disputed)
GROUNDING_TRUST: dict[str, float] = {
    "source_material": 1.0,
    "confirmed": 0.8,
    "inferred": 0.4,
    "disputed": 0.2,
}

# Trust by source type (matches nmem's source column on LTM/shared)
SOURCE_TRUST: dict[str, float] = {
    "system": 1.0,
    "agent": 0.7,
    "promotion": 0.8,
    "consolidation": 0.8,
    "migration": 0.6,
    "external": 0.4,
}
