"""ImmuneBridge — Adapter between nmem (MemorySystem) and nmem-immune.

The ONLY coupling point between the two systems. Both work independently
without it. Lives entirely in nmem-immune's codebase. No nmem code changes
required.

Safety properties:
- No import-time coupling: nmem never imported
- Duck-typing: accepts any object with .on(), .consolidation
- Double error boundary: _safe() wraps everything, nmem also catches handler errors
- Fail-open: audit failures never propagate to nmem
- All behaviors individually toggleable via ImmuneBridgeConfig

Usage:
    pool = await asyncpg.create_pool("postgresql://...")
    bridge = ImmuneBridge(pool)
    bridge.connect(mem)  # registers hooks into nmem's lifecycle

    # Manual operations
    await bridge.mark_poisoned("nmem_long_term_memory", 42)
    print(bridge.stats)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ImmuneBridgeConfig:
    """All immune behaviors individually toggleable."""

    skeptic_on_journal_added: bool = True
    skeptic_on_ltm_saved: bool = True
    skeptic_on_shared_saved: bool = True
    drift_on_full_cycle: bool = True
    quarantine_review_on_full_cycle: bool = True
    immunity_decay_on_nightly: bool = True
    provenance_track_on_ltm_saved: bool = True


@dataclass
class ImmuneStats:
    """Observability counters for bridge operations."""

    entries_audited: int = 0
    entries_quarantined: int = 0
    entries_cleared: int = 0
    drift_checks_run: int = 0
    drift_issues_found: int = 0
    antidote_propagations: int = 0
    immunity_records_created: int = 0
    immunity_records_decayed: int = 0
    provenance_links_recorded: int = 0
    quarantine_promotions: int = 0
    errors: int = 0


class ImmuneBridge:
    """Adapter between nmem (MemorySystem) and nmem-immune.

    Opt-in coupling point. Neither system depends on this bridge.
    All operations are crash-safe — bridge failures never propagate to nmem.
    """

    def __init__(
        self,
        pool,  # asyncpg.Pool
        config: ImmuneBridgeConfig | None = None,
    ) -> None:
        self._pool = pool
        self._config = config or ImmuneBridgeConfig()
        self._memory: Any = None
        self._connected = False
        self._stats = ImmuneStats()

        # Lazy-init sub-systems
        self._skeptic = None
        self._quarantine = None
        self._drift = None
        self._antidote = None
        self._immunity = None

    @property
    def stats(self) -> ImmuneStats:
        return self._stats

    def connect(self, memory: Any) -> None:
        """Wire event handlers and consolidation hooks into nmem.

        This is the ONLY method that touches nmem's registration APIs.
        Called once at application startup.

        Args:
            memory: A MemorySystem instance (duck-typed, not type-checked).

        Raises:
            TypeError: If memory doesn't have the expected interface.
        """
        if self._connected:
            raise RuntimeError("ImmuneBridge.connect() already called — create a new instance to reconnect")
        if not hasattr(memory, "on"):
            raise TypeError("memory must have an .on() method (expected MemorySystem)")
        if not hasattr(memory, "consolidation"):
            raise TypeError("memory must have a .consolidation property")

        self._memory = memory
        self._init_subsystems()

        # ── Event handlers (post-write auditing) ─────────────

        if self._config.skeptic_on_journal_added:

            @memory.on("journal.added")
            async def _on_journal(data: dict) -> None:
                await self._safe("skeptic_journal", self._audit_entry(
                    data, "nmem_journal_entries",
                ))

        if self._config.skeptic_on_ltm_saved:

            @memory.on("ltm.saved")
            async def _on_ltm(data: dict) -> None:
                await self._safe("skeptic_ltm", self._audit_entry(
                    data, "nmem_long_term_memory",
                ))

        if self._config.skeptic_on_shared_saved:

            @memory.on("shared.saved")
            async def _on_shared(data: dict) -> None:
                await self._safe("skeptic_shared", self._audit_entry(
                    data, "nmem_shared_knowledge",
                ))

        if self._config.provenance_track_on_ltm_saved:

            @memory.on("ltm.saved")
            async def _on_ltm_provenance(data: dict) -> None:
                await self._safe("provenance", self._track_provenance(data))

        # ── Consolidation hooks ──────────────────────────────

        if self._config.drift_on_full_cycle:
            memory.consolidation.register_full_cycle_step(
                "immune_drift_detection",
                self._handle_drift_detection,
            )

        if self._config.quarantine_review_on_full_cycle:
            memory.consolidation.register_full_cycle_step(
                "immune_quarantine_review",
                self._handle_quarantine_review,
            )

        if self._config.immunity_decay_on_nightly:
            memory.consolidation.register_nightly_step(
                "immune_immunity_decay",
                self._handle_immunity_decay,
            )

        self._connected = True
        log.info(
            "ImmuneBridge connected (skeptic_j=%s, skeptic_ltm=%s, skeptic_sh=%s, "
            "drift=%s, qreview=%s, decay=%s)",
            self._config.skeptic_on_journal_added,
            self._config.skeptic_on_ltm_saved,
            self._config.skeptic_on_shared_saved,
            self._config.drift_on_full_cycle,
            self._config.quarantine_review_on_full_cycle,
            self._config.immunity_decay_on_nightly,
        )

    def _init_subsystems(self) -> None:
        """Lazily initialize sub-systems to avoid import-time coupling."""
        from nmem_immune.antidote import AntidotePropagator
        from nmem_immune.drift import DriftDetector
        from nmem_immune.immunity import ImmunityClassifier
        from nmem_immune.quarantine import QuarantineManager
        from nmem_immune.skeptic import Skeptic

        self._skeptic = Skeptic(self._pool, self._stats)
        self._quarantine = QuarantineManager(self._pool, self._stats)
        self._drift = DriftDetector(self._pool, self._stats)
        self._antidote = AntidotePropagator(self._pool, self._stats)
        self._immunity = ImmunityClassifier(self._pool, self._stats)

    # ── Error isolation ──────────────────────────────────────

    async def _safe(self, name: str, coro) -> None:
        """Run a coroutine with crash isolation. Never raises."""
        try:
            await coro
        except Exception:
            self._stats.errors += 1
            log.warning("ImmuneBridge.%s failed", name, exc_info=True)

    # ── Core audit path ──────────────────────────────────────

    async def _audit_entry(self, data: dict, source_table: str) -> None:
        """Post-write audit: run skeptic, quarantine if flagged."""
        entry_id = data.get("id")
        agent_id = data.get("agent_id", "unknown")
        if entry_id is None:
            return

        self._stats.entries_audited += 1

        # Check immunity patterns first
        poison_score = await self._immunity.check(source_table, entry_id)

        # Run skeptic
        verdict = await self._skeptic.evaluate(
            source_table, entry_id, agent_id,
            immunity_score=poison_score,
        )

        if verdict.should_quarantine:
            await self._quarantine.quarantine(
                source_table=source_table,
                source_id=entry_id,
                agent_id=agent_id,
                reason=verdict.reason,
                detail=verdict.detail,
                scores=verdict.scores,
            )
            self._stats.entries_quarantined += 1
        else:
            self._stats.entries_cleared += 1

    # ── Provenance tracking ──────────────────────────────────

    async def _track_provenance(self, data: dict) -> None:
        """Record provenance link when LTM entry is promoted from journal."""
        from nmem_immune.provenance import record_link

        entry_id = data.get("id")
        source = data.get("source")
        if entry_id is None:
            return

        if source == "promotion":
            row = await self._pool.fetchrow(
                "SELECT source_journal_id FROM nmem_long_term_memory WHERE id = $1",
                entry_id,
            )
            if row and row["source_journal_id"]:
                await record_link(
                    self._pool,
                    parent_table="nmem_journal_entries",
                    parent_id=row["source_journal_id"],
                    child_table="nmem_long_term_memory",
                    child_id=entry_id,
                    link_type="promoted_from",
                )
                self._stats.provenance_links_recorded += 1

    # ── Consolidation handlers ───────────────────────────────

    async def _handle_drift_detection(self) -> None:
        await self._safe("drift", self._drift.run_cycle())

    async def _handle_quarantine_review(self) -> None:
        await self._safe("quarantine_review", self._quarantine.review_pending())

    async def _handle_immunity_decay(self) -> None:
        await self._safe("immunity_decay", self._immunity.decay_all())

    # ── Manual API ───────────────────────────────────────────

    # Allowlist of nmem tables that can be referenced in immune operations.
    _VALID_TABLES = frozenset({
        "nmem_journal_entries",
        "nmem_long_term_memory",
        "nmem_shared_knowledge",
        "nmem_entity_memory",
    })

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("ImmuneBridge not connected — call connect(memory) first")

    @classmethod
    def _validate_table(cls, table: str) -> None:
        if table not in cls._VALID_TABLES:
            raise ValueError(
                f"table must be one of {sorted(cls._VALID_TABLES)}, got {table!r}"
            )

    async def mark_poisoned(
        self, table: str, record_id: int, reason: str = "manual",
    ) -> dict:
        """Manually mark a record as poisoned and trigger antidote propagation.

        Returns dict with propagation results.

        Raises:
            ValueError: If table is not a recognized nmem table.
        """
        self._require_connected()
        self._validate_table(table)
        result = await self._antidote.propagate(table, record_id, reason)
        await self._immunity.learn_from_poison(table, record_id)
        self._stats.antidote_propagations += 1
        return result

    async def quarantine_status(self) -> dict:
        """Return summary of quarantine state."""
        self._require_connected()
        return await self._quarantine.status_summary()

    async def immunity_status(self) -> dict:
        """Return summary of active immunity records."""
        self._require_connected()
        return await self._immunity.status_summary()
