"""Layer 4 — Antidote propagation.

When a memory is marked poisoned, walk the provenance graph downward.
Every descendant is tainted — quarantined, not deleted. The poisoned
content is kept (flagged) for pattern extraction and recovery.

Triggered manually via bridge.mark_poisoned(), not event-driven in v1.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from nmem_immune import config
from nmem_immune.provenance import get_descendants

if TYPE_CHECKING:
    import asyncpg
    from nmem_immune.bridge import ImmuneStats

log = logging.getLogger(__name__)


class AntidotePropagator:
    """Walks provenance graph and taints descendants of poisoned entries."""

    def __init__(self, pool: asyncpg.Pool, stats: ImmuneStats) -> None:
        self._pool = pool
        self._stats = stats

    async def propagate(
        self, source_table: str, source_id: int, reason: str = "manual",
    ) -> dict:
        """Mark a record as poisoned and taint all descendants.

        Returns:
            {source: (table, id), tainted_count: N, depth_reached: D}
        """
        from nmem_immune.quarantine import QuarantineManager

        qm = QuarantineManager(self._pool, self._stats)

        # Ensure source is in quarantine, then mark as confirmed poison.
        # quarantine() is atomic (ON CONFLICT), so this is safe even if
        # the entry is already quarantined.
        await qm.quarantine(
            source_table=source_table,
            source_id=source_id,
            agent_id="system",
            reason="confirmed_poison",
            detail=f"manually marked poisoned: {reason}",
            scores={"manual": True},
        )
        await qm.mark_confirmed_poison(source_table, source_id)

        # Walk descendants
        descendants = await get_descendants(
            self._pool, source_table, source_id,
            max_depth=config.settings.antidote_max_depth,
        )

        max_depth = 0
        for child_table, child_id, depth in descendants:
            await qm.quarantine(
                source_table=child_table,
                source_id=child_id,
                agent_id="system",
                reason="tainted_by_antidote",
                detail=(
                    f"tainted by poison in {source_table}:{source_id} "
                    f"at depth {depth}"
                ),
                scores={
                    "poison_source_table": source_table,
                    "poison_source_id": source_id,
                    "depth": depth,
                },
            )
            max_depth = max(max_depth, depth)

        # Audit log
        await self._pool.execute(
            """
            INSERT INTO immune_audit_log (action, target_table, target_id, agent_id, detail)
            VALUES ($1, $2, $3, $4, $5)
            """,
            "antidote_propagated",
            source_table,
            source_id,
            "system",
            json.dumps({
                "reason": reason,
                "tainted_count": len(descendants),
                "max_depth": max_depth,
            }),
        )

        result = {
            "source": (source_table, source_id),
            "tainted_count": len(descendants),
            "depth_reached": max_depth,
        }
        log.info(
            "Antidote propagated from %s:%d — tainted %d descendants (max depth %d)",
            source_table, source_id, len(descendants), max_depth,
        )
        return result
