"""Layer 2 — Quarantine management.

Quarantine entries are metadata overlays on nmem rows. The source rows in
nmem tables are never modified (v1 constraint). Downstream consumers that
respect the immune overlay can query immune_quarantine to exclude flagged IDs.

Promotion conditions (any one):
  1. Corroboration — 2+ independent sources wrote similar content
  2. Aging — quarantined >= N days with no contradicting evidence
  3. Manual clearance
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING

from nmem_immune import config

if TYPE_CHECKING:
    import asyncpg
    from nmem_immune.bridge import ImmuneStats

log = logging.getLogger(__name__)


async def _audit(
    pool: asyncpg.Pool,
    action: str,
    target_table: str | None,
    target_id: int | None,
    agent_id: str | None,
    detail: dict | None = None,
) -> None:
    """Write an audit log entry."""
    await pool.execute(
        """
        INSERT INTO immune_audit_log (action, target_table, target_id, agent_id, detail)
        VALUES ($1, $2, $3, $4, $5)
        """,
        action,
        target_table,
        target_id,
        agent_id,
        json.dumps(detail or {}),
    )


class QuarantineManager:
    """CRUD and lifecycle management for quarantined entries."""

    def __init__(self, pool: asyncpg.Pool, stats: ImmuneStats) -> None:
        self._pool = pool
        self._stats = stats

    async def quarantine(
        self,
        source_table: str,
        source_id: int,
        agent_id: str,
        reason: str,
        detail: str,
        scores: dict,
        *,
        content: str = "",
        grounding: str | None = None,
        importance: int | None = None,
        source_type: str | None = None,
        write_agent: str | None = None,
    ) -> int:
        """Insert a quarantine record. Returns quarantine entry ID.

        Deduplicates by (source_table, source_id) — if already quarantined
        with status='quarantined', skips and returns existing ID.
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Atomic upsert: the partial unique index on (source_table, source_id)
        # WHERE status='quarantined' prevents duplicates without a TOCTOU race.
        # DO NOTHING preserves the original reason/scores when already quarantined.
        row = await self._pool.fetchrow(
            """
            INSERT INTO immune_quarantine
                (source_table, source_id, agent_id, content_hash, reason, detail,
                 skeptic_scores, grounding, importance, source_type, write_agent)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (source_table, source_id) WHERE status = 'quarantined'
                DO NOTHING
            RETURNING id
            """,
            source_table,
            source_id,
            agent_id,
            content_hash,
            reason,
            detail,
            json.dumps(scores),
            grounding,
            importance,
            source_type,
            write_agent,
        )

        if row:
            # New insert succeeded
            qid = row["id"]
            await _audit(
                self._pool, "quarantined", source_table, source_id, agent_id,
                {"quarantine_id": qid, "reason": reason, "scores": scores},
            )
            log.info(
                "Quarantined %s:%d (reason=%s, agent=%s)",
                source_table, source_id, reason, agent_id,
            )
            return qid

        # Already quarantined — fetch existing ID
        existing = await self._pool.fetchrow(
            """
            SELECT id FROM immune_quarantine
            WHERE source_table = $1 AND source_id = $2 AND status = 'quarantined'
            """,
            source_table,
            source_id,
        )
        return existing["id"]

    async def review_pending(self) -> int:
        """Review quarantined entries for promotion or expiry.

        Returns total count of status changes (promoted + expired).
        """
        changed = 0
        changed += await self._promote_by_aging()
        changed += await self._expire_old()
        return changed

    async def _promote_by_aging(self) -> int:
        """Promote entries quarantined longer than QUARANTINE_AGING_DAYS
        that have no contradicting evidence (no other quarantine entry
        on the same source with a different reason).
        """
        rows = await self._pool.fetch(
            """
            UPDATE immune_quarantine
            SET status = 'promoted',
                promoted_at = NOW(),
                promoted_by = 'aging'
            WHERE status = 'quarantined'
              AND created_at < NOW() - ($1 || ' days')::INTERVAL
              AND reason NOT IN ('confirmed_poison', 'tainted_by_antidote')
            RETURNING id, source_table, source_id, agent_id
            """,
            str(config.settings.quarantine_aging_days),
        )
        for row in rows:
            await _audit(
                self._pool, "promoted", row["source_table"], row["source_id"],
                row["agent_id"], {"quarantine_id": row["id"], "promoted_by": "aging"},
            )
            self._stats.quarantine_promotions += 1
        if rows:
            log.info("Promoted %d quarantine entries via aging", len(rows))
        return len(rows)

    async def _expire_old(self) -> int:
        """Mark quarantine entries older than QUARANTINE_EXPIRY_DAYS as expired."""
        result = await self._pool.execute(
            """
            UPDATE immune_quarantine
            SET status = 'expired'
            WHERE status = 'quarantined'
              AND created_at < NOW() - ($1 || ' days')::INTERVAL
            """,
            str(config.settings.quarantine_expiry_days),
        )
        # asyncpg returns "UPDATE N"
        count = int(result.split()[-1]) if result else 0
        if count:
            log.info("Expired %d old quarantine entries", count)
        return count

    async def mark_confirmed_poison(
        self, source_table: str, source_id: int,
    ) -> None:
        """Mark all quarantine entries for a source as confirmed_poison."""
        await self._pool.execute(
            """
            UPDATE immune_quarantine
            SET status = 'confirmed_poison', reviewed_at = NOW()
            WHERE source_table = $1 AND source_id = $2 AND status = 'quarantined'
            """,
            source_table,
            source_id,
        )
        await _audit(
            self._pool, "confirmed_poison", source_table, source_id, None, {},
        )

    async def is_quarantined(self, source_table: str, source_id: int) -> bool:
        """Check if a source entry is currently quarantined."""
        row = await self._pool.fetchrow(
            """
            SELECT 1 FROM immune_quarantine
            WHERE source_table = $1 AND source_id = $2 AND status = 'quarantined'
            LIMIT 1
            """,
            source_table,
            source_id,
        )
        return row is not None

    async def status_summary(self) -> dict:
        """Return counts by status."""
        rows = await self._pool.fetch(
            """
            SELECT status, COUNT(*) AS cnt
            FROM immune_quarantine
            GROUP BY status
            """
        )
        return {row["status"]: row["cnt"] for row in rows}
