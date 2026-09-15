"""Layer 5 — Immunity classifier.

Immunity records aggregate features from confirmed poison events into a
simple pattern-matching classifier. The classifier feeds back into the
write-time skeptic so new writes matching known-poison shapes are treated
with extra caution.

Per the design doc:
- Simple, updatable, not a deep net.
- Features DECAY unless reinforced by further poison events.
- The classifier MODULATES trust score — it doesn't veto.
- Periodic false-positive review: quarantined-but-promoted entries become
  negative training examples.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import TYPE_CHECKING

from nmem_immune import config
from nmem_immune.util import extract_keywords

if TYPE_CHECKING:
    import asyncpg
    from nmem_immune.bridge import ImmuneStats

log = logging.getLogger(__name__)

# Allowlist of nmem tables we read from (prevents SQL injection via source_table).
_ALLOWED_TABLES: dict[str, str] = {
    "nmem_journal_entries": "content, agent_id, grounding, importance",
    "nmem_long_term_memory": "content, agent_id, grounding, importance",
    "nmem_shared_knowledge": "content, created_by, grounding, importance",
}


class ImmunityClassifier:
    """Pattern-based poison classifier with decay and false-positive feedback."""

    def __init__(self, pool: asyncpg.Pool, stats: ImmuneStats) -> None:
        self._pool = pool
        self._stats = stats

    async def check(self, source_table: str, source_id: int) -> float:
        """Check a new entry against active immunity patterns.

        Returns a composite poison-likelihood score 0.0-1.0.
        """
        entry = await self._read_entry(source_table, source_id)
        if not entry:
            return 0.0

        content = entry.get("content") or ""
        agent_id = entry.get("agent_id") or entry.get("created_by") or ""

        # Fetch active immunity records
        records = await self._pool.fetch(
            """
            SELECT pattern_type, feature_vector, confidence
            FROM immune_immunity_records
            WHERE status = 'active' AND confidence >= $1
            """,
            config.settings.immunity_min_confidence,
        )

        if not records:
            return 0.0

        scores: list[float] = []
        for rec in records:
            features = (
                json.loads(rec["feature_vector"])
                if isinstance(rec["feature_vector"], str)
                else rec["feature_vector"]
            )
            match_score = self._match_pattern(
                rec["pattern_type"], features, content, agent_id,
            )
            if match_score > 0:
                scores.append(match_score * rec["confidence"])

        if not scores:
            return 0.0

        return min(max(scores), 1.0)

    async def learn_from_poison(
        self, source_table: str, source_id: int,
    ) -> int | None:
        """Extract features from a confirmed poison entry and create
        or reinforce immunity records.

        Returns immunity record ID, or None if entry not found.
        """
        entry = await self._read_entry(source_table, source_id)
        if not entry:
            return None

        content = entry.get("content") or ""
        agent_id = entry.get("agent_id") or entry.get("created_by") or ""

        keywords = extract_keywords(content)

        features = {
            "agent_id": agent_id,
            "keywords": keywords,
            "source_table": source_table,
            "content_hash": hashlib.sha256(content.encode()).hexdigest()[:16],
        }

        # Check if we already have a matching pattern for this agent
        existing = await self._pool.fetchrow(
            """
            SELECT id, hit_count, confidence FROM immune_immunity_records
            WHERE pattern_type = 'content_pattern'
              AND feature_vector->>'agent_id' = $1
              AND status = 'active'
            ORDER BY hit_count DESC
            LIMIT 1
            """,
            agent_id,
        )

        if existing:
            await self._pool.execute(
                """
                UPDATE immune_immunity_records
                SET hit_count = hit_count + 1,
                    confidence = LEAST(confidence + 0.1, 1.0),
                    last_reinforced = NOW(),
                    updated_at = NOW()
                WHERE id = $1
                """,
                existing["id"],
            )
            self._stats.immunity_records_created += 1
            return existing["id"]

        row = await self._pool.fetchrow(
            """
            INSERT INTO immune_immunity_records
                (pattern_type, feature_vector, description, confidence)
            VALUES ($1, $2, $3, $4)
            RETURNING id
            """,
            "content_pattern",
            json.dumps(features),
            f"Poison pattern from {source_table}:{source_id} by {agent_id}",
            0.5,
        )
        self._stats.immunity_records_created += 1
        log.info("Created immunity record #%d from %s:%d", row["id"], source_table, source_id)
        return row["id"]

    async def decay_all(self) -> int:
        """Decay confidence on all active immunity records.

        Records below retirement threshold are retired.
        Returns count of decayed records.
        """
        result = await self._pool.execute(
            """
            UPDATE immune_immunity_records
            SET confidence = confidence - decay_rate,
                updated_at = NOW()
            WHERE status = 'active'
            """,
        )
        decayed = int(result.split()[-1]) if result else 0

        retired = await self._pool.execute(
            """
            UPDATE immune_immunity_records
            SET status = 'retired', updated_at = NOW()
            WHERE status = 'active'
              AND confidence < $1
            """,
            config.settings.immunity_retirement_threshold,
        )
        retired_count = int(retired.split()[-1]) if retired else 0

        if decayed or retired_count:
            self._stats.immunity_records_decayed += decayed
            log.info(
                "Immunity decay: %d records decayed, %d retired",
                decayed, retired_count,
            )
        return decayed

    async def record_false_positive(self, quarantine_id: int) -> None:
        """Record that a quarantined entry was promoted (false positive).

        Increments false_positive_count on matching immunity records.
        If false_positive_count exceeds hit_count, retire the record.
        """
        q_row = await self._pool.fetchrow(
            """
            SELECT agent_id, skeptic_scores FROM immune_quarantine WHERE id = $1
            """,
            quarantine_id,
        )
        if not q_row:
            return

        agent_id = q_row["agent_id"]

        await self._pool.execute(
            """
            UPDATE immune_immunity_records
            SET false_positive_count = false_positive_count + 1,
                updated_at = NOW()
            WHERE status = 'active'
              AND feature_vector->>'agent_id' = $1
            """,
            agent_id,
        )

        # Auto-retire records where false positives exceed hits
        await self._pool.execute(
            """
            UPDATE immune_immunity_records
            SET status = 'retired', updated_at = NOW()
            WHERE status = 'active'
              AND false_positive_count > hit_count
            """,
        )

        await self._pool.execute(
            """
            INSERT INTO immune_audit_log (action, target_table, target_id, agent_id, detail)
            VALUES ($1, $2, $3, $4, $5)
            """,
            "false_positive_recorded",
            "immune_quarantine",
            quarantine_id,
            agent_id,
            json.dumps({"quarantine_id": quarantine_id}),
        )

    async def status_summary(self) -> dict:
        """Return counts by status and total hits."""
        rows = await self._pool.fetch(
            """
            SELECT status, COUNT(*) AS cnt, SUM(hit_count) AS total_hits
            FROM immune_immunity_records
            GROUP BY status
            """
        )
        return {
            row["status"]: {"count": row["cnt"], "total_hits": row["total_hits"] or 0}
            for row in rows
        }

    # ── Internal helpers ─────────────────────────────────────

    async def _read_entry(self, source_table: str, source_id: int) -> dict | None:
        """Read an entry from an nmem table (read-only).

        Only reads from allowlisted tables to prevent SQL injection.
        """
        cols = _ALLOWED_TABLES.get(source_table)
        if cols is None:
            log.warning("Rejecting read from unknown table: %s", source_table)
            return None

        row = await self._pool.fetchrow(
            f"SELECT {cols} FROM {source_table} WHERE id = $1",  # noqa: S608
            source_id,
        )
        return dict(row) if row else None

    def _match_pattern(
        self,
        pattern_type: str,
        features: dict,
        content: str,
        agent_id: str,
    ) -> float:
        """Score how well an immunity record matches the given content."""
        if pattern_type == "source_agent":
            return 1.0 if features.get("agent_id") == agent_id else 0.0

        if pattern_type == "content_pattern":
            score = 0.0
            if features.get("agent_id") == agent_id:
                score += 0.3
            keywords = features.get("keywords", [])
            if keywords:
                content_lower = content.lower()
                # Word-boundary matching to avoid substring false positives
                matches = sum(
                    1 for kw in keywords
                    if re.search(rf"\b{re.escape(kw)}\b", content_lower)
                )
                score += 0.7 * (matches / len(keywords))
            return score

        if pattern_type == "topic_cluster":
            keywords = features.get("keywords", [])
            if not keywords:
                return 0.0
            content_lower = content.lower()
            matches = sum(
                1 for kw in keywords
                if re.search(rf"\b{re.escape(kw)}\b", content_lower)
            )
            return matches / len(keywords)

        return 0.0
