"""Layer 3 — Drift detection (silent antibody).

Runs inside the consolidation cycle as a full-cycle hook. Samples
high-importance, stale LTM entries and checks if they remain compatible
with their nearest validated sibling. Entries that have drifted are
quarantined with reason='drift_detected'.

This addresses threat (d) from the design doc: once-true facts going stale.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nmem_immune import config
from nmem_immune.util import cosine_similarity, parse_embedding, text_similarity

if TYPE_CHECKING:
    import asyncpg
    from nmem_immune.bridge import ImmuneStats

log = logging.getLogger(__name__)


class DriftDetector:
    """Consolidation-cycle drift detector for LTM entries."""

    def __init__(self, pool: asyncpg.Pool, stats: ImmuneStats) -> None:
        self._pool = pool
        self._stats = stats

    async def run_cycle(self) -> int:
        """Sample stale entries and check for drift.

        Returns count of drift issues found.
        """
        from nmem_immune.quarantine import QuarantineManager

        self._stats.drift_checks_run += 1
        entries = await self._sample_stale_entries()
        issues = 0
        qm = QuarantineManager(self._pool, self._stats)

        for entry in entries:
            sibling = await self._find_sibling(entry)
            if sibling is None:
                continue

            compat = self._check_compatibility(entry, sibling)
            if compat < config.settings.drift_compatibility_threshold:
                await qm.quarantine(
                    source_table="nmem_long_term_memory",
                    source_id=entry["id"],
                    agent_id=entry["agent_id"],
                    reason="drift_detected",
                    detail=(
                        f"compatibility {compat:.2f} with sibling "
                        f"nmem_long_term_memory:{sibling['id']}"
                    ),
                    scores={
                        "compatibility": compat,
                        "sibling_id": sibling["id"],
                    },
                    content=entry["content"],
                    grounding=entry.get("grounding"),
                    importance=entry.get("importance"),
                )
                issues += 1

        if issues:
            self._stats.drift_issues_found += issues
            log.info("Drift detection: %d issues found from %d samples", issues, len(entries))

        return issues

    async def _sample_stale_entries(self) -> list[dict]:
        """Read-only: sample LTM entries biased toward staleness."""
        rows = await self._pool.fetch(
            """
            SELECT id, agent_id, key, content, grounding, importance,
                   embedding, last_accessed_at, last_validated_at,
                   project_scope
            FROM nmem_long_term_memory
            WHERE status = 'validated'
              AND grounding IN ('inferred', 'confirmed')
              AND embedding IS NOT NULL
              AND content IS NOT NULL
            ORDER BY
                COALESCE(last_validated_at, created_at) ASC,
                importance DESC
            LIMIT $1
            """,
            config.settings.drift_sample_size,
        )
        return [dict(r) for r in rows]

    async def _find_sibling(self, entry: dict) -> dict | None:
        """Find the nearest validated LTM entry for comparison.

        Same agent, same project_scope, excluding the entry itself.
        Tries pgvector operator first, falls back to Python-side ranking.
        """
        embedding = parse_embedding(entry.get("embedding"))
        if not embedding:
            return None

        # Try pgvector operator first
        try:
            row = await self._pool.fetchrow(
                """
                SELECT id, content, embedding, grounding, importance, agent_id
                FROM nmem_long_term_memory
                WHERE id != $1
                  AND agent_id = $2
                  AND status = 'validated'
                  AND embedding IS NOT NULL
                  AND content IS NOT NULL
                  AND (project_scope = $3 OR (project_scope IS NULL AND $3 IS NULL))
                ORDER BY embedding <=> $4::vector
                LIMIT 1
                """,
                entry["id"],
                entry["agent_id"],
                entry.get("project_scope"),
                str(embedding),
            )
            if row:
                return dict(row)
        except Exception:
            log.debug("pgvector <=> operator unavailable, falling back to Python ranking")

        # Fallback: fetch candidates and rank in Python
        candidates = await self._pool.fetch(
            """
            SELECT id, content, embedding, grounding, importance, agent_id
            FROM nmem_long_term_memory
            WHERE id != $1
              AND agent_id = $2
              AND status = 'validated'
              AND embedding IS NOT NULL
              AND content IS NOT NULL
              AND (project_scope = $3 OR (project_scope IS NULL AND $3 IS NULL))
            ORDER BY created_at DESC
            LIMIT 50
            """,
            entry["id"],
            entry["agent_id"],
            entry.get("project_scope"),
        )

        best = None
        best_sim = -1.0
        for cand in candidates:
            cand_emb = parse_embedding(cand["embedding"])
            if not cand_emb:
                continue
            sim = cosine_similarity(embedding, cand_emb)
            if sim > best_sim:
                best_sim = sim
                best = dict(cand)

        return best

    def _check_compatibility(self, entry: dict, sibling: dict) -> float:
        """Compute compatibility between entry and sibling.

        If they share topic words but diverge in meaning, compatibility is low.
        If they don't share topic words, they're unrelated (no drift concern).
        """
        text_sim = text_similarity(
            entry.get("content") or "",
            sibling.get("content") or "",
        )
        entry_emb = parse_embedding(entry.get("embedding"))
        sibling_emb = parse_embedding(sibling.get("embedding"))

        if entry_emb and sibling_emb:
            vec_sim = cosine_similarity(entry_emb, sibling_emb)
        else:
            vec_sim = 0.5  # neutral if no embeddings

        # Unrelated entries (low text overlap) aren't a drift concern
        if text_sim < 0.3:
            return 1.0

        return vec_sim
