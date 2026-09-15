"""Layer 1 — Intake Skeptic (post-write auditor).

In v1 (no write middleware in core), the skeptic operates after the write
has landed in nmem. It reads the just-written row via asyncpg (read-only),
checks for contradictions against existing entries, and returns a verdict
that the bridge uses to route into quarantine.

Contradiction detection adapts the logic from nmem/conflicts.py:
  - Text similarity (Jaccard, word-level): high overlap = same topic
  - Vector similarity (cosine): alignment of meaning
  - Conflict when: text_sim >= text_threshold AND vec_sim < vector_threshold
    (same topic, different meaning)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from nmem_immune import config
from nmem_immune.util import cosine_similarity, parse_embedding, text_similarity

if TYPE_CHECKING:
    import asyncpg
    from nmem_immune.bridge import ImmuneStats

log = logging.getLogger(__name__)

# nmem table → (content_col, embedding_col, agent_col)
# Only these tables are queried — acts as an allowlist against SQL injection.
_TABLE_COLUMNS: dict[str, tuple[str, str, str]] = {
    "nmem_journal_entries": ("content", "embedding", "agent_id"),
    "nmem_long_term_memory": ("content", "embedding", "agent_id"),
    "nmem_shared_knowledge": ("content", "embedding", "created_by"),
}


@dataclass(frozen=True, slots=True)
class SkepticVerdict:
    """Result of skeptic evaluation."""

    should_quarantine: bool
    reason: str  # clear, contradiction, poison_pattern, low_trust
    detail: str
    scores: dict = field(default_factory=dict)


class Skeptic:
    """Post-write auditor that checks for contradictions and trust issues."""

    def __init__(self, pool: asyncpg.Pool, stats: ImmuneStats) -> None:
        self._pool = pool
        self._stats = stats

    async def evaluate(
        self,
        source_table: str,
        source_id: int,
        agent_id: str,
        *,
        immunity_score: float = 0.0,
    ) -> SkepticVerdict:
        """Evaluate a just-written entry for quarantine signals.

        Returns a SkepticVerdict indicating whether quarantine is warranted.
        """
        if not config.settings.skeptic_enabled:
            return SkepticVerdict(False, "clear", "skeptic disabled", {})

        cols = _TABLE_COLUMNS.get(source_table)
        if not cols:
            return SkepticVerdict(False, "clear", "unknown table", {})

        content_col, embedding_col, agent_col = cols

        # Step 1: Read the just-written entry
        entry = await self._pool.fetchrow(
            f"SELECT {content_col}, {embedding_col}, {agent_col}, "  # noqa: S608
            f"grounding, importance, project_scope "
            f"FROM {source_table} WHERE id = $1",
            source_id,
        )
        if not entry:
            return SkepticVerdict(False, "clear", "entry not found", {})

        content = entry[content_col] or ""
        embedding_raw = entry[embedding_col]
        grounding = entry.get("grounding", "inferred")
        project_scope = entry.get("project_scope")

        embedding = parse_embedding(embedding_raw)
        if not embedding:
            return SkepticVerdict(False, "clear", "no embedding", {})

        # Step 2: Check immunity pattern score
        if immunity_score >= config.settings.skeptic_poison_pattern_threshold:
            return SkepticVerdict(
                True,
                "poison_pattern",
                f"immunity classifier score {immunity_score:.2f} >= threshold",
                {"pattern_match_score": immunity_score},
            )

        # Step 3: Find nearest neighbors and check for contradictions
        contradiction = await self._check_contradiction(
            source_table, source_id, content, embedding, grounding,
            content_col, embedding_col, agent_col, project_scope,
        )
        if contradiction:
            return contradiction

        # Step 4: Trust level check (grounding-based)
        trust = config.GROUNDING_TRUST.get(grounding, 0.3)
        if trust < config.settings.skeptic_trust_delta_threshold:
            return SkepticVerdict(
                True,
                "low_trust",
                f"grounding '{grounding}' trust {trust:.2f} below threshold",
                {"trust_score": trust, "grounding": grounding},
            )

        return SkepticVerdict(False, "clear", "passed all checks", {
            "immunity_score": immunity_score,
        })

    async def _check_contradiction(
        self,
        source_table: str,
        source_id: int,
        content: str,
        embedding: list[float],
        grounding: str,
        content_col: str,
        embedding_col: str,
        agent_col: str,
        project_scope: str | None = None,
    ) -> SkepticVerdict | None:
        """Check for contradiction against existing entries.

        Contradiction = high text overlap (same topic) + low vector similarity
        (different meaning). Uses separate thresholds for each dimension.
        Only compares within the same project_scope to avoid cross-project
        false positives.
        """
        candidates = await self._pool.fetch(
            f"SELECT id, {content_col}, {embedding_col}, {agent_col}, grounding "  # noqa: S608
            f"FROM {source_table} "
            f"WHERE id != $1 AND {embedding_col} IS NOT NULL "
            f"AND (project_scope = $3 OR (project_scope IS NULL AND $3 IS NULL)) "
            f"ORDER BY created_at DESC LIMIT $2",
            source_id,
            config.settings.skeptic_scan_limit,
            project_scope,
        )

        for cand in candidates:
            cand_content = cand[content_col] or ""
            cand_embedding = parse_embedding(cand[embedding_col])
            if not cand_embedding:
                continue

            text_sim = text_similarity(content, cand_content)
            vec_sim = cosine_similarity(embedding, cand_embedding)

            # Contradiction: same topic but divergent meaning
            if (
                text_sim >= config.settings.skeptic_text_overlap_threshold
                and vec_sim < config.settings.skeptic_vector_divergence_threshold
            ):
                cand_grounding = cand.get("grounding", "inferred")
                cand_trust = config.GROUNDING_TRUST.get(cand_grounding, 0.3)
                new_trust = config.GROUNDING_TRUST.get(grounding, 0.3)

                return SkepticVerdict(
                    True,
                    "contradiction",
                    (
                        f"contradicts {source_table}:{cand['id']} "
                        f"(text_sim={text_sim:.2f}, vec_sim={vec_sim:.2f})"
                    ),
                    {
                        "contradiction_score": text_sim,
                        "vector_similarity": vec_sim,
                        "incumbent_id": cand["id"],
                        "trust_delta": cand_trust - new_trust,
                    },
                )

        return None
