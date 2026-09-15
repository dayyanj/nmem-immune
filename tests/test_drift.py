"""Tests for DriftDetector — stale entry sampling and compatibility checks."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from nmem_immune.bridge import ImmuneStats
from nmem_immune.drift import DriftDetector


@pytest.fixture
def pool():
    return AsyncMock()


@pytest.fixture
def stats():
    return ImmuneStats()


class TestDriftDetector:
    @pytest.mark.asyncio
    async def test_run_cycle_no_entries(self, pool, stats):
        pool.fetch = AsyncMock(return_value=[])

        dd = DriftDetector(pool, stats)
        issues = await dd.run_cycle()
        assert issues == 0
        assert stats.drift_checks_run == 1

    @pytest.mark.asyncio
    async def test_no_sibling_found(self, pool, stats):
        pool.fetch = AsyncMock(side_effect=[
            # _sample_stale_entries returns one entry
            [{"id": 1, "agent_id": "a", "key": "k", "content": "hello",
              "grounding": "inferred", "importance": 8, "embedding": [0.1, 0.2],
              "last_accessed_at": None, "last_validated_at": None,
              "project_scope": None}],
            # _find_sibling fallback returns empty
            [],
        ])
        # _find_sibling pgvector attempt fails
        pool.fetchrow = AsyncMock(side_effect=Exception("no pgvector"))

        dd = DriftDetector(pool, stats)
        issues = await dd.run_cycle()
        assert issues == 0

    def test_compatibility_unrelated_entries(self, pool, stats):
        dd = DriftDetector(pool, stats)
        # Low text overlap = unrelated, should return 1.0 (no drift concern)
        compat = dd._check_compatibility(
            {"content": "alpha beta gamma", "embedding": [1, 0, 0]},
            {"content": "delta epsilon zeta", "embedding": [0, 1, 0]},
        )
        assert compat == 1.0

    def test_compatibility_drifted_entries(self, pool, stats):
        dd = DriftDetector(pool, stats)
        # High text overlap but different embeddings = drift
        compat = dd._check_compatibility(
            {"content": "the patient requires medication daily",
             "embedding": [1.0, 0.0, 0.0]},
            {"content": "the patient requires medication weekly",
             "embedding": [0.0, 1.0, 0.0]},
        )
        # text_sim is high (>0.3), so compat = vec_sim which is ~0.0
        assert compat < 0.3
