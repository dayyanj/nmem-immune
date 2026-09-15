"""Tests for the Skeptic layer — contradiction and trust checks."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from nmem_immune.bridge import ImmuneStats
from nmem_immune.skeptic import Skeptic, SkepticVerdict
from nmem_immune.util import cosine_similarity, text_similarity


class TestTextSimilarity:
    def test_identical(self):
        assert text_similarity("hello world", "hello world") == 1.0

    def test_disjoint(self):
        assert text_similarity("hello world", "foo bar") == 0.0

    def test_partial(self):
        sim = text_similarity("the quick brown fox", "the quick red fox")
        assert 0.5 < sim < 1.0

    def test_empty(self):
        assert text_similarity("", "hello") == 0.0


class TestCosineSimilarity:
    def test_identical(self):
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_empty(self):
        assert cosine_similarity([], [1, 0]) == 0.0


class TestSkeptic:
    @pytest.fixture
    def pool(self):
        return AsyncMock()

    @pytest.fixture
    def stats(self):
        return ImmuneStats()

    @pytest.mark.asyncio
    async def test_disabled_returns_clear(self, pool, stats):
        skeptic = Skeptic(pool, stats)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("nmem_immune.skeptic.config.settings.skeptic_enabled", False)
            verdict = await skeptic.evaluate("nmem_journal_entries", 1, "agent1")
        assert not verdict.should_quarantine
        assert verdict.reason == "clear"

    @pytest.mark.asyncio
    async def test_unknown_table_returns_clear(self, pool, stats):
        skeptic = Skeptic(pool, stats)
        verdict = await skeptic.evaluate("unknown_table", 1, "agent1")
        assert not verdict.should_quarantine

    @pytest.mark.asyncio
    async def test_poison_pattern_triggers_quarantine(self, pool, stats):
        pool.fetchrow = AsyncMock(return_value={
            "content": "test content",
            "embedding": [0.1, 0.2, 0.3],
            "agent_id": "agent1",
            "grounding": "inferred",
            "importance": 5,
        })
        pool.fetch = AsyncMock(return_value=[])

        skeptic = Skeptic(pool, stats)
        verdict = await skeptic.evaluate(
            "nmem_journal_entries", 1, "agent1",
            immunity_score=0.8,
        )
        assert verdict.should_quarantine
        assert verdict.reason == "poison_pattern"

    @pytest.mark.asyncio
    async def test_entry_not_found_returns_clear(self, pool, stats):
        pool.fetchrow = AsyncMock(return_value=None)

        skeptic = Skeptic(pool, stats)
        verdict = await skeptic.evaluate("nmem_journal_entries", 999, "agent1")
        assert not verdict.should_quarantine
        assert verdict.reason == "clear"

    @pytest.mark.asyncio
    async def test_no_embedding_returns_clear(self, pool, stats):
        pool.fetchrow = AsyncMock(return_value={
            "content": "test",
            "embedding": None,
            "agent_id": "agent1",
            "grounding": "confirmed",
            "importance": 5,
        })

        skeptic = Skeptic(pool, stats)
        verdict = await skeptic.evaluate("nmem_journal_entries", 1, "agent1")
        assert not verdict.should_quarantine

    @pytest.mark.asyncio
    async def test_clear_verdict_for_benign_entry(self, pool, stats):
        pool.fetchrow = AsyncMock(return_value={
            "content": "normal content about project setup",
            "embedding": [0.1, 0.2, 0.3],
            "agent_id": "agent1",
            "grounding": "confirmed",
            "importance": 5,
        })
        # No contradicting candidates
        pool.fetch = AsyncMock(return_value=[])

        skeptic = Skeptic(pool, stats)
        verdict = await skeptic.evaluate("nmem_journal_entries", 1, "agent1")
        assert not verdict.should_quarantine
        assert verdict.reason == "clear"

    @pytest.mark.asyncio
    async def test_low_trust_grounding_triggers_quarantine(self, pool, stats):
        """Entries with 'disputed' grounding should be quarantined."""
        pool.fetchrow = AsyncMock(return_value={
            "content": "some disputed claim",
            "embedding": [0.1, 0.2, 0.3],
            "agent_id": "agent1",
            "grounding": "disputed",
            "importance": 5,
        })
        pool.fetch = AsyncMock(return_value=[])

        skeptic = Skeptic(pool, stats)
        verdict = await skeptic.evaluate("nmem_journal_entries", 1, "agent1")
        assert verdict.should_quarantine
        assert verdict.reason == "low_trust"
