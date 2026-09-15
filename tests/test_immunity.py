"""Tests for ImmunityClassifier — pattern matching, learning, decay."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from nmem_immune.bridge import ImmuneStats
from nmem_immune.immunity import ImmunityClassifier


@pytest.fixture
def pool():
    return AsyncMock()


@pytest.fixture
def stats():
    return ImmuneStats()


class TestImmunityCheck:
    @pytest.mark.asyncio
    async def test_no_records_returns_zero(self, pool, stats):
        # _read_entry returns an entry
        pool.fetchrow = AsyncMock(return_value={
            "content": "test content",
            "agent_id": "agent1",
            "grounding": "inferred",
            "importance": 5,
        })
        # No immunity records
        pool.fetch = AsyncMock(return_value=[])

        ic = ImmunityClassifier(pool, stats)
        score = await ic.check("nmem_journal_entries", 1)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_matching_agent_pattern(self, pool, stats):
        pool.fetchrow = AsyncMock(return_value={
            "content": "suspicious content with keywords",
            "agent_id": "bad_agent",
            "grounding": "inferred",
            "importance": 5,
        })
        pool.fetch = AsyncMock(return_value=[
            {
                "pattern_type": "content_pattern",
                "feature_vector": json.dumps({
                    "agent_id": "bad_agent",
                    "keywords": ["suspicious", "keywords"],
                }),
                "confidence": 0.8,
            },
        ])

        ic = ImmunityClassifier(pool, stats)
        score = await ic.check("nmem_journal_entries", 1)
        # agent match (0.3) + keyword match (0.7) * confidence (0.8)
        assert score > 0.5

    @pytest.mark.asyncio
    async def test_entry_not_found(self, pool, stats):
        pool.fetchrow = AsyncMock(return_value=None)

        ic = ImmunityClassifier(pool, stats)
        score = await ic.check("nmem_journal_entries", 999)
        assert score == 0.0


class TestImmunityLearn:
    @pytest.mark.asyncio
    async def test_learn_creates_record(self, pool, stats):
        pool.fetchrow = AsyncMock(side_effect=[
            # _read_entry
            {"content": "poisoned content here", "agent_id": "bad", "grounding": "inferred", "importance": 3},
            # No existing immunity record
            None,
            # INSERT RETURNING
            {"id": 42},
        ])

        ic = ImmunityClassifier(pool, stats)
        rid = await ic.learn_from_poison("nmem_journal_entries", 1)
        assert rid == 42
        assert stats.immunity_records_created == 1

    @pytest.mark.asyncio
    async def test_learn_reinforces_existing(self, pool, stats):
        pool.fetchrow = AsyncMock(side_effect=[
            # _read_entry
            {"content": "poisoned again", "agent_id": "bad", "grounding": "inferred", "importance": 3},
            # Existing immunity record found
            {"id": 7, "hit_count": 2, "confidence": 0.6},
        ])
        pool.execute = AsyncMock()

        ic = ImmunityClassifier(pool, stats)
        rid = await ic.learn_from_poison("nmem_journal_entries", 2)
        assert rid == 7
        pool.execute.assert_called_once()  # UPDATE hit_count


class TestImmunityDecay:
    @pytest.mark.asyncio
    async def test_decay_all(self, pool, stats):
        pool.execute = AsyncMock(side_effect=[
            "UPDATE 5",  # decay
            "UPDATE 1",  # retire
        ])

        ic = ImmunityClassifier(pool, stats)
        count = await ic.decay_all()
        assert count == 5
        assert stats.immunity_records_decayed == 5


class TestKeywordExtraction:
    def test_extracts_top_keywords(self):
        from nmem_immune.util import extract_keywords
        text = "the patient requires medication for treatment and the treatment plan includes medication"
        keywords = extract_keywords(text, top_n=3)
        assert "medication" in keywords
        assert "treatment" in keywords
        assert len(keywords) <= 3

    def test_empty_text(self):
        from nmem_immune.util import extract_keywords
        assert extract_keywords("") == []


class TestStatusSummary:
    @pytest.mark.asyncio
    async def test_status_summary(self, pool, stats):
        pool.fetch = AsyncMock(return_value=[
            {"status": "active", "cnt": 10, "total_hits": 25},
            {"status": "retired", "cnt": 3, "total_hits": 8},
        ])

        ic = ImmunityClassifier(pool, stats)
        summary = await ic.status_summary()
        assert summary["active"]["count"] == 10
        assert summary["active"]["total_hits"] == 25
