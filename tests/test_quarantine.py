"""Tests for QuarantineManager — CRUD, promotion, expiry."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from nmem_immune.bridge import ImmuneStats
from nmem_immune.quarantine import QuarantineManager


@pytest.fixture
def pool():
    return AsyncMock()


@pytest.fixture
def stats():
    return ImmuneStats()


class TestQuarantine:
    @pytest.mark.asyncio
    async def test_quarantine_inserts_record(self, pool, stats):
        # ON CONFLICT upsert returns new id
        pool.fetchrow = AsyncMock(return_value={"id": 42})
        pool.execute = AsyncMock()  # audit log

        qm = QuarantineManager(pool, stats)
        qid = await qm.quarantine(
            source_table="nmem_journal_entries",
            source_id=1,
            agent_id="agent1",
            reason="contradiction",
            detail="test",
            scores={"score": 0.9},
        )
        assert qid == 42

    @pytest.mark.asyncio
    async def test_quarantine_deduplicates_via_on_conflict(self, pool, stats):
        # ON CONFLICT DO NOTHING returns None, then fallback SELECT returns existing
        pool.fetchrow = AsyncMock(side_effect=[
            None,       # INSERT returns None (conflict, DO NOTHING)
            {"id": 7},  # fallback SELECT returns existing id
        ])
        pool.execute = AsyncMock()

        qm = QuarantineManager(pool, stats)
        qid = await qm.quarantine(
            source_table="nmem_journal_entries",
            source_id=1,
            agent_id="agent1",
            reason="contradiction",
            detail="test",
            scores={},
        )
        assert qid == 7

    @pytest.mark.asyncio
    async def test_is_quarantined(self, pool, stats):
        pool.fetchrow = AsyncMock(return_value={"?column?": 1})

        qm = QuarantineManager(pool, stats)
        assert await qm.is_quarantined("nmem_journal_entries", 1) is True

    @pytest.mark.asyncio
    async def test_is_not_quarantined(self, pool, stats):
        pool.fetchrow = AsyncMock(return_value=None)

        qm = QuarantineManager(pool, stats)
        assert await qm.is_quarantined("nmem_journal_entries", 999) is False

    @pytest.mark.asyncio
    async def test_status_summary(self, pool, stats):
        pool.fetch = AsyncMock(return_value=[
            {"status": "quarantined", "cnt": 5},
            {"status": "promoted", "cnt": 2},
        ])

        qm = QuarantineManager(pool, stats)
        summary = await qm.status_summary()
        assert summary == {"quarantined": 5, "promoted": 2}


class TestPromotionAndExpiry:
    @pytest.mark.asyncio
    async def test_promote_by_aging(self, pool, stats):
        pool.fetch = AsyncMock(return_value=[
            {"id": 1, "source_table": "nmem_long_term_memory", "source_id": 10, "agent_id": "a"},
            {"id": 2, "source_table": "nmem_long_term_memory", "source_id": 20, "agent_id": "b"},
        ])
        pool.execute = AsyncMock()  # audit log

        qm = QuarantineManager(pool, stats)
        count = await qm._promote_by_aging()
        assert count == 2
        assert stats.quarantine_promotions == 2

    @pytest.mark.asyncio
    async def test_expire_old(self, pool, stats):
        pool.execute = AsyncMock(return_value="UPDATE 3")

        qm = QuarantineManager(pool, stats)
        count = await qm._expire_old()
        assert count == 3
