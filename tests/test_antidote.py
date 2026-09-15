"""Tests for AntidotePropagator — provenance walk and taint."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from nmem_immune.bridge import ImmuneStats
from nmem_immune.antidote import AntidotePropagator


@pytest.fixture
def pool():
    return AsyncMock()


@pytest.fixture
def stats():
    return ImmuneStats()


class TestAntidotePropagator:
    @pytest.mark.asyncio
    async def test_propagate_no_descendants(self, pool, stats):
        with patch("nmem_immune.antidote.get_descendants", new_callable=AsyncMock) as mock_desc:
            mock_desc.return_value = []

            # quarantine() does atomic ON CONFLICT upsert (single fetchrow)
            # then mark_confirmed_poison does execute
            pool.fetchrow = AsyncMock(return_value={"id": 1})
            pool.execute = AsyncMock()

            propagator = AntidotePropagator(pool, stats)
            result = await propagator.propagate("nmem_long_term_memory", 42, "test")

        assert result["tainted_count"] == 0
        assert result["depth_reached"] == 0

    @pytest.mark.asyncio
    async def test_propagate_with_descendants(self, pool, stats):
        with patch("nmem_immune.antidote.get_descendants", new_callable=AsyncMock) as mock_desc:
            mock_desc.return_value = [
                ("nmem_long_term_memory", 43, 1),
                ("nmem_shared_knowledge", 10, 2),
            ]

            # All quarantine calls use the same fetchrow (ON CONFLICT upsert)
            pool.fetchrow = AsyncMock(side_effect=[
                {"id": 1},  # source quarantine
                {"id": 2},  # first descendant
                {"id": 3},  # second descendant
            ])
            pool.execute = AsyncMock()

            propagator = AntidotePropagator(pool, stats)
            result = await propagator.propagate("nmem_long_term_memory", 42)

        assert result["tainted_count"] == 2
        assert result["depth_reached"] == 2
