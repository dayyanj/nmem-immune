"""Tests for provenance link utilities."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from nmem_immune.provenance import get_ancestors, get_descendants, record_link


@pytest.fixture
def pool():
    return AsyncMock()


class TestRecordLink:
    @pytest.mark.asyncio
    async def test_inserts_link(self, pool):
        pool.fetchrow = AsyncMock(return_value={"id": 1})
        lid = await record_link(
            pool, "nmem_journal_entries", 10, "nmem_long_term_memory", 20,
        )
        assert lid == 1

    @pytest.mark.asyncio
    async def test_conflict_returns_none(self, pool):
        pool.fetchrow = AsyncMock(return_value=None)
        lid = await record_link(
            pool, "nmem_journal_entries", 10, "nmem_long_term_memory", 20,
        )
        assert lid is None


class TestGetDescendants:
    @pytest.mark.asyncio
    async def test_empty_graph(self, pool):
        pool.fetch = AsyncMock(return_value=[])
        result = await get_descendants(pool, "t", 1)
        assert result == []

    @pytest.mark.asyncio
    async def test_single_level(self, pool):
        pool.fetch = AsyncMock(side_effect=[
            # Root's children
            [{"child_table": "t", "child_id": 2}, {"child_table": "t", "child_id": 3}],
            # Child 2's children
            [],
            # Child 3's children
            [],
        ])
        result = await get_descendants(pool, "t", 1)
        assert len(result) == 2
        assert (result[0][0], result[0][1]) == ("t", 2)
        assert all(d == 1 for _, _, d in result)

    @pytest.mark.asyncio
    async def test_respects_max_depth(self, pool):
        # Deep chain: 1 -> 2 -> 3 -> 4
        pool.fetch = AsyncMock(side_effect=[
            [{"child_table": "t", "child_id": 2}],
            [{"child_table": "t", "child_id": 3}],
            [],  # depth 2 reached with max_depth=2
        ])
        result = await get_descendants(pool, "t", 1, max_depth=2)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_cycle_back_to_root_terminates(self, pool):
        """A child pointing back to the root must not cause infinite loop."""
        pool.fetch = AsyncMock(side_effect=[
            # Root's children: 2
            [{"child_table": "t", "child_id": 2}],
            # Child 2's children: points back to root (1)
            [{"child_table": "t", "child_id": 1}],
        ])
        result = await get_descendants(pool, "t", 1, max_depth=5)
        # Only child 2 should appear, root is excluded from results
        assert len(result) == 1
        assert result[0] == ("t", 2, 1)

    @pytest.mark.asyncio
    async def test_max_nodes_cap(self, pool):
        """BFS must stop after max_nodes to prevent runaway traversal."""
        # Each level returns 5 children — would be 25 at depth 2
        # but max_nodes=3 should cap it
        pool.fetch = AsyncMock(side_effect=[
            [{"child_table": "t", "child_id": i} for i in range(10, 15)],
            *[[]] * 5,  # children of children (won't all be reached)
        ])
        result = await get_descendants(pool, "t", 1, max_depth=5, max_nodes=3)
        assert len(result) <= 3


class TestGetAncestors:
    @pytest.mark.asyncio
    async def test_empty_graph(self, pool):
        pool.fetch = AsyncMock(return_value=[])
        result = await get_ancestors(pool, "t", 1)
        assert result == []

    @pytest.mark.asyncio
    async def test_single_parent(self, pool):
        pool.fetch = AsyncMock(side_effect=[
            [{"parent_table": "t", "parent_id": 10}],
            [],
        ])
        result = await get_ancestors(pool, "t", 1)
        assert len(result) == 1
        assert result[0] == ("t", 10, 1)
