"""Provenance link utilities — read/write for the derived-from graph.

All functions take an asyncpg.Pool and operate on the immune_provenance_links
table. Used by the antidote propagator for BFS walks and by the bridge for
recording promotion links.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncpg

log = logging.getLogger(__name__)


async def record_link(
    pool: asyncpg.Pool,
    parent_table: str,
    parent_id: int,
    child_table: str,
    child_id: int,
    link_type: str = "derived_from",
) -> int | None:
    """Insert a provenance link. Returns link ID, or None on conflict."""
    row = await pool.fetchrow(
        """
        INSERT INTO immune_provenance_links
            (parent_table, parent_id, child_table, child_id, link_type)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (parent_table, parent_id, child_table, child_id, link_type)
            DO NOTHING
        RETURNING id
        """,
        parent_table,
        parent_id,
        child_table,
        child_id,
        link_type,
    )
    return row["id"] if row else None


_MAX_BFS_NODES = 1000  # Hard cap to prevent runaway traversals.


async def get_descendants(
    pool: asyncpg.Pool,
    table: str,
    record_id: int,
    max_depth: int = 10,
    max_nodes: int = _MAX_BFS_NODES,
) -> list[tuple[str, int, int]]:
    """BFS walk downward through provenance links.

    Returns list of (child_table, child_id, depth).
    Stops after visiting max_nodes to bound resource usage.
    """
    visited: set[tuple[str, int]] = {(table, record_id)}
    result: list[tuple[str, int, int]] = []
    frontier: list[tuple[str, int, int]] = [(table, record_id, 0)]

    while frontier:
        if len(result) >= max_nodes:
            log.warning(
                "get_descendants hit max_nodes=%d from %s:%d, stopping early",
                max_nodes, table, record_id,
            )
            break
        current_table, current_id, depth = frontier.pop(0)
        if depth > 0:
            result.append((current_table, current_id, depth))
        if depth >= max_depth:
            continue

        rows = await pool.fetch(
            """
            SELECT child_table, child_id
            FROM immune_provenance_links
            WHERE parent_table = $1 AND parent_id = $2
            """,
            current_table,
            current_id,
        )
        for row in rows:
            key = (row["child_table"], row["child_id"])
            if key not in visited:
                visited.add(key)
                frontier.append((row["child_table"], row["child_id"], depth + 1))

    return result


async def get_ancestors(
    pool: asyncpg.Pool,
    table: str,
    record_id: int,
    max_depth: int = 10,
    max_nodes: int = _MAX_BFS_NODES,
) -> list[tuple[str, int, int]]:
    """BFS walk upward through provenance links.

    Returns list of (parent_table, parent_id, depth).
    Stops after visiting max_nodes to bound resource usage.
    """
    visited: set[tuple[str, int]] = {(table, record_id)}
    result: list[tuple[str, int, int]] = []
    frontier: list[tuple[str, int, int]] = [(table, record_id, 0)]

    while frontier:
        if len(result) >= max_nodes:
            log.warning(
                "get_ancestors hit max_nodes=%d from %s:%d, stopping early",
                max_nodes, table, record_id,
            )
            break
        current_table, current_id, depth = frontier.pop(0)
        if depth > 0:
            result.append((current_table, current_id, depth))
        if depth >= max_depth:
            continue

        rows = await pool.fetch(
            """
            SELECT parent_table, parent_id
            FROM immune_provenance_links
            WHERE child_table = $1 AND child_id = $2
            """,
            current_table,
            current_id,
        )
        for row in rows:
            key = (row["parent_table"], row["parent_id"])
            if key not in visited:
                visited.add(key)
                frontier.append((row["parent_table"], row["parent_id"], depth + 1))

    return result
