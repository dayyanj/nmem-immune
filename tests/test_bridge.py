"""Tests for ImmuneBridge — wiring, error isolation, event dispatch."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nmem_immune.bridge import ImmuneBridge, ImmuneBridgeConfig, ImmuneStats


class MockConsolidation:
    """Fake consolidation with hook registration."""

    def __init__(self):
        self.full_cycle_hooks: list[tuple[str, object]] = []
        self.nightly_hooks: list[tuple[str, object]] = []

    def register_full_cycle_step(self, name: str, fn) -> None:
        self.full_cycle_hooks.append((name, fn))

    def register_nightly_step(self, name: str, fn) -> None:
        self.nightly_hooks.append((name, fn))


class MockMemory:
    """Fake MemorySystem with event registration."""

    def __init__(self):
        self.consolidation = MockConsolidation()
        self._handlers: dict[str, list] = {}

    def on(self, event: str):
        def decorator(fn):
            self._handlers.setdefault(event, []).append(fn)
            return fn
        return decorator

    async def emit(self, event: str, data: dict) -> None:
        for handler in self._handlers.get(event, []):
            await handler(data)


@pytest.fixture
def pool():
    return AsyncMock()


@pytest.fixture
def memory():
    return MockMemory()


class TestBridgeConnect:
    def test_rejects_missing_on(self, pool):
        bridge = ImmuneBridge(pool)
        with pytest.raises(TypeError, match="on"):
            bridge.connect(object())

    def test_rejects_missing_consolidation(self, pool):
        obj = MagicMock(spec=["on"])
        bridge = ImmuneBridge(pool)
        with pytest.raises(TypeError, match="consolidation"):
            bridge.connect(obj)

    def test_registers_all_handlers(self, pool, memory):
        bridge = ImmuneBridge(pool)
        bridge.connect(memory)

        assert "journal.added" in memory._handlers
        assert "ltm.saved" in memory._handlers
        assert "shared.saved" in memory._handlers
        # ltm.saved has both skeptic and provenance handlers
        assert len(memory._handlers["ltm.saved"]) == 2

    def test_registers_consolidation_hooks(self, pool, memory):
        bridge = ImmuneBridge(pool)
        bridge.connect(memory)

        hook_names = [name for name, _ in memory.consolidation.full_cycle_hooks]
        assert "immune_drift_detection" in hook_names
        assert "immune_quarantine_review" in hook_names

        nightly_names = [name for name, _ in memory.consolidation.nightly_hooks]
        assert "immune_immunity_decay" in nightly_names

    def test_double_connect_raises(self, pool, memory):
        bridge = ImmuneBridge(pool)
        bridge.connect(memory)
        with pytest.raises(RuntimeError, match="already called"):
            bridge.connect(memory)

    def test_config_disables_handlers(self, pool, memory):
        cfg = ImmuneBridgeConfig(
            skeptic_on_journal_added=False,
            skeptic_on_ltm_saved=False,
            skeptic_on_shared_saved=False,
            drift_on_full_cycle=False,
            quarantine_review_on_full_cycle=False,
            immunity_decay_on_nightly=False,
            provenance_track_on_ltm_saved=False,
        )
        bridge = ImmuneBridge(pool, config=cfg)
        bridge.connect(memory)

        assert len(memory._handlers) == 0
        assert len(memory.consolidation.full_cycle_hooks) == 0
        assert len(memory.consolidation.nightly_hooks) == 0


class TestBridgeErrorIsolation:
    @pytest.mark.asyncio
    async def test_safe_swallows_exceptions(self, pool, memory):
        bridge = ImmuneBridge(pool)
        bridge.connect(memory)

        # Force an error in _audit_entry
        bridge._skeptic = MagicMock()
        bridge._skeptic.evaluate = AsyncMock(side_effect=RuntimeError("boom"))
        bridge._immunity = MagicMock()
        bridge._immunity.check = AsyncMock(return_value=0.0)

        # Should not raise
        await memory.emit("journal.added", {"id": 1, "agent_id": "test"})

        assert bridge.stats.errors >= 1

    @pytest.mark.asyncio
    async def test_audit_skips_missing_id(self, pool, memory):
        bridge = ImmuneBridge(pool)
        bridge.connect(memory)

        bridge._immunity = MagicMock()
        bridge._immunity.check = AsyncMock(return_value=0.0)
        bridge._skeptic = MagicMock()
        bridge._skeptic.evaluate = AsyncMock()

        await memory.emit("journal.added", {"agent_id": "test"})

        # evaluate should not be called when id is missing
        bridge._skeptic.evaluate.assert_not_called()


class TestBridgeStats:
    def test_stats_initialized(self, pool):
        bridge = ImmuneBridge(pool)
        assert bridge.stats.entries_audited == 0
        assert bridge.stats.errors == 0


class TestBridgeManualAPI:
    @pytest.mark.asyncio
    async def test_mark_poisoned_before_connect_raises(self, pool):
        bridge = ImmuneBridge(pool)
        with pytest.raises(RuntimeError, match="not connected"):
            await bridge.mark_poisoned("nmem_long_term_memory", 1)

    @pytest.mark.asyncio
    async def test_mark_poisoned_rejects_invalid_table(self, pool, memory):
        bridge = ImmuneBridge(pool)
        bridge.connect(memory)
        with pytest.raises(ValueError, match="must be one of"):
            await bridge.mark_poisoned("evil_table; DROP TABLE", 1)

    @pytest.mark.asyncio
    async def test_mark_poisoned(self, pool, memory):
        bridge = ImmuneBridge(pool)
        bridge.connect(memory)

        bridge._antidote = MagicMock()
        bridge._antidote.propagate = AsyncMock(return_value={
            "source": ("nmem_long_term_memory", 42),
            "tainted_count": 3,
            "depth_reached": 2,
        })
        bridge._immunity = MagicMock()
        bridge._immunity.check = AsyncMock(return_value=0.0)
        bridge._immunity.learn_from_poison = AsyncMock(return_value=1)

        result = await bridge.mark_poisoned("nmem_long_term_memory", 42)

        assert result["tainted_count"] == 3
        bridge._antidote.propagate.assert_called_once()
        bridge._immunity.learn_from_poison.assert_called_once()
        assert bridge.stats.antidote_propagations == 1
