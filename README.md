# nmem-immune

**Memory Immune System — poisoning protection for [nmem](https://github.com/dayyanj/nmem).**

**Status: 0.2 — early, pre-1.0.**

An agent that learns from experience can also learn from *bad* experience — a poisoned
fact, a confident-but-wrong lesson, a slow drift away from ground truth. nmem-immune is a
standalone audit layer that watches an nmem memory system for those failure modes and
contains them — without nmem ever knowing it is there.

## Design: one bridge, zero coupling

The only connection to nmem is `ImmuneBridge`, a duck-typed adapter that hooks into the
memory system's lifecycle. It is **fail-open** (an audit error never propagates into nmem),
**import-decoupled** (nmem is never imported), and every behaviour is individually
toggleable.

```python
pool = await asyncpg.create_pool("postgresql://…")
bridge = ImmuneBridge(pool)
bridge.connect(mem)                       # registers hooks into nmem's lifecycle
await bridge.mark_poisoned("nmem_long_term_memory", 42)
print(bridge.stats)
```

## What it does

- **Skeptic** — challenges suspicious writes before they harden into belief.
- **Drift** — detects a memory tier drifting away from its established ground truth.
- **Quarantine** — isolates memories flagged as poisoned so they stop being recalled.
- **Provenance** — tracks where a memory came from, so a bad source can be traced and swept.
- **Antidote** — remediation for confirmed poisoning.

All behaviours are toggleable via `ImmuneBridgeConfig` / `NMEM_IMMUNE_*` env.

## Status

Early (0.x). The bridge + skeptic / drift / quarantine run wired into a live agent
(fail-open). The API may change before 1.0. See the [changelog](CHANGELOG.md).

## The nmem suite

nmem-immune is one library in a family of composable, framework-agnostic cognitive layers for AI agents. Each is standalone — mix in only the ones you need.

| Repo | Layer | What it does |
|------|-------|--------------|
| [nmem](https://github.com/dayyanj/nmem) | Memory | Hierarchical, self-refining cognitive memory — 6 tiers + a consolidation engine |
| [nmem-sym](https://github.com/dayyanj/nmem-sym) | Reasoning | Symbolic cognition — typed knowledge graph, spreading activation, drives, prediction |
| [nmem-sym-sensor](https://github.com/dayyanj/nmem-sym-sensor) | Perception | Sensory memory — unlabeled visual/audio primitives clustered into grounded concepts |
| [nmem-identity](https://github.com/dayyanj/nmem-identity) | Perception | Self-supervised person identity (voice + face) — learns who it's talking to, no manual tagging |
| [nmem-act](https://github.com/dayyanj/nmem-act) | Action | Typed action/outcome contract + tiered autonomy gate — act to learn, safely |
| [nmem-sandbox](https://github.com/dayyanj/nmem-sandbox) | Action | LLM-agnostic computer-use sandbox — a headless desktop a vision model drives |
| [nmem-exchange](https://github.com/dayyanj/nmem-exchange) | Comms | End-to-end-secured message bus — agents talk without sharing memory |
| [nmem-immune](https://github.com/dayyanj/nmem-immune) | Integrity | Memory immune system — poisoning detection, drift, quarantine |
| [nmem-viz](https://github.com/dayyanj/nmem-viz) | Tooling | Real-time 3D "brain" visualization of any nmem agent |

**Just want to run one?** [**nmem-studio**](https://huggingface.co/dayyanj/nmem-studio) is the pull-and-run appliance — a single Docker image that stands up one fully-configured agent from a web wizard (memory + reasoning, plus optional identity and embodied perception), no config files by hand.
