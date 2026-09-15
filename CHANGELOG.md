# Changelog

All notable changes to nmem-immune are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] — 2026-09-08

**Config unified + docs.** Still pre-1.0.

### Added

- A real README (previously empty).
- Configuration guide + a model-introspecting reference generator.

### Changed

- Migrated to **pydantic-settings** (`NMEM_IMMUNE_*`).

## [0.1.0] — initial

Memory immune system for nmem: **`ImmuneBridge`** (duck-typed, fail-open,
import-decoupled — nmem is never imported) plus **skeptic / drift / quarantine /
provenance / antidote** audit modules.
