# ADR-0002: SQLite as metadata store

**Status:** Accepted  
**Date:** 2026-03

---

## Context

The app needs to store:
- Asset metadata (200k+ rows)
- Scorer run provenance
- Per-asset metric scores (potentially millions of rows across scorers)
- Cluster membership

The store must be:
- Portable (single file, Windows-friendly)
- Queryable (filter, sort, join)
- Not require a server process

---

## Decision

Use **SQLite** as the single metadata store, accessed via the Python standard library (`sqlite3`) or a thin ORM layer (to be decided in Iteration 1).

Key schema design:
- `assets` — one row per image file
- `scorer_runs` — provenance of each scoring job (scorer id, variant, version, params hash)
- `asset_scores` — normalised metric rows: `(asset_id, scorer_run_id, metric_key, value)`
- `clusters` + `cluster_members` — near-duplicate groups

Storing scores as `(asset_id, scorer_run_id, metric_key, value)` rows means:
- Adding new scorers never requires schema migrations on the scores table.
- Multiple variants of the same scorer coexist naturally.
- Historical runs are preserved.

---

## Consequences

**Positive:**
- Zero-dependency server; single file easy to back up.
- Full SQL expressiveness for filtering/sorting/joining.
- Normalised score rows support arbitrary future scorers.

**Negative:**
- SQLite write concurrency is limited; background scoring jobs must serialise writes (acceptable for single-user local tool).
- Very large score tables (50M+ rows) may need indexing care.

---

## Alternatives considered

| Option | Rejected because |
|---|---|
| PostgreSQL / MySQL | Requires server process; too heavy for a local tool |
| DuckDB | Better analytics, but less tooling/ORM support |
| JSON files per asset | No efficient querying/sorting across 200k assets |
| Flat CSV | Same query limitations |
