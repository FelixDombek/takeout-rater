# Database guidelines

> **Status:** Iteration 0 — schema and migration tooling are TBD.  This document captures the conventions to follow once they are implemented in Iteration 1.

---

## Conventions

### Naming

| Object | Convention | Example |
|---|---|---|
| Table | `snake_case`, plural | `assets`, `scorer_runs`, `asset_scores` |
| Column | `snake_case` | `taken_at`, `scorer_run_id` |
| Index | `idx_<table>_<column(s)>` | `idx_asset_scores_asset_id` |
| Foreign key | `fk_<table>_<ref_table>` | `fk_asset_scores_assets` |

### Timestamps

- Store as **Unix timestamps (INTEGER)** for portability.
- Display formatting is the UI's responsibility.

### Nullable columns

- Prefer `NULL` over sentinel values (empty string, -1, 0).
- Document why a column is nullable in a comment or migration note.

### Schema changes

- Every schema change requires a **migration script** (numbered, idempotent).
- Migration scripts live in `src/takeout_rater/db/migrations/`.
- Naming: `NNNN_<short_description>.sql` (e.g. `0001_initial_schema.sql`).
- Migrations are run automatically on app startup if the DB version is behind.

### Score storage

- Scores are stored as normalised rows in `asset_scores`: `(asset_id, scorer_run_id, metric_key, value)`.
- Never store scores in columns on the `assets` table.
- This allows adding new scorers without schema migrations.

---

## Key tables (planned for Iteration 1)

```sql
CREATE TABLE assets (
    id                  INTEGER PRIMARY KEY,
    relpath             TEXT NOT NULL UNIQUE,
    filename            TEXT NOT NULL,
    ext                 TEXT NOT NULL,
    size_bytes          INTEGER,
    sha256              TEXT,
    taken_at            INTEGER,       -- Unix timestamp, from sidecar
    width               INTEGER,
    height              INTEGER,
    geo_lat             REAL,
    geo_lon             REAL,
    geo_alt             REAL,
    google_photos_url   TEXT,
    origin_device_type  TEXT,
    sidecar_relpath     TEXT,
    mime                TEXT,
    indexed_at          INTEGER NOT NULL
);

CREATE TABLE scorer_runs (
    id              INTEGER PRIMARY KEY,
    scorer_id       TEXT NOT NULL,
    variant_id      TEXT NOT NULL,
    scorer_version  TEXT,
    params_json     TEXT,
    params_hash     TEXT,
    started_at      INTEGER,
    finished_at     INTEGER
);

CREATE TABLE asset_scores (
    asset_id        INTEGER NOT NULL REFERENCES assets(id),
    scorer_run_id   INTEGER NOT NULL REFERENCES scorer_runs(id),
    metric_key      TEXT NOT NULL,
    value           REAL NOT NULL,
    PRIMARY KEY (asset_id, scorer_run_id, metric_key)
);

CREATE TABLE phash (
    asset_id    INTEGER PRIMARY KEY REFERENCES assets(id),
    phash_hex   TEXT NOT NULL,
    algo        TEXT NOT NULL DEFAULT 'phash',
    computed_at INTEGER NOT NULL
);

CREATE TABLE clusters (
    id          INTEGER PRIMARY KEY,
    method      TEXT NOT NULL,
    params_json TEXT,
    created_at  INTEGER NOT NULL
);

CREATE TABLE cluster_members (
    cluster_id        INTEGER NOT NULL REFERENCES clusters(id),
    asset_id          INTEGER NOT NULL REFERENCES assets(id),
    distance          REAL,
    is_representative INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (cluster_id, asset_id)
);
```

---

## Query patterns

### Top-N by aesthetic score

```sql
SELECT a.id, a.relpath, s.value AS aesthetic
FROM assets a
JOIN asset_scores s ON s.asset_id = a.id
JOIN scorer_runs r ON r.id = s.scorer_run_id
WHERE r.scorer_id = 'aesthetic'
  AND r.variant_id = 'laion_v2'
  AND s.metric_key = 'aesthetic'
ORDER BY s.value DESC
LIMIT 100;
```

### Cluster representatives

```sql
SELECT a.id, a.relpath, s.value AS aesthetic
FROM assets a
JOIN cluster_members cm ON cm.asset_id = a.id AND cm.is_representative = 1
JOIN asset_scores s ON s.asset_id = a.id
JOIN scorer_runs r ON r.id = s.scorer_run_id
WHERE r.scorer_id = 'aesthetic' AND s.metric_key = 'aesthetic'
ORDER BY s.value DESC;
```
