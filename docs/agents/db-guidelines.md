# Database guidelines

> **Status:** Iteration 11 — `CURRENT_INDEXER_VERSION` bumped to 2 (thumbnail
> regeneration added to the rescan pipeline).
> This document captures the conventions to follow when extending the database
> in future iterations.

---

## Conventions

### Naming

| Object | Convention | Example |
|---|---|---|
| Table | `snake_case`, plural | `assets`, `asset_scores` |
| Column | `snake_case` | `taken_at`, `scorer_id` |
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

- Scores are stored as normalised rows in `asset_scores`: `(asset_id, scorer_id, variant_id, metric_key, value)`.
- Never store scores in columns on the `assets` table.
- This allows adding new scorers without schema migrations.

### Indexer versioning

- `assets.indexer_version` (INTEGER, nullable) tracks which version of the
  indexing pipeline last processed each asset.
- `CURRENT_INDEXER_VERSION` is defined in `src/takeout_rater/db/queries.py`.
- Assets with `indexer_version IS NULL` or `indexer_version < CURRENT_INDEXER_VERSION`
  are candidates for the "Rescan library" job (`missing_only` mode).
- **Non-destructive rule**: rescanning must not modify `asset_scores`,
  `phash`, or `clusters` tables.  Only `assets` columns may be updated.
- When adding a new derived field to `assets`, increment
  `CURRENT_INDEXER_VERSION` so that the UI can prompt the user to rescan.
  Document the bump in the migration file and here.

#### Rescan / upgrade workflow

1. Add the new column via a migration SQL file (e.g. `0005_…`).
   *(If no schema change is needed — e.g. only pipeline logic changes — skip this step.)*
2. Update `CURRENT_INDEXER_VERSION` in `queries.py` (e.g. `1 → 2`).
3. Add population logic in the rescan worker inside `api/jobs.py`.
4. Users navigate to `/jobs` → **Rescan library** → select *Missing only* →
   click **Run Rescan**.  Progress is shown live; the rest of the app remains
   fully navigable during the background job.

#### Version history

| Version | Iteration | What changed |
|---------|-----------|--------------|
| 1 | 8 | Baseline: `indexer_version` column introduced; rescan re-parses sidecar metadata |
| 2 | 11 | Rescan now also regenerates thumbnails (`missing_only`: absent thumbs only; `full`: all thumbs) |

---

## Key tables

```sql
CREATE TABLE assets (
    id                      INTEGER PRIMARY KEY,
    relpath                 TEXT NOT NULL UNIQUE,   -- path relative to takeout root
    filename                TEXT NOT NULL,
    ext                     TEXT NOT NULL,
    size_bytes              INTEGER,
    sha256                  TEXT,                  -- nullable, computed lazily
    taken_at                INTEGER,               -- Unix ts from photoTakenTime.timestamp
    created_at_sidecar      INTEGER,               -- Unix ts from creationTime.timestamp
    width                   INTEGER,               -- nullable until image is decoded
    height                  INTEGER,
    -- Geo from geoData (always present in sidecar; 0.0/0.0 when unknown)
    geo_lat                 REAL,
    geo_lon                 REAL,
    geo_alt                 REAL,
    -- Geo from geoDataExif (optional; present in ~51 % of sidecars)
    geo_exif_lat            REAL,
    geo_exif_lon            REAL,
    geo_exif_alt            REAL,
    -- Sidecar scalar fields
    title                   TEXT,                  -- sidecar title (often == filename)
    description             TEXT,                  -- user description (often empty)
    image_views             INTEGER,               -- imageViews as int (sidecar stores str)
    google_photos_url       TEXT,                  -- url field from sidecar
    -- Google-Photos-specific flags (optional booleans in sidecar)
    favorited               INTEGER,               -- 0/1, NULL when absent
    archived                INTEGER,               -- 0/1, NULL when absent
    trashed                 INTEGER,               -- 0/1, NULL when absent
    -- Upload / origin info (from googlePhotosOrigin, optional)
    origin_type             TEXT,                  -- 'mobileUpload'|'driveSync'|...
    origin_device_type      TEXT,                  -- e.g. 'ANDROID_PHONE' (mobileUpload only)
    origin_device_folder    TEXT,                  -- localFolderName (mobileUpload only)
    app_source_package      TEXT,                  -- appSource.androidPackageName (optional)
    -- Indexing metadata
    sidecar_relpath         TEXT,                  -- path to *.supplemental-metadata.json
    mime                    TEXT,                  -- e.g. 'image/jpeg', 'image/heic'
    indexed_at              INTEGER NOT NULL,
    -- Pipeline versioning (added in migration 0004)
    indexer_version         INTEGER                -- NULL = pre-versioning; set by rescan job
);

CREATE TABLE asset_scores (
    asset_id        INTEGER NOT NULL REFERENCES assets(id),
    scorer_id       TEXT NOT NULL,
    variant_id      TEXT NOT NULL,
    metric_key      TEXT NOT NULL,
    value           REAL NOT NULL,
    scorer_version  TEXT,
    scored_at       INTEGER,
    PRIMARY KEY (asset_id, scorer_id, variant_id, metric_key)
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
WHERE s.scorer_id = 'aesthetic'
  AND s.variant_id = 'laion_v2'
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
WHERE s.scorer_id = 'aesthetic' AND s.variant_id = 'laion_v2' AND s.metric_key = 'aesthetic'
ORDER BY s.value DESC;
```

### Assets needing rescan

```sql
SELECT id, relpath, sidecar_relpath
FROM assets
WHERE indexer_version IS NULL OR indexer_version < :current_version
ORDER BY id;
```
