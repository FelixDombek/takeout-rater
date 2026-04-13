-- Consolidated schema (schema version 6).
-- This is a breaking-change release: existing databases at any prior version
-- (1-5) are not migrated.  The application will refuse to open them and will
-- require a complete re-scan of the Takeout folder to rebuild the library.
--
-- All tables use CREATE TABLE IF NOT EXISTS for idempotency on fresh installs.

CREATE TABLE IF NOT EXISTS assets (
    id                      INTEGER PRIMARY KEY,
    relpath                 TEXT NOT NULL UNIQUE,
    filename                TEXT NOT NULL,
    ext                     TEXT NOT NULL,
    size_bytes              INTEGER,
    sha256                  TEXT,                   -- nullable, computed lazily
    taken_at                INTEGER,                -- Unix ts from photoTakenTime.timestamp
    created_at_sidecar      INTEGER,                -- Unix ts from creationTime.timestamp
    width                   INTEGER,                -- nullable until image is decoded
    height                  INTEGER,
    -- Geo from geoData (0.0/0.0 when unknown)
    geo_lat                 REAL,
    geo_lon                 REAL,
    geo_alt                 REAL,
    -- Geo from geoDataExif (optional; ~51 % of sidecars)
    geo_exif_lat            REAL,
    geo_exif_lon            REAL,
    geo_exif_alt            REAL,
    -- Sidecar scalar fields
    title                   TEXT,
    description             TEXT,
    image_views             INTEGER,
    google_photos_url       TEXT,
    -- Google-Photos flags (optional booleans stored as 0/1)
    favorited               INTEGER,
    archived                INTEGER,
    trashed                 INTEGER,
    -- Upload / origin info
    origin_type             TEXT,
    origin_device_type      TEXT,
    origin_device_folder    TEXT,
    app_source_package      TEXT,
    -- Indexing metadata
    sidecar_relpath         TEXT,
    mime                    TEXT,
    indexed_at              INTEGER NOT NULL,
    -- Pipeline version that last processed this asset (NULL = pre-versioning)
    indexer_version         INTEGER
);

CREATE TABLE IF NOT EXISTS asset_paths (
    id         INTEGER PRIMARY KEY,
    asset_id   INTEGER NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    relpath    TEXT NOT NULL UNIQUE,
    indexed_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS albums (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    relpath     TEXT NOT NULL UNIQUE,  -- path relative to takeout root
    indexed_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS album_assets (
    album_id    INTEGER NOT NULL REFERENCES albums(id),
    asset_id    INTEGER NOT NULL REFERENCES assets(id),
    PRIMARY KEY (album_id, asset_id)
);

CREATE TABLE IF NOT EXISTS scorer_runs (
    id              INTEGER PRIMARY KEY,
    scorer_id       TEXT NOT NULL,
    variant_id      TEXT NOT NULL,
    scorer_version  TEXT,
    params_json     TEXT,
    params_hash     TEXT,
    started_at      INTEGER,
    finished_at     INTEGER
);

CREATE TABLE IF NOT EXISTS asset_scores (
    asset_id        INTEGER NOT NULL REFERENCES assets(id),
    scorer_run_id   INTEGER NOT NULL REFERENCES scorer_runs(id),
    metric_key      TEXT NOT NULL,
    value           REAL NOT NULL,
    PRIMARY KEY (asset_id, scorer_run_id, metric_key)
);

CREATE TABLE IF NOT EXISTS phash (
    asset_id    INTEGER PRIMARY KEY REFERENCES assets(id),
    phash_hex   TEXT NOT NULL,
    algo        TEXT NOT NULL DEFAULT 'phash',
    computed_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS clustering_runs (
    id          INTEGER PRIMARY KEY,
    method      TEXT NOT NULL,
    params_json TEXT,
    created_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS clusters (
    id          INTEGER PRIMARY KEY,
    method      TEXT NOT NULL,
    params_json TEXT,
    run_id      INTEGER NOT NULL REFERENCES clustering_runs(id),
    created_at  INTEGER NOT NULL,
    diameter    REAL    -- max pairwise Hamming distance across cluster members
);

CREATE TABLE IF NOT EXISTS cluster_members (
    cluster_id        INTEGER NOT NULL REFERENCES clusters(id),
    asset_id          INTEGER NOT NULL REFERENCES assets(id),
    distance          REAL,
    is_representative INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (cluster_id, asset_id)
);

CREATE TABLE IF NOT EXISTS view_presets (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    sort_by     TEXT,           -- "scorer_id:metric_key" or NULL (date order)
    favorited   INTEGER,        -- 1 = favourites only, NULL = no filter
    min_score   REAL,           -- inclusive lower bound on sort metric, NULL = unbounded
    max_score   REAL,           -- inclusive upper bound on sort metric, NULL = unbounded
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_assets_taken_at         ON assets (taken_at);
CREATE INDEX IF NOT EXISTS idx_assets_indexed_at       ON assets (indexed_at);
CREATE INDEX IF NOT EXISTS idx_assets_sha256           ON assets (sha256);
CREATE INDEX IF NOT EXISTS idx_asset_paths_asset_id    ON asset_paths (asset_id);
CREATE INDEX IF NOT EXISTS idx_asset_scores_asset_id   ON asset_scores (asset_id);
CREATE INDEX IF NOT EXISTS idx_asset_scores_run_metric ON asset_scores (scorer_run_id, metric_key);
CREATE INDEX IF NOT EXISTS idx_album_assets_asset_id   ON album_assets (asset_id);
CREATE INDEX IF NOT EXISTS idx_clusters_run_id         ON clusters (run_id);

PRAGMA user_version = 9;
