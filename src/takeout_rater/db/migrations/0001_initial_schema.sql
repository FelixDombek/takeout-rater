-- Migration 0001: initial schema
-- All tables use CREATE TABLE IF NOT EXISTS for idempotency.
-- Schema version is tracked via PRAGMA user_version.

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
    indexed_at              INTEGER NOT NULL
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

CREATE TABLE IF NOT EXISTS clusters (
    id          INTEGER PRIMARY KEY,
    method      TEXT NOT NULL,
    params_json TEXT,
    created_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS cluster_members (
    cluster_id        INTEGER NOT NULL REFERENCES clusters(id),
    asset_id          INTEGER NOT NULL REFERENCES assets(id),
    distance          REAL,
    is_representative INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (cluster_id, asset_id)
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_assets_taken_at         ON assets (taken_at);
CREATE INDEX IF NOT EXISTS idx_assets_indexed_at       ON assets (indexed_at);
CREATE INDEX IF NOT EXISTS idx_asset_scores_asset_id   ON asset_scores (asset_id);
CREATE INDEX IF NOT EXISTS idx_asset_scores_run_metric ON asset_scores (scorer_run_id, metric_key);
CREATE INDEX IF NOT EXISTS idx_album_assets_asset_id   ON album_assets (asset_id);

PRAGMA user_version = 1;
