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

CREATE TABLE IF NOT EXISTS asset_scores (
    asset_id        INTEGER NOT NULL REFERENCES assets(id),
    scorer_id       TEXT NOT NULL,
    variant_id      TEXT NOT NULL,
    metric_key      TEXT NOT NULL,
    value           REAL NOT NULL,
    scorer_version  TEXT,
    scored_at       INTEGER,
    PRIMARY KEY (asset_id, scorer_id, variant_id, metric_key)
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
    created_at  INTEGER NOT NULL,
    n_skipped   INTEGER NOT NULL DEFAULT 0
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

CREATE TABLE IF NOT EXISTS clip_embeddings (
    asset_id    INTEGER PRIMARY KEY REFERENCES assets(id),
    embedding   BLOB NOT NULL,
    model_id    TEXT NOT NULL DEFAULT 'ViT-L-14/openai',
    computed_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS image_embeddings (
    asset_id    INTEGER NOT NULL REFERENCES assets(id),
    model_key   TEXT    NOT NULL,   -- search backbone key, e.g. 'siglip2-so400m-16-256'
    dim         INTEGER NOT NULL,   -- vector length (float32 values)
    embedding   BLOB    NOT NULL,   -- L2-normalised little-endian float32 vector
    computed_at INTEGER NOT NULL,
    PRIMARY KEY (model_key, asset_id)
);

CREATE TABLE IF NOT EXISTS clip_user_tags (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    term       TEXT    UNIQUE NOT NULL,
    created_at INTEGER NOT NULL
);

-- ── Facial detection tables ────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS face_detection_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id    TEXT NOT NULL,
    params_json TEXT,
    started_at  INTEGER NOT NULL,
    finished_at INTEGER
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id        INTEGER NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    run_id          INTEGER NOT NULL REFERENCES face_detection_runs(id) ON DELETE CASCADE,
    face_index      INTEGER NOT NULL DEFAULT 0,
    bbox_x1         REAL NOT NULL,
    bbox_y1         REAL NOT NULL,
    bbox_x2         REAL NOT NULL,
    bbox_y2         REAL NOT NULL,
    det_score       REAL,
    embedding       BLOB NOT NULL,
    computed_at     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS face_cluster_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    method      TEXT NOT NULL,
    params_json TEXT,
    detection_run_id INTEGER REFERENCES face_detection_runs(id),
    created_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS face_clusters (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES face_cluster_runs(id) ON DELETE CASCADE,
    label       TEXT,
    created_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS face_cluster_members (
    cluster_id  INTEGER NOT NULL REFERENCES face_clusters(id) ON DELETE CASCADE,
    face_id     INTEGER NOT NULL REFERENCES face_embeddings(id) ON DELETE CASCADE,
    distance    REAL,
    is_representative INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (cluster_id, face_id)
);

-- OCR text + FTS5 index (see 0011_ocr_text.sql)
CREATE TABLE IF NOT EXISTS ocr_text (
    asset_id     INTEGER PRIMARY KEY REFERENCES assets(id) ON DELETE CASCADE,
    text         TEXT    NOT NULL DEFAULT '',
    text_folded  TEXT    NOT NULL DEFAULT '',
    n_lines      INTEGER NOT NULL DEFAULT 0,
    mean_score   REAL,
    source       TEXT    NOT NULL,
    engine       TEXT    NOT NULL,
    processed_at INTEGER NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS ocr_text_fts USING fts5(
    text_folded,
    content = 'ocr_text',
    content_rowid = 'asset_id',
    tokenize = 'trigram'
);

CREATE TRIGGER IF NOT EXISTS ocr_text_ai AFTER INSERT ON ocr_text BEGIN
    INSERT INTO ocr_text_fts (rowid, text_folded) VALUES (new.asset_id, new.text_folded);
END;

CREATE TRIGGER IF NOT EXISTS ocr_text_ad AFTER DELETE ON ocr_text BEGIN
    INSERT INTO ocr_text_fts (ocr_text_fts, rowid, text_folded) VALUES ('delete', old.asset_id, old.text_folded);
END;

CREATE TRIGGER IF NOT EXISTS ocr_text_au AFTER UPDATE ON ocr_text BEGIN
    INSERT INTO ocr_text_fts (ocr_text_fts, rowid, text_folded) VALUES ('delete', old.asset_id, old.text_folded);
    INSERT INTO ocr_text_fts (rowid, text_folded) VALUES (new.asset_id, new.text_folded);
END;

-- Trips and events (schema v18)
CREATE TABLE IF NOT EXISTS events (
    id               INTEGER PRIMARY KEY,
    kind             TEXT NOT NULL CHECK (kind IN ('trip', 'event')),
    parent_id        INTEGER REFERENCES events(id) ON DELETE CASCADE,  -- trip of a nested event
    auto_name        TEXT NOT NULL,
    custom_name      TEXT,                     -- user rename; NULL = use auto_name
    place_label      TEXT,                     -- e.g. "Lisbon, Portugal" when known
    start_ts         INTEGER NOT NULL,         -- Unix ts of first photo
    end_ts           INTEGER NOT NULL,         -- Unix ts of last photo
    n_assets         INTEGER NOT NULL,
    n_days           INTEGER NOT NULL,
    center_lat       REAL,
    center_lon       REAL,
    max_distance_km  REAL,                     -- furthest geotagged photo from home
    cover_asset_id   INTEGER REFERENCES assets(id) ON DELETE SET NULL,
    detected_at      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS event_assets (
    event_id  INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    asset_id  INTEGER NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    PRIMARY KEY (event_id, asset_id)
);

CREATE INDEX IF NOT EXISTS idx_events_start_ts        ON events (start_ts);
CREATE INDEX IF NOT EXISTS idx_events_parent_id       ON events (parent_id);
CREATE INDEX IF NOT EXISTS idx_event_assets_asset_id  ON event_assets (asset_id);

-- Smart album suggestions (schema v19, see 0013_smart_albums.sql)
CREATE TABLE IF NOT EXISTS smart_albums (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,  -- never reused: stale pages must not hit a new album
    auto_name       TEXT NOT NULL,
    custom_name     TEXT,                       -- user rename; NULL = use auto_name
    terms           TEXT NOT NULL DEFAULT '[]', -- JSON [[term, score], ...], best first
    kept            INTEGER NOT NULL DEFAULT 0, -- 1 = saved by the user, survives re-runs
    n_assets        INTEGER NOT NULL,
    cohesion        REAL,                       -- mean cosine similarity of members to centroid
    cover_asset_id  INTEGER REFERENCES assets(id) ON DELETE SET NULL,
    created_at      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS smart_album_assets (
    smart_album_id  INTEGER NOT NULL REFERENCES smart_albums(id) ON DELETE CASCADE,
    asset_id        INTEGER NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    similarity      REAL NOT NULL,              -- cosine similarity to the album centroid
    PRIMARY KEY (smart_album_id, asset_id)
);

CREATE INDEX IF NOT EXISTS idx_smart_album_assets_asset_id ON smart_album_assets (asset_id);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_assets_taken_at         ON assets (taken_at);
CREATE INDEX IF NOT EXISTS idx_assets_indexed_at       ON assets (indexed_at);
CREATE INDEX IF NOT EXISTS idx_assets_sha256           ON assets (sha256);
CREATE INDEX IF NOT EXISTS idx_asset_paths_asset_id    ON asset_paths (asset_id);
CREATE INDEX IF NOT EXISTS idx_asset_scores_asset_id   ON asset_scores (asset_id);
CREATE INDEX IF NOT EXISTS idx_asset_scores_scorer     ON asset_scores (scorer_id, variant_id, metric_key);
CREATE INDEX IF NOT EXISTS idx_album_assets_asset_id   ON album_assets (asset_id);
CREATE INDEX IF NOT EXISTS idx_clusters_run_id         ON clusters (run_id);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_asset_id ON face_embeddings(asset_id);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_run_id   ON face_embeddings(run_id);
CREATE INDEX IF NOT EXISTS idx_face_clusters_run_id     ON face_clusters(run_id);
CREATE INDEX IF NOT EXISTS idx_face_cluster_members_face_id ON face_cluster_members(face_id);
CREATE INDEX IF NOT EXISTS idx_image_embeddings_asset_id ON image_embeddings (asset_id);

-- Reverse-geocoded place names (added in schema v16, see 0010_asset_places.sql)
CREATE TABLE IF NOT EXISTS asset_places (
    asset_id         INTEGER PRIMARY KEY REFERENCES assets(id) ON DELETE CASCADE,
    lat              REAL NOT NULL,         -- coordinates the lookup used
    lon              REAL NOT NULL,
    country_code     TEXT,                  -- ISO 3166-1 alpha-2, e.g. 'PT'
    country          TEXT,                  -- English name, e.g. 'Portugal'
    admin1           TEXT,                  -- state / region / province
    admin2           TEXT,                  -- county / district
    city             TEXT,                  -- populated place (districts merged into cities)
    distance_km      REAL,                  -- distance to the nearest gazetteer entry
    geocoder_version INTEGER NOT NULL,
    geocoded_at      INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_asset_places_place ON asset_places (country_code, admin1, city);
CREATE INDEX IF NOT EXISTS idx_asset_places_city  ON asset_places (city);

PRAGMA user_version = 19;
