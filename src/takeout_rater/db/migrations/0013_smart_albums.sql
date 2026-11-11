-- Migration: version 18 → 19
-- Smart album suggestions (takeout_rater.smart_albums): CLIP embeddings are
-- clustered and each cluster is named after its best-matching clip_vocab terms.
-- Suggestions are replaced on every run; albums the user kept (or renamed)
-- survive re-runs.

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

PRAGMA user_version = 19;
