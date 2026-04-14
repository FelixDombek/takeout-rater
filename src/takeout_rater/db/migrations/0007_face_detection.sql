-- Migration: schema version 11 → 12
-- Adds tables for facial detection, face embeddings, and face (person) clustering.

-- Each detection run records the model pack used and its parameters.
CREATE TABLE IF NOT EXISTS face_detection_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id    TEXT NOT NULL,       -- e.g. 'buffalo_l', 'buffalo_sc'
    params_json TEXT,                -- JSON with det_size, det_thresh, etc.
    started_at  INTEGER NOT NULL,
    finished_at INTEGER
);

-- Per-face records: one row per detected face in an asset.
-- An asset with three faces produces three rows.
CREATE TABLE IF NOT EXISTS face_embeddings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id        INTEGER NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    run_id          INTEGER NOT NULL REFERENCES face_detection_runs(id) ON DELETE CASCADE,
    face_index      INTEGER NOT NULL DEFAULT 0,  -- 0-based index within the asset
    bbox_x1         REAL NOT NULL,
    bbox_y1         REAL NOT NULL,
    bbox_x2         REAL NOT NULL,
    bbox_y2         REAL NOT NULL,
    det_score       REAL,            -- detection confidence [0, 1]
    embedding       BLOB NOT NULL,   -- 512 float32 values = 2048 bytes (ArcFace)
    computed_at     INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_face_embeddings_asset_id ON face_embeddings(asset_id);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_run_id   ON face_embeddings(run_id);

-- Face cluster runs (analogous to clustering_runs).
CREATE TABLE IF NOT EXISTS face_cluster_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    method      TEXT NOT NULL,       -- e.g. 'dbscan'
    params_json TEXT,                -- JSON with eps, min_samples, etc.
    detection_run_id INTEGER REFERENCES face_detection_runs(id),
    created_at  INTEGER NOT NULL
);

-- Person clusters: groups of face embeddings that belong to the same person.
CREATE TABLE IF NOT EXISTS face_clusters (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES face_cluster_runs(id) ON DELETE CASCADE,
    label       TEXT,                -- user-assigned name (NULL until labelled)
    created_at  INTEGER NOT NULL
);

-- Members of a face cluster: links face_embeddings rows to face_clusters.
CREATE TABLE IF NOT EXISTS face_cluster_members (
    cluster_id  INTEGER NOT NULL REFERENCES face_clusters(id) ON DELETE CASCADE,
    face_id     INTEGER NOT NULL REFERENCES face_embeddings(id) ON DELETE CASCADE,
    distance    REAL,                -- distance from cluster centre
    is_representative INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (cluster_id, face_id)
);

CREATE INDEX IF NOT EXISTS idx_face_clusters_run_id ON face_clusters(run_id);
CREATE INDEX IF NOT EXISTS idx_face_cluster_members_face_id ON face_cluster_members(face_id);

PRAGMA user_version = 12;
