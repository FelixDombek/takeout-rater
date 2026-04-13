-- Migration: version 8 → 9
-- Introduces clustering_runs table so multiple clustering runs with different
-- settings can coexist.  Existing cluster data is cleared (acceptable per
-- design decision: users must re-run clustering after upgrading).

DELETE FROM cluster_members;
DELETE FROM clusters;

CREATE TABLE IF NOT EXISTS clustering_runs (
    id          INTEGER PRIMARY KEY,
    method      TEXT NOT NULL,
    params_json TEXT,
    created_at  INTEGER NOT NULL
);

-- SQLite does not allow adding a NOT NULL column without a DEFAULT via ALTER TABLE,
-- so run_id is added as nullable here.  All new cluster rows are inserted via
-- insert_cluster() in queries.py which always receives a run_id from the builder;
-- any NULL run_id in this column would only arise from a direct SQL insert that
-- bypasses the application layer, which is not supported.
ALTER TABLE clusters ADD COLUMN run_id INTEGER REFERENCES clustering_runs(id);

CREATE INDEX IF NOT EXISTS idx_clusters_run_id ON clusters (run_id);

PRAGMA user_version = 9;
