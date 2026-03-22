-- Migration 0002: view_presets table
-- Stores named filter + sort combinations for quick recall in the browse UI.
-- Schema version is bumped from 1 to 2.

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

PRAGMA user_version = 2;
