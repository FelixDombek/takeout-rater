-- Migration: version 17 → 18
-- Trips and events detected from photo timestamps and GPS (takeout_rater.events).
-- The tables are fully rebuilt on every detection run; only custom_name is
-- carried over to the matching re-detected trip/event.

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

PRAGMA user_version = 18;
