-- Migration: version 14 → 15
-- Add asset_places: offline reverse-geocoded place names (GeoNames) for
-- assets with usable GPS coordinates.  One row per processed asset; rows whose
-- coordinates are far from any populated place keep NULL name columns so the
-- geocode job does not retry them.

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

PRAGMA user_version = 15;
