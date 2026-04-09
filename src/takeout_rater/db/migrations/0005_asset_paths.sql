-- Migration 0005: asset_paths table + deduplicate assets by SHA-256
--
-- Goal: each unique binary file (identified by SHA-256) should have exactly
-- one row in the assets table.  Duplicate paths (e.g., the same photo appearing
-- in both a "Photos from YYYY" folder and an album folder) are stored in the
-- new asset_paths table instead.
--
-- The asset with the lowest id for a given sha256 is the "canonical" asset.
-- All other asset rows with the same sha256 are "secondary" and are merged
-- into the canonical one during this migration.

-- 1. Create asset_paths table for storing alias paths of canonical assets.
CREATE TABLE IF NOT EXISTS asset_paths (
    id         INTEGER PRIMARY KEY,
    asset_id   INTEGER NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    relpath    TEXT NOT NULL UNIQUE,
    indexed_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_asset_paths_asset_id ON asset_paths (asset_id);

-- 2. Insert secondary (duplicate) relpaths into asset_paths, pointing to
--    the canonical (minimum-id) asset for each sha256 group.
INSERT OR IGNORE INTO asset_paths (asset_id, relpath, indexed_at)
SELECT
    c.canonical_id,
    a.relpath,
    a.indexed_at
FROM assets a
INNER JOIN (
    SELECT sha256, MIN(id) AS canonical_id
    FROM assets
    WHERE sha256 IS NOT NULL
    GROUP BY sha256
) c ON a.sha256 = c.sha256 AND a.id != c.canonical_id;

-- 3. Transfer album memberships from secondary assets to their canonical equivalents.
INSERT OR IGNORE INTO album_assets (album_id, asset_id)
SELECT aa.album_id, c.canonical_id
FROM album_assets aa
INNER JOIN assets a ON a.id = aa.asset_id
INNER JOIN (
    SELECT sha256, MIN(id) AS canonical_id
    FROM assets
    WHERE sha256 IS NOT NULL
    GROUP BY sha256
) c ON a.sha256 = c.sha256 AND a.id != c.canonical_id;

-- 4. Remove album memberships for secondary assets.
DELETE FROM album_assets
WHERE asset_id IN (
    SELECT a.id FROM assets a
    INNER JOIN (
        SELECT sha256, MIN(id) AS canonical_id
        FROM assets WHERE sha256 IS NOT NULL GROUP BY sha256
    ) c ON a.sha256 = c.sha256 AND a.id != c.canonical_id
);

-- 5. Remove asset_scores for secondary assets (the canonical asset keeps its own scores).
DELETE FROM asset_scores
WHERE asset_id IN (
    SELECT a.id FROM assets a
    INNER JOIN (
        SELECT sha256, MIN(id) AS canonical_id
        FROM assets WHERE sha256 IS NOT NULL GROUP BY sha256
    ) c ON a.sha256 = c.sha256 AND a.id != c.canonical_id
);

-- 6. Transfer phash from secondary to canonical (INSERT OR IGNORE so the canonical's
--    own phash, if it exists, is preserved).
INSERT OR IGNORE INTO phash (asset_id, phash_hex, algo, computed_at)
SELECT c.canonical_id, p.phash_hex, p.algo, p.computed_at
FROM phash p
INNER JOIN assets a ON a.id = p.asset_id
INNER JOIN (
    SELECT sha256, MIN(id) AS canonical_id
    FROM assets WHERE sha256 IS NOT NULL GROUP BY sha256
) c ON a.sha256 = c.sha256 AND a.id != c.canonical_id;

-- 7. Remove phash rows for secondary assets.
DELETE FROM phash
WHERE asset_id IN (
    SELECT a.id FROM assets a
    INNER JOIN (
        SELECT sha256, MIN(id) AS canonical_id
        FROM assets WHERE sha256 IS NOT NULL GROUP BY sha256
    ) c ON a.sha256 = c.sha256 AND a.id != c.canonical_id
);

-- 8. Remove cluster memberships for secondary assets.
DELETE FROM cluster_members
WHERE asset_id IN (
    SELECT a.id FROM assets a
    INNER JOIN (
        SELECT sha256, MIN(id) AS canonical_id
        FROM assets WHERE sha256 IS NOT NULL GROUP BY sha256
    ) c ON a.sha256 = c.sha256 AND a.id != c.canonical_id
);

-- 9. Delete secondary asset rows (all FK references cleaned above).
DELETE FROM assets
WHERE id IN (
    SELECT a.id FROM assets a
    INNER JOIN (
        SELECT sha256, MIN(id) AS canonical_id
        FROM assets WHERE sha256 IS NOT NULL GROUP BY sha256
    ) c ON a.sha256 = c.sha256 AND a.id != c.canonical_id
);

PRAGMA user_version = 5;
