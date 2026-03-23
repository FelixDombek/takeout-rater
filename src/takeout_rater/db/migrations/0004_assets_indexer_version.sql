-- Migration 0004: add indexer_version to assets
-- Tracks which pipeline version last processed each asset so that
-- the "Rescan library" job can identify stale rows that need reprocessing.
-- NULL means the asset was indexed before versioning was introduced.

ALTER TABLE assets ADD COLUMN indexer_version INTEGER;

PRAGMA user_version = 4;
