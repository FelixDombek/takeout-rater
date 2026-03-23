-- Migration 0003: index on sha256 for content-hash deduplication
-- Adds a non-unique index so that GROUP BY / MIN queries over sha256 are fast.
-- Schema version is bumped from 2 to 3.

CREATE INDEX IF NOT EXISTS idx_assets_sha256 ON assets (sha256);

PRAGMA user_version = 3;
