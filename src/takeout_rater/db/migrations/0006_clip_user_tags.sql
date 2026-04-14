-- Migration: schema version 10 → 11
-- Adds clip_user_tags table for user-defined CLIP vocabulary terms.
-- These terms are compared against image embeddings alongside the predefined
-- vocabulary and displayed in the CLIP tab of the asset detail view.

CREATE TABLE IF NOT EXISTS clip_user_tags (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    term       TEXT    UNIQUE NOT NULL,
    created_at INTEGER NOT NULL
);

PRAGMA user_version = 11;
