-- Migration: schema version 9 → 10
-- Adds clip_embeddings table for CLIP ViT-L/14 image embeddings used by
-- semantic search.

CREATE TABLE IF NOT EXISTS clip_embeddings (
    asset_id    INTEGER PRIMARY KEY REFERENCES assets(id),
    embedding   BLOB NOT NULL,
    model_id    TEXT NOT NULL DEFAULT 'ViT-L-14/openai',
    computed_at INTEGER NOT NULL
);

PRAGMA user_version = 10;
