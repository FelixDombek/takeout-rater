-- Migration: version 14 → 15
-- Adds a model-keyed image_embeddings table for search backbones other than
-- CLIP ViT-L/14 (e.g. SigLIP 2).  CLIP L/14 vectors stay in clip_embeddings,
-- because the LAION, cafe_style, clip_iqa and clip_vocab scorers, clustering,
-- the embedding map and image similarity all depend on that exact space.

CREATE TABLE IF NOT EXISTS image_embeddings (
    asset_id    INTEGER NOT NULL REFERENCES assets(id),
    model_key   TEXT    NOT NULL,   -- search backbone key, e.g. 'siglip2-so400m-16-256'
    dim         INTEGER NOT NULL,   -- vector length (float32 values)
    embedding   BLOB    NOT NULL,   -- L2-normalised little-endian float32 vector
    computed_at INTEGER NOT NULL,
    PRIMARY KEY (model_key, asset_id)
);

CREATE INDEX IF NOT EXISTS idx_image_embeddings_asset_id ON image_embeddings (asset_id);

PRAGMA user_version = 15;
