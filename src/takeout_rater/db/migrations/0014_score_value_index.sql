-- Migration: version 18 → 19
-- Index score values per metric so "sort by score" can walk the index in
-- order instead of sorting every scored asset on each browse page (and so
-- min/max score ranges become index range scans).  asset_id is included as
-- the tie-breaker of the browse order, which makes the index covering.
-- It starts with the same columns as idx_asset_scores_scorer, so that index
-- is redundant and dropped.
--
-- (File number 0014 because PR #186 uses 0013; whichever merges second
-- renumbers its schema version.)

CREATE INDEX IF NOT EXISTS idx_asset_scores_metric_value
    ON asset_scores (scorer_id, variant_id, metric_key, value, asset_id);
DROP INDEX IF EXISTS idx_asset_scores_scorer;

PRAGMA user_version = 19;
