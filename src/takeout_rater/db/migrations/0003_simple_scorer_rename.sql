-- Migration: rename the three individual Pillow-only scorers to SimpleScorer variants.
--
-- Before this migration scorer_runs rows for the three merged scorers looked like:
--   scorer_id='blur',       variant_id='default'
--   scorer_id='luminosity', variant_id='default'
--   scorer_id='noise',      variant_id='default'
--
-- After migration they become:
--   scorer_id='simple', variant_id='blur'
--   scorer_id='simple', variant_id='luminosity'
--   scorer_id='simple', variant_id='noise'
--
-- asset_scores rows are joined via scorer_run_id and need no direct update.

UPDATE scorer_runs
SET scorer_id  = 'simple',
    variant_id = 'blur'
WHERE scorer_id = 'blur'
  AND variant_id = 'default';

UPDATE scorer_runs
SET scorer_id  = 'simple',
    variant_id = 'luminosity'
WHERE scorer_id = 'luminosity'
  AND variant_id = 'default';

UPDATE scorer_runs
SET scorer_id  = 'simple',
    variant_id = 'noise'
WHERE scorer_id = 'noise'
  AND variant_id = 'default';

PRAGMA user_version = 8;
