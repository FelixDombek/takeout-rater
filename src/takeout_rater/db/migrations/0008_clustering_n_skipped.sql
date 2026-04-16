-- Migration: version 13 → 14
-- Add n_skipped column to clustering_runs to record the number of components
-- that were skipped during a clustering run (due to MemoryError or max_cluster_size).

ALTER TABLE clustering_runs ADD COLUMN n_skipped INTEGER NOT NULL DEFAULT 0;

PRAGMA user_version = 14;
