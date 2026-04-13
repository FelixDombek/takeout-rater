-- Incremental migration: schema version 6 → 7.
-- Adds the `diameter` column to the `clusters` table so the cluster builder
-- can store the intra-cluster diameter (max pairwise Hamming distance).

ALTER TABLE clusters ADD COLUMN diameter REAL;

PRAGMA user_version = 7;
