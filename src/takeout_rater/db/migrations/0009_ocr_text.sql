-- Migration: version 14 → 15
-- OCR text extracted from photos (RapidOCR) plus an FTS5 index for text search.
--
-- ocr_text holds one row per processed asset.  Assets in which no text was found
-- still get a row (with text = '') so the OCR job can skip them on the next run.
-- ocr_text_fts is an external-content FTS5 table kept in sync by triggers.  It uses
-- the trigram tokenizer so any substring of 3+ characters matches (German compounds
-- like "Hauptbahnhof" are found by "bahn"); matching is case-insensitive.
-- It indexes text_folded, a copy of text with diacritics stripped character by
-- character (same length, so match offsets map back onto text): "brot" finds
-- "Brötchen".  The application computes text_folded when writing.
-- Writers must use INSERT ... ON CONFLICT DO UPDATE (not INSERT OR REPLACE):
-- REPLACE deletes do not fire the delete trigger unless recursive_triggers is on.

CREATE TABLE IF NOT EXISTS ocr_text (
    asset_id     INTEGER PRIMARY KEY REFERENCES assets(id) ON DELETE CASCADE,
    text         TEXT    NOT NULL DEFAULT '',
    text_folded  TEXT    NOT NULL DEFAULT '',
    n_lines      INTEGER NOT NULL DEFAULT 0,
    mean_score   REAL,
    source       TEXT    NOT NULL,
    engine       TEXT    NOT NULL,
    processed_at INTEGER NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS ocr_text_fts USING fts5(
    text_folded,
    content = 'ocr_text',
    content_rowid = 'asset_id',
    tokenize = 'trigram'
);

CREATE TRIGGER IF NOT EXISTS ocr_text_ai AFTER INSERT ON ocr_text BEGIN
    INSERT INTO ocr_text_fts (rowid, text_folded) VALUES (new.asset_id, new.text_folded);
END;

CREATE TRIGGER IF NOT EXISTS ocr_text_ad AFTER DELETE ON ocr_text BEGIN
    INSERT INTO ocr_text_fts (ocr_text_fts, rowid, text_folded) VALUES ('delete', old.asset_id, old.text_folded);
END;

CREATE TRIGGER IF NOT EXISTS ocr_text_au AFTER UPDATE ON ocr_text BEGIN
    INSERT INTO ocr_text_fts (ocr_text_fts, rowid, text_folded) VALUES ('delete', old.asset_id, old.text_folded);
    INSERT INTO ocr_text_fts (rowid, text_folded) VALUES (new.asset_id, new.text_folded);
END;

PRAGMA user_version = 15;
