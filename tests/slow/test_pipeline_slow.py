"""Slow end-to-end pipeline test.

Excluded from the default test run (--ignore=tests/slow) because it calls the
index CLI which hangs in environments without a display or GPU for thumbnail
generation.
"""

from __future__ import annotations

from pathlib import Path

from takeout_rater.scoring.pipeline import run_scorer
from takeout_rater.scoring.scorers.simple import SimpleScorer

FIXTURE_TAKEOUT = Path(__file__).parent.parent / "fixtures" / "takeout_tree" / "Takeout"


# ── end-to-end: index then score ──────────────────────────────────────────────


def test_index_then_score_pipeline(tmp_path: Path) -> None:
    """E2E: index a real Takeout folder then score all assets with SimpleScorer (blur variant).

    This test exercises the full pipeline:
      1. Run the index job to populate the DB and generate thumbnails.
      2. Open the library DB written by the indexer.
      3. Run the score job over the indexed assets.
      4. Assert that every asset whose thumbnail was generated has a score.
    """
    from takeout_rater.cli import main  # noqa: PLC0415
    from takeout_rater.db.connection import open_library_db  # noqa: PLC0415

    # Link fixture photos tree into the temp photos root.
    photos_root = tmp_path / "photos"
    photos_root.symlink_to(FIXTURE_TAKEOUT.resolve(), target_is_directory=True)

    # Step 1: index (scans files, populates DB, generates thumbnails).
    rc = main(["index", str(photos_root), "--db-root", str(tmp_path)])
    assert rc == 0

    # Step 2: open the DB the indexer just created.
    conn = open_library_db(tmp_path)
    thumbs_dir = tmp_path / "takeout-rater" / "thumbs"

    # Step 3: score all assets.
    scorer = SimpleScorer.create(variant_id="blur")
    run_scorer(conn, scorer, thumbs_dir)

    # Step 4: every asset that has a thumbnail must have a score row.
    thumbs = list(thumbs_dir.rglob("*.jpg"))
    assert len(thumbs) >= 1, "Indexer should have generated at least one thumbnail"

    scored = conn.execute("SELECT COUNT(*) FROM asset_scores").fetchone()[0]
    assert scored == len(thumbs), (
        f"Expected {len(thumbs)} score rows (one per thumbnail), got {scored}"
    )
