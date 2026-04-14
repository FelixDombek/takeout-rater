"""Tests for the scoring pipeline."""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from takeout_rater.db.queries import (
    get_asset_scores,
    list_assets_by_score,
    upsert_asset,
)
from takeout_rater.db.schema import migrate
from takeout_rater.scorers.heuristics.simple import SimpleScorer
from takeout_rater.scoring.pipeline import run_scorer, run_scorer_by_id

FIXTURE_TAKEOUT = Path(__file__).parent / "fixtures" / "takeout_tree" / "Takeout"

# ── Helpers ───────────────────────────────────────────────────────────────────


def _open_in_memory() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    return conn


def _add_asset(conn: sqlite3.Connection, relpath: str = "Photos/img.jpg") -> int:
    return upsert_asset(
        conn,
        {
            "relpath": relpath,
            "filename": Path(relpath).name,
            "ext": Path(relpath).suffix.lower(),
            "size_bytes": 512,
            "mime": "image/jpeg",
            "indexed_at": int(time.time()),
        },
    )


def _make_thumbnail(thumbs_dir: Path, asset_id: int) -> Path:
    """Create a minimal JPEG thumbnail file and return its path."""
    from PIL import Image  # noqa: PLC0415

    from takeout_rater.indexing.thumbnailer import thumb_path_for_id  # noqa: PLC0415

    thumb = thumb_path_for_id(thumbs_dir, asset_id)
    thumb.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(100, 150, 200)).save(thumb, "JPEG")
    return thumb


# ── run_scorer ────────────────────────────────────────────────────────────────


def test_run_scorer_returns_int(tmp_path: Path) -> None:
    conn = _open_in_memory()
    _add_asset(conn)
    scorer = SimpleScorer.create(variant_id="blur")
    num_scored = run_scorer(conn, scorer, tmp_path / "thumbs")
    assert isinstance(num_scored, int)
    assert num_scored >= 0


def test_run_scorer_writes_scores_when_thumb_present(tmp_path: Path) -> None:
    conn = _open_in_memory()
    thumbs_dir = tmp_path / "thumbs"
    asset_id = _add_asset(conn)
    _make_thumbnail(thumbs_dir, asset_id)

    scorer = SimpleScorer.create(variant_id="blur")
    run_scorer(conn, scorer, thumbs_dir)

    scores = get_asset_scores(conn, asset_id)
    assert len(scores) == 1
    assert scores[0]["metric_key"] == "sharpness"
    assert 0.0 <= scores[0]["value"] <= 100.0


def test_run_scorer_skips_missing_thumbnails(tmp_path: Path) -> None:
    conn = _open_in_memory()
    asset_id = _add_asset(conn)
    # No thumbnail created

    scorer = SimpleScorer.create(variant_id="blur")
    run_scorer(conn, scorer, tmp_path / "thumbs")

    scores = get_asset_scores(conn, asset_id)
    assert scores == []


def test_run_scorer_skip_existing_true(tmp_path: Path) -> None:
    """Second run with skip_existing=True should not re-score already-scored assets."""
    conn = _open_in_memory()
    thumbs_dir = tmp_path / "thumbs"
    asset_id = _add_asset(conn)
    _make_thumbnail(thumbs_dir, asset_id)

    scorer = SimpleScorer.create(variant_id="blur")
    num1 = run_scorer(conn, scorer, thumbs_dir, skip_existing=True)
    num2 = run_scorer(conn, scorer, thumbs_dir, skip_existing=True)

    # Second run should score 0 new assets (all already scored)
    assert num1 == 1
    assert num2 == 0

    # Only one score row should exist
    count = conn.execute("SELECT COUNT(*) FROM asset_scores").fetchone()[0]
    assert count == 1


def test_skip_existing_does_not_hide_scores_from_browse(tmp_path: Path) -> None:
    """Regression: re-running with skip_existing=True must not hide existing scores.

    This was the original bug: the old scorer_runs model created an empty
    "latest finished run" that shadowed real scores.  With the new flat model,
    scores are always directly visible.
    """
    conn = _open_in_memory()
    thumbs_dir = tmp_path / "thumbs"
    asset_id = _add_asset(conn)
    _make_thumbnail(thumbs_dir, asset_id)

    scorer = SimpleScorer.create(variant_id="blur")
    spec = scorer.spec()
    scorer_id = spec.scorer_id
    metric_key = spec.metrics[0].key
    run_scorer(conn, scorer, thumbs_dir, skip_existing=True)
    # Second run — simulates clicking "Score" again from the UI
    run_scorer(conn, scorer, thumbs_dir, skip_existing=True)

    results = list_assets_by_score(conn, scorer_id, metric_key, variant_id="blur")
    assert len(results) == 1, "Scores should still be visible after a second skip_existing=True run"


def test_run_scorer_rerun(tmp_path: Path) -> None:
    """skip_existing=False should re-score even existing assets."""
    conn = _open_in_memory()
    thumbs_dir = tmp_path / "thumbs"
    asset_id = _add_asset(conn)
    _make_thumbnail(thumbs_dir, asset_id)

    scorer = SimpleScorer.create(variant_id="blur")
    run_scorer(conn, scorer, thumbs_dir, skip_existing=False)
    num2 = run_scorer(conn, scorer, thumbs_dir, skip_existing=False)

    # Second run should have re-scored the asset
    assert num2 == 1
    count = conn.execute("SELECT COUNT(*) FROM asset_scores").fetchone()[0]
    assert count == 1  # overwritten, not duplicated


def test_run_scorer_progress_callback(tmp_path: Path) -> None:
    conn = _open_in_memory()
    thumbs_dir = tmp_path / "thumbs"
    for i in range(3):
        aid = _add_asset(conn, f"p/{i}.jpg")
        _make_thumbnail(thumbs_dir, aid)

    calls: list[tuple[int, int]] = []
    scorer = SimpleScorer.create(variant_id="blur")
    run_scorer(
        conn, scorer, thumbs_dir, batch_size=2, on_progress=lambda d, t: calls.append((d, t))
    )

    assert len(calls) > 0
    # Last call should report all processed
    assert calls[-1][0] == calls[-1][1]


def test_run_scorer_explicit_asset_ids(tmp_path: Path) -> None:
    conn = _open_in_memory()
    thumbs_dir = tmp_path / "thumbs"
    ids = [_add_asset(conn, f"p/{i}.jpg") for i in range(3)]
    for aid in ids:
        _make_thumbnail(thumbs_dir, aid)

    scorer = SimpleScorer.create(variant_id="blur")
    # Only score the first asset
    run_scorer(conn, scorer, thumbs_dir, asset_ids=[ids[0]])

    count = conn.execute("SELECT COUNT(*) FROM asset_scores").fetchone()[0]
    assert count == 1


# ── run_scorer_by_id ──────────────────────────────────────────────────────────


def test_run_scorer_by_id_unknown_raises_key_error(tmp_path: Path) -> None:
    conn = _open_in_memory()
    with pytest.raises(KeyError):
        run_scorer_by_id(conn, "nonexistent_scorer", tmp_path / "thumbs")


def test_run_scorer_by_id_simple_blur_variant(tmp_path: Path) -> None:
    conn = _open_in_memory()
    thumbs_dir = tmp_path / "thumbs"
    asset_id = _add_asset(conn)
    _make_thumbnail(thumbs_dir, asset_id)

    num_scored = run_scorer_by_id(conn, "simple", thumbs_dir, variant_id="blur")
    assert isinstance(num_scored, int)

    scores = get_asset_scores(conn, asset_id)
    assert len(scores) == 1
    assert scores[0]["scorer_id"] == "simple"


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

    # Link fixture Takeout tree into the temp library root.
    (tmp_path / "Takeout").symlink_to(FIXTURE_TAKEOUT.resolve(), target_is_directory=True)

    # Step 1: index (scans files, populates DB, generates thumbnails).
    rc = main(["index", str(tmp_path)])
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


# ── error context ─────────────────────────────────────────────────────────────


def test_run_scorer_error_includes_scorer_id(tmp_path: Path) -> None:
    """When score_batch raises, the re-raised RuntimeError includes the scorer id."""
    conn = _open_in_memory()
    thumbs_dir = tmp_path / "thumbs"
    asset_id = _add_asset(conn)
    _make_thumbnail(thumbs_dir, asset_id)

    scorer = SimpleScorer.create(variant_id="blur")
    with (
        patch.object(scorer, "score_batch", side_effect=RuntimeError("inner boom")),
        pytest.raises(RuntimeError, match="simple"),
    ):
        run_scorer(conn, scorer, thumbs_dir)


def test_run_scorer_error_includes_asset_path(tmp_path: Path) -> None:
    """When score_batch raises, the error message includes the affected asset path."""
    conn = _open_in_memory()
    thumbs_dir = tmp_path / "thumbs"
    asset_id = _add_asset(conn)
    thumb = _make_thumbnail(thumbs_dir, asset_id)

    scorer = SimpleScorer.create(variant_id="blur")
    with (
        patch.object(scorer, "score_batch", side_effect=RuntimeError("inner boom")),
        pytest.raises(RuntimeError, match=str(thumb)),
    ):
        run_scorer(conn, scorer, thumbs_dir)


def test_run_scorer_error_is_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """When score_batch raises, an ERROR log record is emitted with scorer and asset info."""
    conn = _open_in_memory()
    thumbs_dir = tmp_path / "thumbs"
    asset_id = _add_asset(conn)
    _make_thumbnail(thumbs_dir, asset_id)

    scorer = SimpleScorer.create(variant_id="blur")
    with (
        caplog.at_level(logging.ERROR, logger="takeout_rater.scoring.pipeline"),
        patch.object(scorer, "score_batch", side_effect=RuntimeError("boom")),
        pytest.raises(RuntimeError),
    ):
        run_scorer(conn, scorer, thumbs_dir)

    assert any("blur" in r.message and r.levelno == logging.ERROR for r in caplog.records)
