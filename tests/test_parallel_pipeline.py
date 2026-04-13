"""Tests for the parallel scoring pipeline (run_scorers_parallel)."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest
from PIL import Image

from takeout_rater.db.queries import (
    bulk_insert_asset_scores,
    finish_scorer_run,
    get_asset_scores,
    insert_scorer_run,
    list_asset_ids_without_score,
    upsert_asset,
)
from takeout_rater.db.schema import migrate
from takeout_rater.scorers.heuristics.blur import BlurScorer
from takeout_rater.scorers.heuristics.luminosity import LuminosityScorer
from takeout_rater.scoring.pipeline import run_scorers_parallel

# ── helpers ──────────────────────────────────────────────────────────────────


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


def _make_thumb(thumbs_dir: Path, asset_id: int, size: int = 32) -> Path:
    """Create a minimal JPEG thumbnail for *asset_id* in *thumbs_dir*."""
    from takeout_rater.indexing.thumbnailer import thumb_path_for_id  # noqa: PLC0415

    p = thumb_path_for_id(thumbs_dir, asset_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (size, size), color=(asset_id * 17 % 256, 128, 64))  # vary by id
    img.save(p, "JPEG")
    return p


# ── run_scorers_parallel ──────────────────────────────────────────────────────


def test_parallel_empty_scorers_returns_empty(tmp_path: Path) -> None:
    """With an empty scorer list the function returns immediately."""
    conn = _open_in_memory()
    result = run_scorers_parallel(conn, [], tmp_path)
    assert result == {}


def test_parallel_no_assets_creates_and_finishes_runs(tmp_path: Path) -> None:
    """When the DB is empty every scorer run is created and immediately finished."""
    conn = _open_in_memory()
    scorer = BlurScorer.create()
    spec = scorer.spec()

    result = run_scorers_parallel(conn, [scorer], tmp_path)

    assert (spec.scorer_id, scorer.variant_id) in result
    run_id = result[(spec.scorer_id, scorer.variant_id)]
    row = conn.execute("SELECT finished_at FROM scorer_runs WHERE id = ?", (run_id,)).fetchone()
    assert row is not None
    assert row["finished_at"] is not None


def test_parallel_scores_single_asset(tmp_path: Path) -> None:
    """A single asset is scored by a single scorer."""
    conn = _open_in_memory()
    asset_id = _add_asset(conn)
    _make_thumb(tmp_path, asset_id)

    scorer = BlurScorer.create()
    spec = scorer.spec()

    result = run_scorers_parallel(conn, [scorer], tmp_path)

    assert (spec.scorer_id, scorer.variant_id) in result
    scores = get_asset_scores(conn, asset_id)
    keys = {s["metric_key"] for s in scores}
    assert "sharpness" in keys


def test_parallel_scores_multiple_assets(tmp_path: Path) -> None:
    """All assets receive scores from all scorers."""
    conn = _open_in_memory()
    n = 5
    ids = [_add_asset(conn, f"p/{i}.jpg") for i in range(n)]
    for aid in ids:
        _make_thumb(tmp_path, aid)

    scorers = [BlurScorer.create(), LuminosityScorer.create()]
    run_scorers_parallel(conn, scorers, tmp_path)

    for aid in ids:
        scores = get_asset_scores(conn, aid)
        keys = {s["metric_key"] for s in scores}
        assert "sharpness" in keys
        assert "brightness" in keys


def test_parallel_skips_existing_scores(tmp_path: Path) -> None:
    """skip_existing=True avoids re-scoring already scored assets."""
    conn = _open_in_memory()
    asset_id = _add_asset(conn)
    _make_thumb(tmp_path, asset_id)

    # Pre-score with a fixed value.
    run_id = insert_scorer_run(conn, "blur", "default")
    bulk_insert_asset_scores(conn, run_id, [(asset_id, "sharpness", 99.0)])
    finish_scorer_run(conn, run_id)

    # Run parallel scorer with skip_existing=True — should create a new run
    # but score nothing (asset already has sharpness).
    scorer = BlurScorer.create()
    run_scorers_parallel(conn, [scorer], tmp_path, skip_existing=True)

    # The original pre-scored value (99.0) should still be the only score in
    # asset_scores; a new run was opened but nothing was inserted into it.
    scores = get_asset_scores(conn, asset_id)
    sharpness_scores = [s for s in scores if s["metric_key"] == "sharpness"]
    assert len(sharpness_scores) == 1
    assert sharpness_scores[0]["value"] == pytest.approx(99.0)


def test_parallel_rescores_when_skip_false(tmp_path: Path) -> None:
    """skip_existing=False causes all assets to be re-scored."""
    conn = _open_in_memory()
    asset_id = _add_asset(conn)
    _make_thumb(tmp_path, asset_id)

    # Pre-score with a sentinel value that the real scorer would not produce.
    run_id = insert_scorer_run(conn, "blur", "default")
    bulk_insert_asset_scores(conn, run_id, [(asset_id, "sharpness", 99.0)])
    finish_scorer_run(conn, run_id)

    scorer = BlurScorer.create()
    run_scorers_parallel(conn, [scorer], tmp_path, skip_existing=False)

    # With skip_existing=False the new run should have written a fresh score.
    # Because bulk_insert_asset_scores replaces old scores from earlier runs,
    # there should be exactly one sharpness score for the asset, from the
    # latest run.
    latest_run_id = max(
        row[0]
        for row in conn.execute("SELECT id FROM scorer_runs WHERE scorer_id = 'blur'").fetchall()
    )
    rows = conn.execute(
        "SELECT value FROM asset_scores WHERE asset_id = ? AND scorer_run_id = ?",
        (asset_id, latest_run_id),
    ).fetchall()
    assert len(rows) == 1


def test_parallel_missing_thumbnail_is_skipped(tmp_path: Path) -> None:
    """Assets whose thumbnails do not exist are silently skipped."""
    conn = _open_in_memory()
    asset_id = _add_asset(conn)
    # Intentionally do NOT create a thumbnail.

    scorer = BlurScorer.create()
    run_scorers_parallel(conn, [scorer], tmp_path)

    # No score should have been written.
    assert get_asset_scores(conn, asset_id) == []


def test_parallel_progress_callback_called(tmp_path: Path) -> None:
    """on_progress is called once per processed asset."""
    conn = _open_in_memory()
    n = 3
    ids = [_add_asset(conn, f"p/{i}.jpg") for i in range(n)]
    for aid in ids:
        _make_thumb(tmp_path, aid)

    calls: list[tuple[int, int]] = []

    def _cb(done: int, total: int) -> None:
        calls.append((done, total))

    run_scorers_parallel(conn, [BlurScorer.create()], tmp_path, on_progress=_cb)

    assert len(calls) == n
    assert calls[-1][0] == n
    assert calls[-1][1] == n


def test_parallel_cancel_stops_early(tmp_path: Path) -> None:
    """Setting cancel_check=True prevents any assets from being processed."""
    conn = _open_in_memory()
    n = 5
    ids = [_add_asset(conn, f"p/{i}.jpg") for i in range(n)]
    for aid in ids:
        _make_thumb(tmp_path, aid)

    run_scorers_parallel(conn, [BlurScorer.create()], tmp_path, cancel_check=lambda: True)

    # No scores should have been written (cancel fires before any task is submitted).
    for aid in ids:
        assert get_asset_scores(conn, aid) == []


def test_parallel_run_ids_returned(tmp_path: Path) -> None:
    """Return dict maps (scorer_id, variant_id) to valid run IDs."""
    conn = _open_in_memory()
    asset_id = _add_asset(conn)
    _make_thumb(tmp_path, asset_id)

    blur = BlurScorer.create()
    lum = LuminosityScorer.create()
    result = run_scorers_parallel(conn, [blur, lum], tmp_path)

    assert ("blur", "default") in result
    assert ("luminosity", "default") in result
    for run_id in result.values():
        assert isinstance(run_id, int)
        assert run_id > 0


def test_parallel_scorer_run_finished_at_set(tmp_path: Path) -> None:
    """All scorer_runs records have finished_at set after the function returns."""
    conn = _open_in_memory()
    asset_id = _add_asset(conn)
    _make_thumb(tmp_path, asset_id)

    result = run_scorers_parallel(conn, [BlurScorer.create()], tmp_path)

    for run_id in result.values():
        row = conn.execute("SELECT finished_at FROM scorer_runs WHERE id = ?", (run_id,)).fetchone()
        assert row is not None
        assert row["finished_at"] is not None


def test_parallel_two_scorers_only_needed_assets(tmp_path: Path) -> None:
    """When one scorer already scored all assets, it skips but the other still runs."""
    conn = _open_in_memory()
    asset_id = _add_asset(conn)
    _make_thumb(tmp_path, asset_id)

    # Pre-score blur.
    run_id = insert_scorer_run(conn, "blur", "default")
    bulk_insert_asset_scores(conn, run_id, [(asset_id, "sharpness", 50.0)])
    finish_scorer_run(conn, run_id)

    # Run both scorers; blur should skip, luminosity should score.
    run_scorers_parallel(
        conn, [BlurScorer.create(), LuminosityScorer.create()], tmp_path, skip_existing=True
    )

    scores = get_asset_scores(conn, asset_id)
    keys = {s["metric_key"] for s in scores}
    assert "brightness" in keys  # newly scored
    # sharpness still present (unchanged from pre-score)
    assert "sharpness" in keys


def test_parallel_no_missing_assets_after_run(tmp_path: Path) -> None:
    """After a full run, list_asset_ids_without_score returns empty for scored metrics."""
    conn = _open_in_memory()
    n = 4
    ids = [_add_asset(conn, f"p/{i}.jpg") for i in range(n)]
    for aid in ids:
        _make_thumb(tmp_path, aid)

    scorer = BlurScorer.create()
    run_scorers_parallel(conn, [scorer], tmp_path)

    missing = list_asset_ids_without_score(conn, "blur", "default", "sharpness")
    assert missing == []
