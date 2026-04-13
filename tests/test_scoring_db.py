"""Tests for scoring-related DB query helpers."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from takeout_rater.db.queries import (
    bulk_insert_asset_scores,
    count_assets_with_score,
    finish_scorer_run,
    get_asset_scores,
    get_latest_scorer_run_id,
    get_phash,
    insert_scorer_run,
    list_asset_ids_without_phash,
    list_asset_ids_without_score,
    list_assets_by_score,
    upsert_asset,
    upsert_phash,
)
from takeout_rater.db.schema import migrate

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


# ── insert_scorer_run ─────────────────────────────────────────────────────────


def test_insert_scorer_run_returns_int() -> None:
    conn = _open_in_memory()
    run_id = insert_scorer_run(conn, "blur", "default")
    assert isinstance(run_id, int)
    assert run_id > 0


def test_insert_scorer_run_sets_started_at() -> None:
    conn = _open_in_memory()
    before = int(time.time())
    run_id = insert_scorer_run(conn, "blur", "default")
    after = int(time.time())
    row = conn.execute(
        "SELECT started_at, finished_at FROM scorer_runs WHERE id = ?", (run_id,)
    ).fetchone()
    assert row is not None
    assert before <= row["started_at"] <= after
    assert row["finished_at"] is None


def test_insert_scorer_run_with_version() -> None:
    conn = _open_in_memory()
    run_id = insert_scorer_run(conn, "blur", "default", scorer_version="1.0.0")
    row = conn.execute("SELECT scorer_version FROM scorer_runs WHERE id = ?", (run_id,)).fetchone()
    assert row["scorer_version"] == "1.0.0"


# ── finish_scorer_run ─────────────────────────────────────────────────────────


def test_finish_scorer_run_sets_finished_at() -> None:
    conn = _open_in_memory()
    run_id = insert_scorer_run(conn, "blur", "default")
    before = int(time.time())
    finish_scorer_run(conn, run_id)
    after = int(time.time())
    row = conn.execute("SELECT finished_at FROM scorer_runs WHERE id = ?", (run_id,)).fetchone()
    assert row is not None
    assert before <= row["finished_at"] <= after


# ── bulk_insert_asset_scores ──────────────────────────────────────────────────


def test_bulk_insert_asset_scores_writes_rows() -> None:
    conn = _open_in_memory()
    asset_id = _add_asset(conn)
    run_id = insert_scorer_run(conn, "blur", "default")
    bulk_insert_asset_scores(conn, run_id, [(asset_id, "sharpness", 42.5)])
    row = conn.execute(
        "SELECT value FROM asset_scores WHERE asset_id = ? AND scorer_run_id = ?",
        (asset_id, run_id),
    ).fetchone()
    assert row is not None
    assert row["value"] == pytest.approx(42.5)


def test_bulk_insert_asset_scores_ignores_duplicates() -> None:
    conn = _open_in_memory()
    asset_id = _add_asset(conn)
    run_id = insert_scorer_run(conn, "blur", "default")
    bulk_insert_asset_scores(conn, run_id, [(asset_id, "sharpness", 10.0)])
    # Inserting the same row again must not raise
    bulk_insert_asset_scores(conn, run_id, [(asset_id, "sharpness", 20.0)])
    count = conn.execute("SELECT COUNT(*) FROM asset_scores").fetchone()[0]
    assert count == 1  # first value preserved


def test_bulk_insert_asset_scores_new_run_overwrites_old() -> None:
    """Scores from a new run replace those from a previous run of the same scorer."""
    conn = _open_in_memory()
    asset_id = _add_asset(conn)
    run1 = insert_scorer_run(conn, "blur", "default")
    bulk_insert_asset_scores(conn, run1, [(asset_id, "sharpness", 10.0)])
    finish_scorer_run(conn, run1)

    run2 = insert_scorer_run(conn, "blur", "default")
    bulk_insert_asset_scores(conn, run2, [(asset_id, "sharpness", 99.0)])
    finish_scorer_run(conn, run2)

    # Only the run2 score should survive in asset_scores
    total = conn.execute("SELECT COUNT(*) FROM asset_scores").fetchone()[0]
    assert total == 1
    row = conn.execute("SELECT value FROM asset_scores WHERE scorer_run_id = ?", (run2,)).fetchone()
    assert row is not None
    assert row["value"] == pytest.approx(99.0)


def test_get_asset_scores_no_duplicates_across_runs() -> None:
    """get_asset_scores returns one score per metric even across multiple runs."""
    conn = _open_in_memory()
    asset_id = _add_asset(conn)

    run1 = insert_scorer_run(conn, "blur", "default")
    bulk_insert_asset_scores(conn, run1, [(asset_id, "sharpness", 10.0)])
    finish_scorer_run(conn, run1)

    run2 = insert_scorer_run(conn, "blur", "default")
    bulk_insert_asset_scores(conn, run2, [(asset_id, "sharpness", 99.0)])
    finish_scorer_run(conn, run2)

    scores = get_asset_scores(conn, asset_id)
    assert len(scores) == 1
    assert scores[0]["value"] == pytest.approx(99.0)


def test_bulk_insert_asset_scores_multiple() -> None:
    conn = _open_in_memory()
    ids = [_add_asset(conn, f"p/{i}.jpg") for i in range(3)]
    run_id = insert_scorer_run(conn, "blur", "default")
    scores = [(aid, "sharpness", float(i * 10)) for i, aid in enumerate(ids)]
    bulk_insert_asset_scores(conn, run_id, scores)
    count = conn.execute("SELECT COUNT(*) FROM asset_scores").fetchone()[0]
    assert count == 3


# ── get_asset_scores ──────────────────────────────────────────────────────────


def test_get_asset_scores_empty() -> None:
    conn = _open_in_memory()
    asset_id = _add_asset(conn)
    assert get_asset_scores(conn, asset_id) == []


def test_get_asset_scores_returns_finished_runs_only() -> None:
    conn = _open_in_memory()
    asset_id = _add_asset(conn)
    # Unfinished run — should NOT appear
    run_open = insert_scorer_run(conn, "blur", "default")
    bulk_insert_asset_scores(conn, run_open, [(asset_id, "sharpness", 1.0)])
    # Finished run — SHOULD appear
    run_done = insert_scorer_run(conn, "blur", "default")
    bulk_insert_asset_scores(conn, run_done, [(asset_id, "sharpness", 55.0)])
    finish_scorer_run(conn, run_done)

    scores = get_asset_scores(conn, asset_id)
    assert len(scores) == 1
    assert scores[0]["metric_key"] == "sharpness"
    assert scores[0]["value"] == pytest.approx(55.0)
    assert scores[0]["scorer_id"] == "blur"


def test_get_asset_scores_multiple_metrics() -> None:
    conn = _open_in_memory()
    asset_id = _add_asset(conn)
    run_id = insert_scorer_run(conn, "multi", "default")
    bulk_insert_asset_scores(conn, run_id, [(asset_id, "alpha", 1.0), (asset_id, "beta", 2.0)])
    finish_scorer_run(conn, run_id)
    scores = get_asset_scores(conn, asset_id)
    keys = {s["metric_key"] for s in scores}
    assert keys == {"alpha", "beta"}


# ── get_latest_scorer_run_id ──────────────────────────────────────────────────


def test_get_latest_scorer_run_id_none_when_no_runs() -> None:
    conn = _open_in_memory()
    assert get_latest_scorer_run_id(conn, "blur", "default") is None


def test_get_latest_scorer_run_id_none_when_not_finished() -> None:
    conn = _open_in_memory()
    insert_scorer_run(conn, "blur", "default")
    assert get_latest_scorer_run_id(conn, "blur", "default") is None


def test_get_latest_scorer_run_id_returns_most_recent() -> None:
    conn = _open_in_memory()
    run1 = insert_scorer_run(conn, "blur", "default")
    finish_scorer_run(conn, run1)
    run2 = insert_scorer_run(conn, "blur", "default")
    finish_scorer_run(conn, run2)
    assert get_latest_scorer_run_id(conn, "blur", "default") == run2


# ── list_assets_by_score ──────────────────────────────────────────────────────


def test_list_assets_by_score_empty_when_no_runs() -> None:
    conn = _open_in_memory()
    _add_asset(conn)
    result = list_assets_by_score(conn, "blur", "sharpness")
    assert result == []


def test_list_assets_by_score_ordered_descending() -> None:
    conn = _open_in_memory()
    ids = [_add_asset(conn, f"p/{i}.jpg") for i in range(3)]
    run_id = insert_scorer_run(conn, "blur", "default")
    bulk_insert_asset_scores(
        conn,
        run_id,
        [(ids[0], "sharpness", 10.0), (ids[1], "sharpness", 80.0), (ids[2], "sharpness", 50.0)],
    )
    finish_scorer_run(conn, run_id)

    pairs = list_assets_by_score(conn, "blur", "sharpness")
    scores = [s for _, s in pairs]
    assert scores == sorted(scores, reverse=True)


def test_list_assets_by_score_ordered_ascending() -> None:
    conn = _open_in_memory()
    ids = [_add_asset(conn, f"p/{i}.jpg") for i in range(3)]
    run_id = insert_scorer_run(conn, "blur", "default")
    bulk_insert_asset_scores(
        conn,
        run_id,
        [(ids[0], "sharpness", 10.0), (ids[1], "sharpness", 80.0), (ids[2], "sharpness", 50.0)],
    )
    finish_scorer_run(conn, run_id)

    pairs = list_assets_by_score(conn, "blur", "sharpness", descending=False)
    scores = [s for _, s in pairs]
    assert scores == sorted(scores)


def test_list_assets_by_score_pagination() -> None:
    conn = _open_in_memory()
    ids = [_add_asset(conn, f"p/{i}.jpg") for i in range(5)]
    run_id = insert_scorer_run(conn, "blur", "default")
    bulk_insert_asset_scores(
        conn, run_id, [(aid, "sharpness", float(i)) for i, aid in enumerate(ids)]
    )
    finish_scorer_run(conn, run_id)

    page1 = list_assets_by_score(conn, "blur", "sharpness", limit=3, offset=0)
    page2 = list_assets_by_score(conn, "blur", "sharpness", limit=3, offset=3)
    assert len(page1) == 3
    assert len(page2) == 2


# ── count_assets_with_score ───────────────────────────────────────────────────


def test_count_assets_with_score_zero_no_runs() -> None:
    conn = _open_in_memory()
    _add_asset(conn)
    assert count_assets_with_score(conn, "blur", "sharpness") == 0


def test_count_assets_with_score_correct() -> None:
    conn = _open_in_memory()
    ids = [_add_asset(conn, f"p/{i}.jpg") for i in range(4)]
    run_id = insert_scorer_run(conn, "blur", "default")
    bulk_insert_asset_scores(
        conn, run_id, [(aid, "sharpness", float(i)) for i, aid in enumerate(ids)]
    )
    finish_scorer_run(conn, run_id)
    assert count_assets_with_score(conn, "blur", "sharpness") == 4


def test_count_assets_with_score_min_filter() -> None:
    conn = _open_in_memory()
    ids = [_add_asset(conn, f"p/{i}.jpg") for i in range(5)]
    run_id = insert_scorer_run(conn, "blur", "default")
    bulk_insert_asset_scores(
        conn, run_id, [(aid, "sharpness", float(i * 10)) for i, aid in enumerate(ids)]
    )
    finish_scorer_run(conn, run_id)
    # scores: 0, 10, 20, 30, 40 — only those >= 20 pass
    assert count_assets_with_score(conn, "blur", "sharpness", min_score=20.0) == 3


def test_count_assets_with_score_max_filter() -> None:
    conn = _open_in_memory()
    ids = [_add_asset(conn, f"p/{i}.jpg") for i in range(5)]
    run_id = insert_scorer_run(conn, "blur", "default")
    bulk_insert_asset_scores(
        conn, run_id, [(aid, "sharpness", float(i * 10)) for i, aid in enumerate(ids)]
    )
    finish_scorer_run(conn, run_id)
    # scores: 0, 10, 20, 30, 40 — only those <= 20 pass
    assert count_assets_with_score(conn, "blur", "sharpness", max_score=20.0) == 3


def test_count_assets_with_score_min_max_range() -> None:
    conn = _open_in_memory()
    ids = [_add_asset(conn, f"p/{i}.jpg") for i in range(5)]
    run_id = insert_scorer_run(conn, "blur", "default")
    bulk_insert_asset_scores(
        conn, run_id, [(aid, "sharpness", float(i * 10)) for i, aid in enumerate(ids)]
    )
    finish_scorer_run(conn, run_id)
    # scores: 0, 10, 20, 30, 40 — only 10, 20, 30 fall in [10, 30]
    assert count_assets_with_score(conn, "blur", "sharpness", min_score=10.0, max_score=30.0) == 3


# ── list_assets_by_score min/max ──────────────────────────────────────────────


def test_list_assets_by_score_min_filter() -> None:
    conn = _open_in_memory()
    ids = [_add_asset(conn, f"p/{i}.jpg") for i in range(4)]
    run_id = insert_scorer_run(conn, "blur", "default")
    bulk_insert_asset_scores(
        conn, run_id, [(aid, "sharpness", float(i * 20)) for i, aid in enumerate(ids)]
    )
    finish_scorer_run(conn, run_id)
    # scores: 0, 20, 40, 60 — min_score=40 keeps 40 and 60
    pairs = list_assets_by_score(conn, "blur", "sharpness", min_score=40.0)
    scores = [s for _, s in pairs]
    assert all(s >= 40.0 for s in scores)
    assert len(scores) == 2


def test_list_assets_by_score_max_filter() -> None:
    conn = _open_in_memory()
    ids = [_add_asset(conn, f"p/{i}.jpg") for i in range(4)]
    run_id = insert_scorer_run(conn, "blur", "default")
    bulk_insert_asset_scores(
        conn, run_id, [(aid, "sharpness", float(i * 20)) for i, aid in enumerate(ids)]
    )
    finish_scorer_run(conn, run_id)
    # scores: 0, 20, 40, 60 — max_score=20 keeps 0 and 20
    pairs = list_assets_by_score(conn, "blur", "sharpness", max_score=20.0)
    scores = [s for _, s in pairs]
    assert all(s <= 20.0 for s in scores)
    assert len(scores) == 2


def test_list_assets_by_score_range_filter() -> None:
    conn = _open_in_memory()
    ids = [_add_asset(conn, f"p/{i}.jpg") for i in range(5)]
    run_id = insert_scorer_run(conn, "blur", "default")
    bulk_insert_asset_scores(
        conn, run_id, [(aid, "sharpness", float(i * 10)) for i, aid in enumerate(ids)]
    )
    finish_scorer_run(conn, run_id)
    # scores: 0, 10, 20, 30, 40 — range [10, 30] keeps 10, 20, 30
    pairs = list_assets_by_score(conn, "blur", "sharpness", min_score=10.0, max_score=30.0)
    scores = [s for _, s in pairs]
    assert len(scores) == 3
    assert all(10.0 <= s <= 30.0 for s in scores)


# ── upsert_phash / get_phash ──────────────────────────────────────────────────


def test_upsert_phash_stores_hash() -> None:
    conn = _open_in_memory()
    asset_id = _add_asset(conn)
    upsert_phash(conn, asset_id, "deadbeef12345678")
    result = get_phash(conn, asset_id)
    assert result is not None
    assert result["phash_hex"] == "deadbeef12345678"
    assert result["algo"] == "dhash16"


def test_get_phash_missing_returns_none() -> None:
    conn = _open_in_memory()
    assert get_phash(conn, 99999) is None


def test_upsert_phash_is_idempotent() -> None:
    conn = _open_in_memory()
    asset_id = _add_asset(conn)
    upsert_phash(conn, asset_id, "aaaa")
    upsert_phash(conn, asset_id, "bbbb")  # update
    result = get_phash(conn, asset_id)
    assert result is not None
    assert result["phash_hex"] == "bbbb"


# ── list_asset_ids_without_phash ──────────────────────────────────────────────


def test_list_asset_ids_without_phash_all_missing() -> None:
    conn = _open_in_memory()
    ids = [_add_asset(conn, f"p/{i}.jpg") for i in range(3)]
    missing = list_asset_ids_without_phash(conn)
    assert set(missing) == set(ids)


def test_list_asset_ids_without_phash_partial() -> None:
    conn = _open_in_memory()
    ids = [_add_asset(conn, f"p/{i}.jpg") for i in range(3)]
    upsert_phash(conn, ids[1], "ff")
    missing = list_asset_ids_without_phash(conn)
    assert ids[1] not in missing
    assert ids[0] in missing
    assert ids[2] in missing


# ── list_asset_ids_without_score ──────────────────────────────────────────────


def test_list_asset_ids_without_score_all_missing() -> None:
    conn = _open_in_memory()
    ids = [_add_asset(conn, f"p/{i}.jpg") for i in range(3)]
    missing = list_asset_ids_without_score(conn, "blur", "default", "sharpness")
    assert set(missing) == set(ids)


def test_list_asset_ids_without_score_after_scoring() -> None:
    conn = _open_in_memory()
    ids = [_add_asset(conn, f"p/{i}.jpg") for i in range(3)]
    run_id = insert_scorer_run(conn, "blur", "default")
    bulk_insert_asset_scores(conn, run_id, [(ids[0], "sharpness", 1.0)])
    finish_scorer_run(conn, run_id)
    missing = list_asset_ids_without_score(conn, "blur", "default", "sharpness")
    assert ids[0] not in missing
    assert ids[1] in missing
    assert ids[2] in missing
