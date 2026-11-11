"""Tests for the kNN semantic outlier detection and its library scorer."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np
import pytest

from takeout_rater.clustering.outliers import knn_mean_similarity, outlier_percentiles
from takeout_rater.db.queries import (
    bulk_upsert_clip_embeddings,
    get_asset_scores,
    list_available_score_metrics_with_variants,
    upsert_asset,
)
from takeout_rater.db.schema import migrate
from takeout_rater.scoring.pipeline import run_scorer, run_scorer_by_id
from takeout_rater.scoring.scorers.registry import list_scorers
from takeout_rater.scoring.scorers.semantic_outlier import SemanticOutlierScorer

# ── Helpers ───────────────────────────────────────────────────────────────────


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    return conn


def _add_asset(conn: sqlite3.Connection, i: int) -> int:
    return upsert_asset(
        conn,
        {
            "relpath": f"Photos/img{i}.jpg",
            "filename": f"img{i}.jpg",
            "ext": ".jpg",
            "size_bytes": 512,
            "mime": "image/jpeg",
            "indexed_at": int(time.time()),
        },
    )


def _clustered(n_per_cluster: int = 8, dim: int = 16, seed: int = 0) -> np.ndarray:
    """Two tight clusters plus one point far from both (the last row)."""
    rng = np.random.default_rng(seed)
    a = np.zeros(dim, dtype=np.float32)
    a[0] = 1.0
    b = np.zeros(dim, dtype=np.float32)
    b[1] = 1.0
    odd = np.zeros(dim, dtype=np.float32)
    odd[2] = 1.0
    rows = [a + 0.05 * rng.standard_normal(dim) for _ in range(n_per_cluster)]
    rows += [b + 0.05 * rng.standard_normal(dim) for _ in range(n_per_cluster)]
    rows.append(odd)
    return np.asarray(rows, dtype=np.float32)


def _store(conn: sqlite3.Connection, matrix: np.ndarray) -> list[int]:
    ids = [_add_asset(conn, i) for i in range(len(matrix))]
    bulk_upsert_clip_embeddings(
        conn, [(aid, row.astype("<f4").tobytes()) for aid, row in zip(ids, matrix, strict=True)]
    )
    return ids


# ── knn_mean_similarity / outlier_percentiles ─────────────────────────────────


@pytest.mark.parametrize("use_torch", [True, False])
def test_knn_flags_isolated_point(use_torch: bool) -> None:
    m = _clustered()
    sims = knn_mean_similarity(m, k=3, block_rows=5, use_torch=use_torch)
    assert sims is not None
    assert int(np.argmin(sims)) == len(m) - 1
    assert sims[:-1].min() > 0.9
    assert sims[-1] < 0.2


def test_knn_torch_and_numpy_agree(monkeypatch: pytest.MonkeyPatch) -> None:
    # Compare float32 against float32: on CUDA the torch path uses float16.
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    m = np.random.default_rng(1).standard_normal((50, 8)).astype(np.float32)
    a = knn_mean_similarity(m, k=4, block_rows=7, use_torch=True)
    b = knn_mean_similarity(m, k=4, block_rows=50, use_torch=False)
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_knn_excludes_self() -> None:
    # Two orthogonal vectors: each one's only neighbour has similarity 0, not 1.
    m = np.eye(2, dtype=np.float32)
    for use_torch in (True, False):
        sims = knn_mean_similarity(m, k=5, use_torch=use_torch)
        np.testing.assert_allclose(sims, [0.0, 0.0], atol=1e-6)


def test_knn_small_inputs() -> None:
    assert knn_mean_similarity(np.empty((0, 4), dtype=np.float32)).shape == (0,)
    assert np.isnan(knn_mean_similarity(np.ones((1, 4), dtype=np.float32))).all()


def test_knn_cancel_returns_none() -> None:
    m = _clustered()
    assert knn_mean_similarity(m, block_rows=2, cancel_check=lambda: True) is None


def test_knn_reports_progress() -> None:
    seen: list[tuple[int, int]] = []
    knn_mean_similarity(
        _clustered(), k=2, block_rows=5, on_progress=lambda d, t: seen.append((d, t))
    )
    assert seen[-1] == (17, 17)
    assert len(seen) == 4


def test_outlier_percentiles() -> None:
    ranks = outlier_percentiles(np.array([0.9, 0.1, 0.5, 0.9], dtype=np.float32))
    assert ranks[1] == pytest.approx(1.0)  # least similar = most unusual
    assert ranks[2] == pytest.approx(2 / 3)
    assert ranks[0] == ranks[3] == pytest.approx(1 / 6)  # tie shares the average rank
    assert outlier_percentiles(np.array([0.3], dtype=np.float32)).tolist() == [0.0]


# ── Scorer + pipeline ─────────────────────────────────────────────────────────


def test_registered_and_available() -> None:
    assert SemanticOutlierScorer in list_scorers(available_only=True)
    spec = SemanticOutlierScorer.spec()
    assert spec.scorer_id == "semantic_outlier"
    assert {v.variant_id for v in spec.variants} == {"clip_l14", "siglip2"}


def test_score_batch_not_supported() -> None:
    with pytest.raises(NotImplementedError):
        SemanticOutlierScorer.create().score_batch([Path("x.jpg")])


def test_pipeline_scores_whole_library(tmp_path: Path) -> None:
    conn = _conn()
    ids = _store(conn, _clustered())
    scorer = SemanticOutlierScorer.create()
    progress: list[tuple[int, int]] = []

    n = run_scorer(conn, scorer, tmp_path, on_progress=lambda d, t: progress.append((d, t)))

    assert n == len(ids)
    assert progress[-1] == (len(ids), len(ids))
    odd = {s["metric_key"]: s["value"] for s in get_asset_scores(conn, ids[-1])}
    assert odd["outlier"] == pytest.approx(1.0)
    normal = {s["metric_key"]: s["value"] for s in get_asset_scores(conn, ids[0])}
    assert normal["outlier"] < 0.95
    assert normal["knn_similarity"] > odd["knn_similarity"]
    triples = set(list_available_score_metrics_with_variants(conn))
    assert ("semantic_outlier", "clip_l14", "outlier") in triples


def test_pipeline_skips_when_up_to_date_and_rescoring_all_after_new_photos(
    tmp_path: Path,
) -> None:
    conn = _conn()
    ids = _store(conn, _clustered())
    assert run_scorer_by_id(conn, "semantic_outlier", tmp_path) == len(ids)
    # Nothing new: skipped.
    assert run_scorer_by_id(conn, "semantic_outlier", tmp_path) == 0
    # One new photo: the whole library is rescored (neighbours changed).
    new_id = _add_asset(conn, 999)
    vec = np.zeros(16, dtype="<f4")
    vec[0] = 1.0
    bulk_upsert_clip_embeddings(conn, [(new_id, vec.tobytes())])
    assert run_scorer_by_id(conn, "semantic_outlier", tmp_path) == len(ids) + 1
    # Forced rerun scores everything again.
    assert run_scorer_by_id(conn, "semantic_outlier", tmp_path, skip_existing=False) == len(ids) + 1


def test_pipeline_without_embeddings_scores_nothing(tmp_path: Path) -> None:
    conn = _conn()
    _add_asset(conn, 1)
    assert run_scorer_by_id(conn, "semantic_outlier", tmp_path) == 0
    assert run_scorer_by_id(conn, "semantic_outlier", tmp_path, variant_id="siglip2") == 0


def test_pipeline_cancel_writes_nothing(tmp_path: Path) -> None:
    conn = _conn()
    ids = _store(conn, _clustered())
    n = run_scorer(conn, SemanticOutlierScorer.create(), tmp_path, cancel_check=lambda: True)
    assert n == 0
    assert get_asset_scores(conn, ids[0]) == []
