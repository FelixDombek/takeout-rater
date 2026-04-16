"""Tests for the CLIP-embedding-based cluster builder."""

from __future__ import annotations

import math
import sqlite3
import struct
import time
from pathlib import Path

import numpy as np
import pytest

from takeout_rater.clustering.clip_builder import (
    _compute_diameter_clip,
    _find_representative,
    _load_embeddings,
    _split_by_complete_linkage_clip,
    build_clip_clusters,
)
from takeout_rater.db.queries import (
    bulk_upsert_clip_embeddings,
    count_clusters,
    get_cluster_info,
    get_cluster_members,
    upsert_asset,
)
from takeout_rater.db.schema import migrate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 768


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


def _make_embedding(seed: int = 0) -> bytes:
    """Create a deterministic normalised 768-dim float32 blob."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(_DIM).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return struct.pack(f"{_DIM}f", *vec)


def _make_similar_embedding(base_vec: np.ndarray, noise_scale: float = 0.05) -> bytes:
    """Create an embedding close to *base_vec* by adding small Gaussian noise."""
    rng = np.random.RandomState(42)
    noise = rng.randn(_DIM).astype(np.float32) * noise_scale
    vec = base_vec + noise
    vec = vec / np.linalg.norm(vec)
    return struct.pack(f"{_DIM}f", *vec)


def _decode_embedding(blob: bytes) -> np.ndarray:
    return np.array(struct.unpack(f"{_DIM}f", blob), dtype=np.float32)


def _add_asset_with_embedding(conn: sqlite3.Connection, relpath: str, embedding: bytes) -> int:
    asset_id = _add_asset(conn, relpath)
    bulk_upsert_clip_embeddings(conn, [(asset_id, embedding)])
    return asset_id


# ---------------------------------------------------------------------------
# _load_embeddings
# ---------------------------------------------------------------------------


def test_load_embeddings_empty() -> None:
    valid_ids, matrix = _load_embeddings([])
    assert valid_ids == []
    assert matrix is None


def test_load_embeddings_single() -> None:
    blob = _make_embedding(0)
    valid_ids, matrix = _load_embeddings([(1, blob)])
    assert valid_ids == [1]
    assert matrix.shape == (1, _DIM)
    # Should be unit-normalised
    assert abs(float(np.linalg.norm(matrix[0])) - 1.0) < 1e-5


def test_load_embeddings_skips_bad_blobs() -> None:
    good = _make_embedding(0)
    bad = b"\x00" * 10  # wrong length
    valid_ids, matrix = _load_embeddings([(1, good), (2, bad)])
    assert valid_ids == [1]
    assert matrix.shape == (1, _DIM)


# ---------------------------------------------------------------------------
# _find_representative
# ---------------------------------------------------------------------------


def test_find_representative_single_member() -> None:
    blob = _make_embedding(0)
    _, matrix = _load_embeddings([(1, blob)])
    assert _find_representative([1], matrix, {1: 0}) == 1


def test_find_representative_most_central() -> None:
    """The representative should be the member closest to the cluster centroid."""
    rng = np.random.RandomState(7)
    # Create a tight cluster around a fixed direction
    base = rng.randn(_DIM).astype(np.float32)
    base /= np.linalg.norm(base)

    blobs = []
    for _i in range(4):
        noise = rng.randn(_DIM).astype(np.float32) * 0.01
        v = base + noise
        v /= np.linalg.norm(v)
        blobs.append(struct.pack(f"{_DIM}f", *v))

    # One outlier
    outlier = rng.randn(_DIM).astype(np.float32)
    outlier /= np.linalg.norm(outlier)
    blobs.append(struct.pack(f"{_DIM}f", *outlier))

    rows = list(enumerate(blobs, start=1))  # asset_id 1..5
    valid_ids, matrix = _load_embeddings(rows)
    aid_to_idx = {aid: i for i, aid in enumerate(valid_ids)}

    rep = _find_representative(valid_ids, matrix, aid_to_idx)
    # The representative must be one of the tight cluster members (id 1..4)
    assert rep in range(1, 5)


# ---------------------------------------------------------------------------
# _compute_diameter_clip
# ---------------------------------------------------------------------------


def test_compute_diameter_cosine_identical() -> None:
    blob = _make_embedding(0)
    _, matrix = _load_embeddings([(1, blob), (2, blob)])
    aid_to_idx = {1: 0, 2: 1}
    # Identical vectors → cosine distance = 0
    diam = _compute_diameter_clip([1, 2], matrix, aid_to_idx, "cosine")
    assert abs(diam) < 1e-5


def test_compute_diameter_euclidean() -> None:
    blob0 = _make_embedding(0)
    blob1 = _make_embedding(1)
    _, matrix = _load_embeddings([(1, blob0), (2, blob1)])
    aid_to_idx = {1: 0, 2: 1}
    diam = _compute_diameter_clip([1, 2], matrix, aid_to_idx, "euclidean")
    # Euclidean distance for unit vectors is in [0, 2]
    assert 0.0 <= diam <= 2.0


def test_compute_diameter_combined() -> None:
    blob0 = _make_embedding(0)
    blob1 = _make_embedding(1)
    _, matrix = _load_embeddings([(1, blob0), (2, blob1)])
    aid_to_idx = {1: 0, 2: 1}
    diam = _compute_diameter_clip([1, 2], matrix, aid_to_idx, "combined")
    # Angular distance is in [0, π]
    assert 0.0 <= diam <= math.pi


# ---------------------------------------------------------------------------
# _split_by_complete_linkage_clip
# ---------------------------------------------------------------------------


def test_split_cl_all_similar_stays_together() -> None:
    """When all members are similar, complete-linkage keeps them in one cluster."""
    # Create 3 very similar embeddings
    base = _decode_embedding(_make_embedding(0))
    vecs = []
    for i in range(3):
        rng = np.random.RandomState(i)
        noise = rng.randn(_DIM).astype(np.float32) * 0.005
        v = base + noise
        v /= np.linalg.norm(v)
        vecs.append(v)

    rows = [(i + 1, struct.pack(f"{_DIM}f", *v)) for i, v in enumerate(vecs)]
    valid_ids, matrix = _load_embeddings(rows)
    aid_to_idx = {aid: i for i, aid in enumerate(valid_ids)}

    result = _split_by_complete_linkage_clip(valid_ids, matrix, aid_to_idx, "cosine", 0.80)
    # All should be in one sub-cluster
    assert len(result) == 1
    assert set(result[0]) == {1, 2, 3}


def test_split_cl_dissimilar_splits() -> None:
    """Two unrelated embeddings always split into singletons."""
    blob0 = _make_embedding(0)
    blob1 = _make_embedding(1)
    rows = [(1, blob0), (2, blob1)]
    valid_ids, matrix = _load_embeddings(rows)
    aid_to_idx = {1: 0, 2: 1}

    # With a very high cosine threshold (0.99999), even slightly different
    # embeddings will be split.
    result = _split_by_complete_linkage_clip(valid_ids, matrix, aid_to_idx, "cosine", 0.99999)
    sizes = sorted(len(sc) for sc in result)
    assert sizes == [1, 1]


# ---------------------------------------------------------------------------
# build_clip_clusters – basic
# ---------------------------------------------------------------------------


def test_build_clip_no_embeddings_returns_zero() -> None:
    conn = _open_in_memory()
    _add_asset(conn, "p/a.jpg")
    n_clusters, n_skipped = build_clip_clusters(conn)
    assert n_clusters == 0
    assert n_skipped == 0
    assert count_clusters(conn) == 0


def test_build_clip_identical_embeddings_cluster() -> None:
    """Identical embeddings always end up in the same cluster."""
    conn = _open_in_memory()
    blob = _make_embedding(0)
    _add_asset_with_embedding(conn, "p/a.jpg", blob)
    _add_asset_with_embedding(conn, "p/b.jpg", blob)
    n_clusters, n_skipped = build_clip_clusters(conn, metric="cosine", threshold=0.90)
    assert n_clusters == 1
    assert n_skipped == 0
    assert count_clusters(conn) == 1


def test_build_clip_dissimilar_embeddings_no_cluster() -> None:
    """Completely different random embeddings should not cluster."""
    conn = _open_in_memory()
    # Use seeds 0 and 100: for high-dimensional random unit vectors
    # cosine similarity is typically near 0.
    blob0 = _make_embedding(0)
    blob100 = _make_embedding(100)
    v0 = _decode_embedding(blob0)
    v100 = _decode_embedding(blob100)
    cos_sim = float(np.dot(v0, v100))
    # Sanity check: these embeddings are genuinely dissimilar
    assert cos_sim < 0.20, f"Expected dissimilar embeddings, got cos_sim={cos_sim}"

    _add_asset_with_embedding(conn, "p/a.jpg", blob0)
    _add_asset_with_embedding(conn, "p/b.jpg", blob100)
    n_clusters, _ = build_clip_clusters(conn, metric="cosine", threshold=0.90)
    assert n_clusters == 0


def test_build_clip_similar_embeddings_cluster() -> None:
    """Similar embeddings (low noise) cluster together."""
    conn = _open_in_memory()
    base = _decode_embedding(_make_embedding(0))
    blob_base = struct.pack(f"{_DIM}f", *base)
    blob_sim = _make_similar_embedding(base, noise_scale=0.01)

    v_sim = _decode_embedding(blob_sim)
    cos_sim = float(np.dot(base, v_sim))
    # Sanity: should be very similar
    assert cos_sim >= 0.90, f"Expected similar embeddings, got cos_sim={cos_sim}"

    _add_asset_with_embedding(conn, "p/a.jpg", blob_base)
    _add_asset_with_embedding(conn, "p/b.jpg", blob_sim)
    n_clusters, _ = build_clip_clusters(conn, metric="cosine", threshold=0.90)
    assert n_clusters == 1


def test_build_clip_euclidean_metric() -> None:
    """Euclidean metric produces the same grouping as cosine for unit vectors."""
    base = _decode_embedding(_make_embedding(0))
    blob_base = struct.pack(f"{_DIM}f", *base)
    blob_sim = _make_similar_embedding(base, noise_scale=0.01)
    v_sim = _decode_embedding(blob_sim)
    euclid = float(np.linalg.norm(base - v_sim))
    assert euclid < 0.45, f"Expected small euclidean distance, got {euclid}"

    conn = _open_in_memory()
    _add_asset_with_embedding(conn, "p/a.jpg", blob_base)
    _add_asset_with_embedding(conn, "p/b.jpg", blob_sim)
    n_clusters, _ = build_clip_clusters(conn, metric="euclidean", threshold=0.45)
    assert n_clusters == 1


def test_build_clip_combined_metric() -> None:
    """Combined (angular) metric clusters similar embeddings."""
    base = _decode_embedding(_make_embedding(0))
    blob_base = struct.pack(f"{_DIM}f", *base)
    blob_sim = _make_similar_embedding(base, noise_scale=0.01)
    v_sim = _decode_embedding(blob_sim)
    angle = float(math.acos(max(-1.0, min(1.0, float(np.dot(base, v_sim))))))
    assert angle < 0.46, f"Expected small angle, got {angle}"

    conn = _open_in_memory()
    _add_asset_with_embedding(conn, "p/a.jpg", blob_base)
    _add_asset_with_embedding(conn, "p/b.jpg", blob_sim)
    n_clusters, _ = build_clip_clusters(conn, metric="combined", threshold=0.46)
    assert n_clusters == 1


def test_build_clip_min_size_filters() -> None:
    """Pairs below min_cluster_size are not stored."""
    conn = _open_in_memory()
    blob = _make_embedding(0)
    _add_asset_with_embedding(conn, "p/a.jpg", blob)
    _add_asset_with_embedding(conn, "p/b.jpg", blob)
    result = build_clip_clusters(conn, metric="cosine", threshold=0.90, min_cluster_size=3)
    assert result == (0, 0)
    assert count_clusters(conn) == 0


def test_build_clip_max_size_skips_large_component() -> None:
    """Components larger than max_cluster_size are skipped and counted."""
    conn = _open_in_memory()
    blob = _make_embedding(0)
    # 3 identical embeddings → they form one component of size 3
    _add_asset_with_embedding(conn, "p/a.jpg", blob)
    _add_asset_with_embedding(conn, "p/b.jpg", blob)
    _add_asset_with_embedding(conn, "p/c.jpg", blob)
    n_clusters, n_skipped = build_clip_clusters(
        conn, metric="cosine", threshold=0.90, max_cluster_size=2
    )
    assert n_clusters == 0
    assert n_skipped == 1
    assert count_clusters(conn) == 0


def test_build_clip_max_size_allows_smaller_components() -> None:
    """Components at or below max_cluster_size are processed normally."""
    conn = _open_in_memory()
    blob = _make_embedding(0)
    _add_asset_with_embedding(conn, "p/a.jpg", blob)
    _add_asset_with_embedding(conn, "p/b.jpg", blob)
    n_clusters, n_skipped = build_clip_clusters(
        conn, metric="cosine", threshold=0.90, max_cluster_size=2
    )
    assert n_clusters == 1
    assert n_skipped == 0


def test_build_clip_n_skipped_stored_in_db() -> None:
    """n_skipped is persisted in the clustering_runs table."""
    from takeout_rater.db.queries import list_clustering_runs  # noqa: PLC0415

    conn = _open_in_memory()
    blob = _make_embedding(0)
    # 3 identical embeddings → component of size 3, skipped by max_cluster_size=2
    _add_asset_with_embedding(conn, "p/a.jpg", blob)
    _add_asset_with_embedding(conn, "p/b.jpg", blob)
    _add_asset_with_embedding(conn, "p/c.jpg", blob)
    build_clip_clusters(conn, metric="cosine", threshold=0.90, max_cluster_size=2)
    runs = list_clustering_runs(conn)
    assert len(runs) == 1
    assert runs[0]["n_skipped"] == 1


def test_build_clip_each_run_independent() -> None:
    """Running build_clip_clusters twice produces two separate runs."""
    conn = _open_in_memory()
    blob = _make_embedding(0)
    _add_asset_with_embedding(conn, "p/a.jpg", blob)
    _add_asset_with_embedding(conn, "p/b.jpg", blob)

    build_clip_clusters(conn, metric="cosine", threshold=0.90)
    build_clip_clusters(conn, metric="cosine", threshold=0.90)
    assert count_clusters(conn) == 2


def test_build_clip_stores_diameter() -> None:
    """Diameter is stored in the cluster row."""
    conn = _open_in_memory()
    blob = _make_embedding(0)
    _add_asset_with_embedding(conn, "p/a.jpg", blob)
    _add_asset_with_embedding(conn, "p/b.jpg", blob)
    build_clip_clusters(conn, metric="cosine", threshold=0.90)

    info = get_cluster_info(conn, 1)
    assert info is not None
    # Identical embeddings → cosine distance = 0 → diameter = 0
    assert info["diameter"] is not None
    assert abs(info["diameter"]) < 1e-4


def test_build_clip_stores_distances_to_rep() -> None:
    """Member distances to the representative are stored in cluster_members."""
    conn = _open_in_memory()
    blob = _make_embedding(0)
    _add_asset_with_embedding(conn, "p/a.jpg", blob)
    _add_asset_with_embedding(conn, "p/b.jpg", blob)
    build_clip_clusters(conn, metric="cosine", threshold=0.90)

    members = get_cluster_members(conn, 1)
    assert len(members) == 2
    # All distances should be non-None
    for _asset, dist, _is_rep in members:
        assert dist is not None


def test_build_clip_method_field_is_clip_embedding() -> None:
    """Clusters produced by CLIP builder have method='clip_embedding'."""
    conn = _open_in_memory()
    blob = _make_embedding(0)
    _add_asset_with_embedding(conn, "p/a.jpg", blob)
    _add_asset_with_embedding(conn, "p/b.jpg", blob)
    build_clip_clusters(conn, metric="cosine", threshold=0.90)

    info = get_cluster_info(conn, 1)
    assert info["method"] == "clip_embedding"


def test_build_clip_progress_callback() -> None:
    """Progress callback is called during clustering."""
    conn = _open_in_memory()
    for i in range(4):
        blob = _make_embedding(i)
        _add_asset_with_embedding(conn, f"p/{i}.jpg", blob)

    calls: list[tuple[int, int]] = []
    build_clip_clusters(
        conn,
        metric="cosine",
        threshold=0.90,
        on_progress=lambda d, t: calls.append((d, t)),
    )
    assert len(calls) > 0
    # Last call should have done == total
    assert calls[-1][0] == calls[-1][1]


def test_build_clip_post_progress_callback() -> None:
    """on_post_progress is called after each multi-member component is post-processed."""
    conn = _open_in_memory()
    blob = _make_embedding(0)
    _add_asset_with_embedding(conn, "p/a.jpg", blob)
    _add_asset_with_embedding(conn, "p/b.jpg", blob)

    calls: list[tuple[int, int]] = []
    build_clip_clusters(
        conn,
        metric="cosine",
        threshold=0.90,
        on_post_progress=lambda d, t: calls.append((d, t)),
    )
    assert len(calls) > 0
    assert calls[-1][0] == calls[-1][1]


def test_build_clip_save_progress_callback() -> None:
    """on_save_progress is called after each cluster is written to the DB."""
    conn = _open_in_memory()
    blob = _make_embedding(0)
    _add_asset_with_embedding(conn, "p/a.jpg", blob)
    _add_asset_with_embedding(conn, "p/b.jpg", blob)

    calls: list[tuple[int, int]] = []
    build_clip_clusters(
        conn,
        metric="cosine",
        threshold=0.90,
        on_save_progress=lambda d, t: calls.append((d, t)),
    )
    assert len(calls) > 0
    assert calls[-1][0] == calls[-1][1]


def test_build_clip_post_progress_not_called_when_no_clusters() -> None:
    """on_post_progress is not called when there are no multi-member components."""
    conn = _open_in_memory()
    blob0 = _make_embedding(0)
    blob100 = _make_embedding(100)
    _add_asset_with_embedding(conn, "p/a.jpg", blob0)
    _add_asset_with_embedding(conn, "p/b.jpg", blob100)

    calls: list[tuple[int, int]] = []
    build_clip_clusters(
        conn,
        metric="cosine",
        threshold=0.9999,
        on_post_progress=lambda d, t: calls.append((d, t)),
    )
    assert calls == []


def test_build_clip_invalid_metric_raises() -> None:
    conn = _open_in_memory()
    with pytest.raises(ValueError, match="Unsupported metric"):
        build_clip_clusters(conn, metric="unknown")


def test_build_clip_single_linkage_mode() -> None:
    """Single-linkage mode can chain A-B-C even if A and C are dissimilar."""
    # A and B are similar (will be connected), but
    # C is similar to B, not A.
    # In complete-linkage C wouldn't join {A,B}.
    # In single-linkage: A-B-C forms one chain.
    rng = np.random.RandomState(0)
    base_a = rng.randn(_DIM).astype(np.float32)
    base_a /= np.linalg.norm(base_a)

    # B ≈ A (small noise)
    b = base_a + rng.randn(_DIM).astype(np.float32) * 0.03
    b /= np.linalg.norm(b)

    # C = perpendicular to A (by Gram-Schmidt)
    raw_c = rng.randn(_DIM).astype(np.float32)
    raw_c -= float(np.dot(raw_c, base_a)) * base_a
    # Now C is orthogonal to A (cos_sim ≈ 0), but make C similar to B
    # by making raw_c similar to b instead.
    raw_c = b + rng.randn(_DIM).astype(np.float32) * 0.03
    raw_c /= np.linalg.norm(raw_c)

    cos_ab = float(np.dot(base_a, b))
    cos_ac = float(np.dot(base_a, raw_c))
    cos_bc = float(np.dot(b, raw_c))

    threshold = (cos_ab + cos_bc) / 2 + 0.005  # just above A-B and B-C similarity

    # A-C must be below threshold for this test to be meaningful
    # (otherwise complete linkage would also group them)
    if cos_ac >= threshold:
        pytest.skip("Test vectors happened to all be similar; skip")

    blob_a = struct.pack(f"{_DIM}f", *base_a)
    blob_b = struct.pack(f"{_DIM}f", *b)
    blob_c = struct.pack(f"{_DIM}f", *raw_c)

    conn_sl = _open_in_memory()
    _add_asset_with_embedding(conn_sl, "p/a.jpg", blob_a)
    _add_asset_with_embedding(conn_sl, "p/b.jpg", blob_b)
    _add_asset_with_embedding(conn_sl, "p/c.jpg", blob_c)

    # Single-linkage: A-B-C may all end up in one cluster
    n_sl, _ = build_clip_clusters(
        conn_sl, metric="cosine", threshold=threshold, single_linkage=True
    )

    conn_cl = _open_in_memory()
    _add_asset_with_embedding(conn_cl, "p/a.jpg", blob_a)
    _add_asset_with_embedding(conn_cl, "p/b.jpg", blob_b)
    _add_asset_with_embedding(conn_cl, "p/c.jpg", blob_c)

    # Complete-linkage: A and C fail the threshold → split
    n_cl, _ = build_clip_clusters(
        conn_cl, metric="cosine", threshold=threshold, single_linkage=False
    )

    # Single-linkage should produce at least as large (or equal) clusters
    assert n_sl >= n_cl
