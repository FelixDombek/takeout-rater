"""Slow embedding-map tests.

Excluded from the default test run (--ignore=tests/slow) because these tests
exercise the full UMAP pipeline.
"""

from __future__ import annotations

import struct
import time
from pathlib import Path

from fastapi.testclient import TestClient

from takeout_rater.db.queries import bulk_upsert_clip_embeddings
from takeout_rater.ui.app import create_app

_DIM = 768


def _make_embedding(seed: int = 0) -> bytes:
    """Create a deterministic normalised 768-dim float32 embedding blob."""
    import numpy as np

    rng = np.random.RandomState(seed)
    vec = rng.randn(_DIM).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return struct.pack(f"{_DIM}f", *vec)


def _insert_asset(conn, relpath: str = "photo.jpg") -> int:
    now = int(time.time())
    cur = conn.execute(
        "INSERT INTO assets (relpath, filename, ext, indexed_at) VALUES (?, ?, ?, ?)",
        (relpath, relpath, ".jpg", now),
    )
    conn.commit()
    return cur.lastrowid


def test_single_point() -> None:
    from takeout_rater.clustering.embedding_map import build_embedding_map

    blob = _make_embedding(0)
    rows = [(1, blob, "a.jpg")]
    result = build_embedding_map(rows)
    assert result["total"] == 1
    assert len(result["points"]) == 1
    pt = result["points"][0]
    assert pt["asset_id"] == 1
    assert pt["relpath"] == "a.jpg"
    assert "x" in pt and "y" in pt and "z" in pt
    assert "cluster_id" in pt
    assert len(result["clusters"]) >= 1


def test_few_points_skips_umap() -> None:
    from takeout_rater.clustering.embedding_map import build_embedding_map

    n = 5
    rows = [(i + 1, _make_embedding(i), f"img{i}.jpg") for i in range(n)]
    result = build_embedding_map(rows)
    assert result["total"] == n
    for pt in result["points"]:
        assert "x" in pt and "y" in pt and "z" in pt


def test_many_points_full_pipeline() -> None:
    """With enough points UMAP runs; output shape and keys are correct."""
    from takeout_rater.clustering.embedding_map import build_embedding_map

    n = 30
    rows = [(i + 1, _make_embedding(i), f"img{i}.jpg") for i in range(n)]
    result = build_embedding_map(rows)
    assert result["total"] == n
    assert len(result["points"]) == n
    for pt in result["points"]:
        assert {"asset_id", "x", "y", "z", "cluster_id", "relpath"} <= pt.keys()
    for cl in result["clusters"]:
        assert cl["representative_asset_id"] is not None
        assert cl["size"] > 0


def test_max_clusters_caps_kmeans() -> None:
    from takeout_rater.clustering.embedding_map import build_embedding_map

    n = 25
    rows = [(i + 1, _make_embedding(i), f"img{i}.jpg") for i in range(n)]
    result = build_embedding_map(rows, max_clusters=3)
    assert result["params"]["max_clusters"] == 3
    assert len(result["clusters"]) <= 3


def test_hdbscan_method() -> None:
    from takeout_rater.clustering.embedding_map import build_embedding_map

    n = 25
    rows = [(i + 1, _make_embedding(i), f"img{i}.jpg") for i in range(n)]
    result = build_embedding_map(rows, clustering_method="hdbscan", max_clusters=4)
    assert result["params"] == {
        "clustering_method": "hdbscan",
        "max_clusters": 4,
    }
    assert result["total"] == n
    assert len(result["points"]) == n
    non_noise = [cl for cl in result["clusters"] if cl["cluster_id"] >= 0]
    assert len(non_noise) <= 4


def test_with_embeddings_returns_points(tmp_path: Path) -> None:
    from takeout_rater.db.connection import open_library_db

    conn = open_library_db(tmp_path)
    n = 20
    rows = []
    for i in range(n):
        aid = _insert_asset(conn, f"img{i}.jpg")
        rows.append((aid, _make_embedding(i)))
    bulk_upsert_clip_embeddings(conn, rows)
    app = create_app(tmp_path, conn, db_root=tmp_path)
    client = TestClient(app, follow_redirects=False)

    r = client.get("/api/clip/embedding-map")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == n
    assert len(data["points"]) == n
    assert len(data["clusters"]) >= 1
    assert data["params"] == {
        "clustering_method": "kmeans",
        "max_clusters": 24,
    }


def test_endpoint_accepts_clustering_params(tmp_path: Path) -> None:
    from takeout_rater.db.connection import open_library_db

    conn = open_library_db(tmp_path)
    for i in range(20):
        aid = _insert_asset(conn, f"img{i}.jpg")
        bulk_upsert_clip_embeddings(conn, [(aid, _make_embedding(i))])
    app = create_app(tmp_path, conn, db_root=tmp_path)
    client = TestClient(app, follow_redirects=False)

    r = client.get("/api/clip/embedding-map?clustering_method=hdbscan&max_clusters=3")
    assert r.status_code == 200
    data = r.json()
    assert data["params"] == {
        "clustering_method": "hdbscan",
        "max_clusters": 3,
    }
    non_noise = [cl for cl in data["clusters"] if cl["cluster_id"] >= 0]
    assert len(non_noise) <= 3


def test_result_is_cached(tmp_path: Path) -> None:
    from takeout_rater.db.connection import open_library_db

    conn = open_library_db(tmp_path)
    n = 10
    rows_db = []
    for i in range(n):
        aid = _insert_asset(conn, f"img{i}.jpg")
        rows_db.append((aid, _make_embedding(i)))
    bulk_upsert_clip_embeddings(conn, rows_db)
    app = create_app(tmp_path, conn, db_root=tmp_path)
    client = TestClient(app, follow_redirects=False)

    r1 = client.get("/api/clip/embedding-map")
    assert r1.status_code == 200
    app.state.clip_embedding_map["cache_sentinel"] = "cached"

    r2 = client.get("/api/clip/embedding-map")
    assert r2.status_code == 200
    assert r1.json()["total"] == r2.json()["total"]
    assert r2.json()["cache_sentinel"] == "cached"
    assert getattr(app.state, "clip_embedding_map", None) is not None


def test_clustering_param_change_bypasses_cache(tmp_path: Path) -> None:
    from takeout_rater.db.connection import open_library_db

    conn = open_library_db(tmp_path)
    n = 10
    rows_db = []
    for i in range(n):
        aid = _insert_asset(conn, f"img{i}.jpg")
        rows_db.append((aid, _make_embedding(i)))
    bulk_upsert_clip_embeddings(conn, rows_db)
    app = create_app(tmp_path, conn, db_root=tmp_path)
    client = TestClient(app, follow_redirects=False)

    r1 = client.get("/api/clip/embedding-map")
    assert r1.status_code == 200
    app.state.clip_embedding_map["cache_sentinel"] = "cached"

    r2 = client.get("/api/clip/embedding-map?max_clusters=3")
    assert r2.status_code == 200
    data = r2.json()
    assert data["total"] == n
    assert data["params"]["max_clusters"] == 3
    assert "cache_sentinel" not in data


def test_refresh_clears_cache(tmp_path: Path) -> None:
    from takeout_rater.db.connection import open_library_db

    conn = open_library_db(tmp_path)
    n = 10
    for i in range(n):
        aid = _insert_asset(conn, f"img{i}.jpg")
        bulk_upsert_clip_embeddings(conn, [(aid, _make_embedding(i))])
    app = create_app(tmp_path, conn, db_root=tmp_path)
    client = TestClient(app, follow_redirects=False)

    client.get("/api/clip/embedding-map")
    r = client.get("/api/clip/embedding-map?refresh=true")
    assert r.status_code == 200
    assert r.json()["total"] == n
