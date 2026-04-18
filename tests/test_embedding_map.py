"""Tests for the CLIP embedding-map endpoint and builder.

Covers:
- load_clip_embeddings_with_relpaths DB query
- build_embedding_map with small synthetic data
- GET /api/clip/embedding-map endpoint (empty, with data, cache, refresh)
"""

from __future__ import annotations

import struct
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from takeout_rater.db.queries import (
    bulk_upsert_clip_embeddings,
    load_clip_embeddings_with_relpaths,
)
from takeout_rater.ui.app import create_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 768


def _make_embedding(seed: int = 0) -> bytes:
    """Create a deterministic normalised 768-dim float32 embedding blob."""
    import numpy as np

    rng = np.random.RandomState(seed)
    vec = rng.randn(_DIM).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return struct.pack(f"{_DIM}f", *vec)


def _setup_db(tmp_path: Path):
    from takeout_rater.db.connection import open_library_db

    return open_library_db(tmp_path)


def _insert_asset(conn, relpath: str = "photo.jpg") -> int:
    now = int(time.time())
    cur = conn.execute(
        "INSERT INTO assets (relpath, filename, ext, indexed_at) VALUES (?, ?, ?, ?)",
        (relpath, relpath, ".jpg", now),
    )
    conn.commit()
    return cur.lastrowid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_conn(tmp_path: Path):
    return _setup_db(tmp_path)


@pytest.fixture()
def client_with_db(tmp_path: Path) -> TestClient:
    from takeout_rater.db.connection import open_library_db

    conn = open_library_db(tmp_path)
    app = create_app(tmp_path, conn, db_root=tmp_path)
    return TestClient(app, follow_redirects=False)


# ---------------------------------------------------------------------------
# DB query: load_clip_embeddings_with_relpaths
# ---------------------------------------------------------------------------


class TestLoadClipEmbeddingsWithRelpaths:
    def test_empty(self, db_conn) -> None:
        rows = load_clip_embeddings_with_relpaths(db_conn)
        assert rows == []

    def test_returns_asset_id_blob_relpath(self, db_conn) -> None:
        aid = _insert_asset(db_conn, "Photos/img.jpg")
        blob = _make_embedding(0)
        bulk_upsert_clip_embeddings(db_conn, [(aid, blob)])

        rows = load_clip_embeddings_with_relpaths(db_conn)
        assert len(rows) == 1
        got_id, got_blob, got_relpath = rows[0]
        assert got_id == aid
        assert got_blob == blob
        assert got_relpath == "Photos/img.jpg"

    def test_ordered_by_asset_id(self, db_conn) -> None:
        a1 = _insert_asset(db_conn, "a.jpg")
        a2 = _insert_asset(db_conn, "b.jpg")
        a3 = _insert_asset(db_conn, "c.jpg")
        bulk_upsert_clip_embeddings(
            db_conn,
            [(a1, _make_embedding(1)), (a3, _make_embedding(3)), (a2, _make_embedding(2))],
        )
        rows = load_clip_embeddings_with_relpaths(db_conn)
        ids = [r[0] for r in rows]
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# build_embedding_map unit tests
# ---------------------------------------------------------------------------


class TestBuildEmbeddingMap:
    def test_empty_input(self) -> None:
        from takeout_rater.clustering.embedding_map import build_embedding_map

        result = build_embedding_map([])
        assert result == {
            "points": [],
            "clusters": [],
            "total": 0,
            "params": {"clustering_method": "kmeans", "max_clusters": 24},
        }

    def test_single_point(self) -> None:
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

    def test_many_points_full_pipeline(self) -> None:
        """With enough points UMAP runs; output shape and keys are correct."""
        from takeout_rater.clustering.embedding_map import build_embedding_map

        n = 30
        rows = [(i + 1, _make_embedding(i), f"img{i}.jpg") for i in range(n)]
        result = build_embedding_map(rows)
        assert result["total"] == n
        assert len(result["points"]) == n
        # Every point has required keys
        for pt in result["points"]:
            assert {"asset_id", "x", "y", "z", "cluster_id", "relpath"} <= pt.keys()
        # Every cluster has a representative
        for cl in result["clusters"]:
            assert cl["representative_asset_id"] is not None
            assert cl["size"] > 0

    def test_few_points_skips_umap(self) -> None:
        """With < 15 points UMAP is skipped; output is still valid."""
        from takeout_rater.clustering.embedding_map import build_embedding_map

        n = 5
        rows = [(i + 1, _make_embedding(i), f"img{i}.jpg") for i in range(n)]
        result = build_embedding_map(rows)
        assert result["total"] == n
        for pt in result["points"]:
            assert "x" in pt and "y" in pt and "z" in pt

    def test_max_clusters_caps_kmeans(self) -> None:
        from takeout_rater.clustering.embedding_map import build_embedding_map

        n = 25
        rows = [(i + 1, _make_embedding(i), f"img{i}.jpg") for i in range(n)]
        result = build_embedding_map(rows, max_clusters=3)
        assert result["params"]["max_clusters"] == 3
        assert len(result["clusters"]) <= 3

    def test_hdbscan_method(self) -> None:
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


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


class TestEmbeddingMapEndpoint:
    def test_empty_returns_zero_total(self, client_with_db: TestClient) -> None:
        r = client_with_db.get("/api/clip/embedding-map")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 0
        assert data["points"] == []
        assert data["clusters"] == []

    def test_with_embeddings_returns_points(self, tmp_path: Path) -> None:
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

    def test_endpoint_accepts_clustering_params(self, tmp_path: Path) -> None:
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

    def test_result_is_cached(self, tmp_path: Path) -> None:
        """Second call with identical params returns cached result."""
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
        # The cache attribute is set on app.state
        assert getattr(app.state, "clip_embedding_map", None) is not None

    def test_clustering_param_change_bypasses_cache(self, tmp_path: Path) -> None:
        """Changing clustering params recomputes instead of returning cached data."""
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

    def test_refresh_clears_cache(self, tmp_path: Path) -> None:
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

    def test_no_library_returns_503(self) -> None:
        app = create_app(None, None)
        client = TestClient(app, follow_redirects=False)
        r = client.get("/api/clip/embedding-map")
        assert r.status_code == 503
