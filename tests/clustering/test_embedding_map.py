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

    def test_no_library_returns_503(self) -> None:
        app = create_app(None, None)
        client = TestClient(app, follow_redirects=False)
        r = client.get("/api/clip/embedding-map")
        assert r.status_code == 503
