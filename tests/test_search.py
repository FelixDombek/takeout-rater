"""Tests for the CLIP semantic search feature.

Covers:
- clip_embeddings DB queries (upsert, bulk_upsert, load, count, list without)
- /api/search endpoint (empty query, no embeddings, results)
- /search page returns 200
- embed job start endpoint
- shared CLIP backbone module
"""

from __future__ import annotations

import struct
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from takeout_rater.db.queries import (
    bulk_upsert_clip_embeddings,
    count_clip_embeddings,
    delete_clip_user_tag,
    insert_clip_user_tag,
    list_asset_ids_without_embedding,
    list_clip_user_tags,
    load_all_clip_embeddings,
    upsert_clip_embedding,
)
from takeout_rater.ui.app import create_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DIM = 768


def _make_embedding(seed: int = 0) -> bytes:
    """Create a deterministic fake 768-dim float32 embedding blob."""
    import numpy as np

    rng = np.random.RandomState(seed)
    vec = rng.randn(_DIM).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return struct.pack(f"{_DIM}f", *vec)


def _setup_db(tmp_path: Path):
    """Create a fresh library DB and return (conn, tmp_path)."""
    from takeout_rater.db.connection import open_library_db

    conn = open_library_db(tmp_path)
    return conn


def _insert_asset(conn, relpath: str = "photo.jpg") -> int:
    """Insert a minimal asset row and return its id."""
    now = int(time.time())
    cur = conn.execute(
        "INSERT INTO assets (relpath, filename, ext, indexed_at) VALUES (?, ?, ?, ?)",
        (relpath, relpath, ".jpg", now),
    )
    conn.commit()
    return cur.lastrowid


@pytest.fixture()
def db_conn(tmp_path: Path):
    """Return a connection to a fresh library DB."""
    return _setup_db(tmp_path)


@pytest.fixture()
def client_with_db(tmp_path: Path) -> TestClient:
    """App with an in-memory SQLite DB and a library root."""
    from takeout_rater.db.connection import open_library_db

    conn = open_library_db(tmp_path)
    app = create_app(tmp_path, conn)
    return TestClient(app, follow_redirects=False)


@pytest.fixture()
def client_no_db() -> TestClient:
    """App with no library configured."""
    app = create_app(None, None)
    return TestClient(app, follow_redirects=False)


# ---------------------------------------------------------------------------
# DB query tests
# ---------------------------------------------------------------------------


class TestClipEmbeddingQueries:
    def test_count_empty(self, db_conn) -> None:
        assert count_clip_embeddings(db_conn) == 0

    def test_upsert_and_count(self, db_conn) -> None:
        aid = _insert_asset(db_conn, "a.jpg")
        blob = _make_embedding(1)
        upsert_clip_embedding(db_conn, aid, blob)
        db_conn.commit()
        assert count_clip_embeddings(db_conn) == 1

    def test_upsert_overwrites(self, db_conn) -> None:
        aid = _insert_asset(db_conn, "a.jpg")
        blob1 = _make_embedding(1)
        blob2 = _make_embedding(2)
        upsert_clip_embedding(db_conn, aid, blob1)
        db_conn.commit()
        upsert_clip_embedding(db_conn, aid, blob2)
        db_conn.commit()
        assert count_clip_embeddings(db_conn) == 1
        rows = load_all_clip_embeddings(db_conn)
        assert rows[0][1] == blob2

    def test_list_asset_ids_without_embedding(self, db_conn) -> None:
        a1 = _insert_asset(db_conn, "a.jpg")
        a2 = _insert_asset(db_conn, "b.jpg")
        upsert_clip_embedding(db_conn, a1, _make_embedding(1))
        db_conn.commit()
        without = list_asset_ids_without_embedding(db_conn)
        assert a2 in without
        assert a1 not in without

    def test_list_asset_ids_without_embedding_all_embedded(self, db_conn) -> None:
        a1 = _insert_asset(db_conn, "a.jpg")
        upsert_clip_embedding(db_conn, a1, _make_embedding(1))
        db_conn.commit()
        assert list_asset_ids_without_embedding(db_conn) == []

    def test_bulk_upsert(self, db_conn) -> None:
        a1 = _insert_asset(db_conn, "a.jpg")
        a2 = _insert_asset(db_conn, "b.jpg")
        rows = [(a1, _make_embedding(1)), (a2, _make_embedding(2))]
        bulk_upsert_clip_embeddings(db_conn, rows)
        assert count_clip_embeddings(db_conn) == 2

    def test_load_all_returns_sorted(self, db_conn) -> None:
        a1 = _insert_asset(db_conn, "a.jpg")
        a2 = _insert_asset(db_conn, "b.jpg")
        blob1 = _make_embedding(1)
        blob2 = _make_embedding(2)
        bulk_upsert_clip_embeddings(db_conn, [(a2, blob2), (a1, blob1)])
        loaded = load_all_clip_embeddings(db_conn)
        assert len(loaded) == 2
        assert loaded[0][0] == a1  # sorted by asset_id
        assert loaded[1][0] == a2
        assert loaded[0][1] == blob1
        assert loaded[1][1] == blob2


# ---------------------------------------------------------------------------
# CLIP user tags DB query tests
# ---------------------------------------------------------------------------


class TestClipUserTagQueries:
    def test_list_empty(self, db_conn) -> None:
        assert list_clip_user_tags(db_conn) == []

    def test_insert_and_list(self, db_conn) -> None:
        assert insert_clip_user_tag(db_conn, "beach sunset")
        tags = list_clip_user_tags(db_conn)
        assert "beach sunset" in tags

    def test_insert_duplicate_returns_false(self, db_conn) -> None:
        insert_clip_user_tag(db_conn, "mountain")
        assert not insert_clip_user_tag(db_conn, "mountain")
        assert list_clip_user_tags(db_conn).count("mountain") == 1

    def test_delete_existing(self, db_conn) -> None:
        insert_clip_user_tag(db_conn, "forest")
        assert delete_clip_user_tag(db_conn, "forest")
        assert "forest" not in list_clip_user_tags(db_conn)

    def test_delete_nonexistent_returns_false(self, db_conn) -> None:
        assert not delete_clip_user_tag(db_conn, "doesnotexist")


# ---------------------------------------------------------------------------
# Search API tests
# ---------------------------------------------------------------------------


class TestSearchAPI:
    def test_search_empty_query_returns_empty(self, client_with_db: TestClient) -> None:
        resp = client_with_db.get("/api/search?q=")
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"] == []

    def test_search_no_embeddings_returns_error(self, client_with_db: TestClient) -> None:
        resp = client_with_db.get("/api/search?q=sunset")
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"] == []
        assert "error" in data or data["total"] == 0

    def test_search_page_returns_200(self, client_with_db: TestClient) -> None:
        resp = client_with_db.get("/search")
        assert resp.status_code == 200
        assert b"Semantic Search" in resp.content

    def test_search_page_redirects_without_db(self, client_no_db: TestClient) -> None:
        resp = client_no_db.get("/search")
        assert resp.status_code in (302, 307)
        assert "/setup" in resp.headers.get("location", "")

    def test_search_partial_empty_query(self, client_with_db: TestClient) -> None:
        resp = client_with_db.get("/api/search?q=&partial=1")
        assert resp.status_code == 200
        assert resp.headers.get("content-type", "").startswith("text/html")

    def test_search_partial_no_embeddings(self, client_with_db: TestClient) -> None:
        resp = client_with_db.get("/api/search?q=sunset&partial=1")
        assert resp.status_code == 200
        assert b"data-no-embeddings" in resp.content


# ---------------------------------------------------------------------------
# Embed job tests
# ---------------------------------------------------------------------------


class TestEmbedJob:
    def test_start_embed_job_returns_started(self, client_with_db: TestClient) -> None:
        resp = client_with_db.post("/api/jobs/embed/start", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "started"

    def test_start_embed_job_conflicts_when_running(self, client_with_db: TestClient) -> None:
        from takeout_rater.api.jobs import JobProgress

        # Inject a fake running job
        app = client_with_db.app
        jobs = app.state.jobs
        jobs["embed"] = JobProgress(job_type="embed", running=True)

        resp = client_with_db.post("/api/jobs/embed/start", json={})
        assert resp.status_code == 409

    def test_embed_status_endpoint(self, client_with_db: TestClient) -> None:
        resp = client_with_db.get("/api/jobs/status?job_type=embed")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_type"] == "embed"
        assert data["running"] is False

    def test_start_embed_without_db_returns_503(self, client_no_db: TestClient) -> None:
        resp = client_no_db.post("/api/jobs/embed/start", json={})
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Shared CLIP backbone tests
# ---------------------------------------------------------------------------


class TestClipBackbone:
    def test_is_available(self) -> None:
        from takeout_rater.scorers.adapters.clip_backbone import is_available

        result = is_available()
        assert isinstance(result, bool)

    def test_embedding_dim_constant(self) -> None:
        from takeout_rater.scorers.adapters.clip_backbone import EMBEDDING_DIM

        assert EMBEDDING_DIM == 768

    def test_get_clip_model_passes_quick_gelu(self, monkeypatch) -> None:
        """get_clip_model must pass quick_gelu=True to suppress the openai pretrained warning."""
        from unittest.mock import MagicMock  # noqa: PLC0415

        import takeout_rater.scorers.adapters.clip_backbone as backbone  # noqa: PLC0415

        create_calls: list[dict] = []

        fake_model = MagicMock()

        def fake_create(model_name, pretrained=None, **kwargs):  # type: ignore[no-untyped-def]
            create_calls.append({"model_name": model_name, "pretrained": pretrained, **kwargs})
            return fake_model, None, MagicMock()

        # Reset the singleton so get_clip_model triggers a fresh load
        monkeypatch.setattr(backbone, "_clip_model", None)
        monkeypatch.setattr(backbone, "_preprocess", None)
        monkeypatch.setattr(backbone, "_tokenizer", None)
        monkeypatch.setattr(backbone, "_device", None)

        import open_clip  # noqa: PLC0415

        monkeypatch.setattr(open_clip, "create_model_and_transforms", fake_create)
        monkeypatch.setattr(open_clip, "get_tokenizer", lambda _name: MagicMock())

        backbone.get_clip_model()

        assert len(create_calls) == 1
        assert create_calls[0].get("quick_gelu") is True

        # Clean up singleton state so other tests are not affected
        monkeypatch.setattr(backbone, "_clip_model", None)
        monkeypatch.setattr(backbone, "_preprocess", None)
        monkeypatch.setattr(backbone, "_tokenizer", None)
        monkeypatch.setattr(backbone, "_device", None)


# ---------------------------------------------------------------------------
# Migration test
# ---------------------------------------------------------------------------


class TestClipEmbeddingsMigration:
    def test_migration_creates_table(self, db_conn) -> None:
        """The clip_embeddings table should exist after migration."""
        tables = {
            row[0]
            for row in db_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "clip_embeddings" in tables

    def test_migration_creates_user_tags_table(self, db_conn) -> None:
        """The clip_user_tags table should exist after migration."""
        tables = {
            row[0]
            for row in db_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "clip_user_tags" in tables

    def test_incremental_migration_from_v9(self, tmp_path: Path) -> None:
        """Migrating from v9 should create clip_embeddings and clip_user_tags."""
        import sqlite3

        from takeout_rater.db.schema import _MIGRATIONS_DIR, CURRENT_SCHEMA_VERSION, migrate

        # Create a v9 DB by applying baseline then rolling back to v9
        db_path = tmp_path / "lib.sqlite"
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")

        # Apply baseline
        sql = (_MIGRATIONS_DIR / "0001_initial_schema.sql").read_text(encoding="utf-8")
        conn.executescript(sql)

        # Roll back to v9
        conn.execute("PRAGMA user_version = 9")
        conn.execute("DROP TABLE IF EXISTS clip_embeddings")
        conn.execute("DROP TABLE IF EXISTS clip_user_tags")
        conn.commit()

        # Now migrate
        migrate(conn)

        version = conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == CURRENT_SCHEMA_VERSION

        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "clip_embeddings" in tables
        assert "clip_user_tags" in tables
        conn.close()


# ---------------------------------------------------------------------------
# Jobs page embed card
# ---------------------------------------------------------------------------


class TestJobsPageEmbedCard:
    def test_clip_page_contains_embed_card(self, client_with_db: TestClient) -> None:
        resp = client_with_db.get("/clip")
        assert resp.status_code == 200
        assert b"CLIP Embeddings" in resp.content
        assert b"btn-embed" in resp.content

    def test_jobs_page_does_not_contain_embed_card(self, client_with_db: TestClient) -> None:
        resp = client_with_db.get("/jobs")
        assert resp.status_code == 200
        assert b"btn-embed" not in resp.content

    def test_nav_bar_contains_search_link(self, client_with_db: TestClient) -> None:
        resp = client_with_db.get("/clip")
        assert resp.status_code == 200
        assert b"/search" in resp.content
        assert b"Search" in resp.content

    def test_nav_bar_contains_clip_link(self, client_with_db: TestClient) -> None:
        resp = client_with_db.get("/jobs")
        assert resp.status_code == 200
        assert b"/clip" in resp.content
        assert b"CLIP" in resp.content


# ---------------------------------------------------------------------------
# CLIP user tags API
# ---------------------------------------------------------------------------


class TestClipUserTagsApi:
    def test_list_tags_empty(self, client_with_db: TestClient) -> None:
        resp = client_with_db.get("/api/clip/tags")
        assert resp.status_code == 200
        assert resp.json() == {"tags": []}

    def test_add_tag(self, client_with_db: TestClient) -> None:
        resp = client_with_db.post("/api/clip/tags", json={"term": "golden retriever"})
        assert resp.status_code == 201
        data = resp.json()
        assert "golden retriever" in data["tags"]

    def test_add_duplicate_tag_returns_409(self, client_with_db: TestClient) -> None:
        client_with_db.post("/api/clip/tags", json={"term": "puppy"})
        resp = client_with_db.post("/api/clip/tags", json={"term": "puppy"})
        assert resp.status_code == 409

    def test_add_empty_tag_returns_400(self, client_with_db: TestClient) -> None:
        resp = client_with_db.post("/api/clip/tags", json={"term": "  "})
        assert resp.status_code == 400

    def test_delete_tag(self, client_with_db: TestClient) -> None:
        client_with_db.post("/api/clip/tags", json={"term": "mountain lake"})
        resp = client_with_db.delete("/api/clip/tags/mountain lake")
        assert resp.status_code == 200
        data = resp.json()
        assert "mountain lake" not in data["tags"]

    def test_delete_nonexistent_tag_returns_404(self, client_with_db: TestClient) -> None:
        resp = client_with_db.delete("/api/clip/tags/doesnotexist")
        assert resp.status_code == 404

    def test_list_tags_after_add(self, client_with_db: TestClient) -> None:
        client_with_db.post("/api/clip/tags", json={"term": "autumn forest"})
        client_with_db.post("/api/clip/tags", json={"term": "misty morning"})
        resp = client_with_db.get("/api/clip/tags")
        assert resp.status_code == 200
        tags = resp.json()["tags"]
        assert "autumn forest" in tags
        assert "misty morning" in tags

    def test_clip_page_loads(self, client_with_db: TestClient) -> None:
        resp = client_with_db.get("/clip")
        assert resp.status_code == 200
        assert b"CLIP" in resp.content
        assert b"Custom Tag Terms" in resp.content

    def test_clip_page_redirects_without_db(self, client_no_db: TestClient) -> None:
        resp = client_no_db.get("/clip")
        assert resp.status_code in (302, 307)
