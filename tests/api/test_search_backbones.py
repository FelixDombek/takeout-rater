"""Tests for model-keyed embeddings and multi-backbone (SigLIP 2 / CLIP) search.

Covers:
- blobs_to_matrix / load_embedding_matrix (np.frombuffer path)
- image_embeddings queries and the v14 → v15 migration
- /api/search backbone resolution, ranking, pagination and cache invalidation
- /api/search/backbones
- embed job for a non-CLIP backbone (with a fake encoder)
"""

from __future__ import annotations

import struct
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from takeout_rater.db.queries import (
    CLIP_L14_MODEL_KEY,
    blobs_to_matrix,
    bulk_upsert_clip_embeddings,
    bulk_upsert_image_embeddings,
    count_image_embeddings,
    list_asset_ids_without_image_embedding,
    load_embedding_matrix,
)
from takeout_rater.embeddings import search_backbones
from takeout_rater.embeddings.search_backbones import SIGLIP2_MODEL_KEY
from takeout_rater.ui.app import create_app

_CLIP_DIM = 768
_SIGLIP_DIM = 1152


def _unit(vec: np.ndarray) -> np.ndarray:
    return (vec / np.linalg.norm(vec)).astype(np.float32)


def _blob(vec: np.ndarray) -> bytes:
    return vec.astype("<f4").tobytes()


def _axis(dim: int, i: int, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """Unit vector pointing mostly along axis *i*."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32) * noise
    v[i] += 1.0
    return _unit(v)


def _insert_asset(conn, relpath: str) -> int:
    cur = conn.execute(
        "INSERT INTO assets (relpath, filename, ext, indexed_at) VALUES (?, ?, ?, ?)",
        (relpath, relpath, ".jpg", int(time.time())),
    )
    conn.commit()
    return cur.lastrowid


def _set_available(monkeypatch: pytest.MonkeyPatch, key: str, value: bool) -> None:
    import dataclasses

    entry = dataclasses.replace(search_backbones.BACKBONES[key], available=lambda: value)
    monkeypatch.setitem(search_backbones.BACKBONES, key, entry)


@pytest.fixture(autouse=True)
def _all_backbones_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Don't depend on the installed open_clip version (CI may predate SigLIP 2)."""
    for key in list(search_backbones.BACKBONES):
        _set_available(monkeypatch, key, True)


@pytest.fixture()
def db_conn(tmp_path: Path):
    from takeout_rater.db.connection import open_library_db

    return open_library_db(tmp_path)


@pytest.fixture()
def client(tmp_path: Path, db_conn) -> TestClient:
    app = create_app(tmp_path, db_conn, db_root=tmp_path)
    return TestClient(app, follow_redirects=False)


# ---------------------------------------------------------------------------
# Matrix loading
# ---------------------------------------------------------------------------


class TestBlobsToMatrix:
    def test_matches_struct_unpack(self) -> None:
        rng = np.random.RandomState(1)
        vecs = [rng.randn(_CLIP_DIM).astype(np.float32) for _ in range(5)]
        blobs = [struct.pack(f"{_CLIP_DIM}f", *v) for v in vecs]
        matrix = blobs_to_matrix(blobs)
        assert matrix.shape == (5, _CLIP_DIM)
        assert matrix.dtype == np.float32
        assert matrix.flags.writeable
        np.testing.assert_array_equal(matrix, np.stack(vecs))

    def test_empty(self) -> None:
        assert blobs_to_matrix([]).shape[0] == 0

    def test_inconsistent_lengths_raise(self) -> None:
        with pytest.raises(ValueError):
            blobs_to_matrix([b"\0" * 16, b"\0" * 12])

    def test_dim_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            blobs_to_matrix([b"\0" * 16], dim=8)


class TestImageEmbeddingQueries:
    def test_models_are_isolated(self, db_conn) -> None:
        a1 = _insert_asset(db_conn, "a.jpg")
        a2 = _insert_asset(db_conn, "b.jpg")
        bulk_upsert_clip_embeddings(db_conn, [(a1, _blob(_axis(_CLIP_DIM, 0)))])
        bulk_upsert_image_embeddings(
            db_conn,
            SIGLIP2_MODEL_KEY,
            [(a2, _blob(_axis(_SIGLIP_DIM, 1))), (a1, _blob(_axis(_SIGLIP_DIM, 2)))],
        )

        assert count_image_embeddings(db_conn, CLIP_L14_MODEL_KEY) == 1
        assert count_image_embeddings(db_conn, SIGLIP2_MODEL_KEY) == 2
        assert count_image_embeddings(db_conn, "other-model") == 0

        ids, matrix = load_embedding_matrix(db_conn, SIGLIP2_MODEL_KEY)
        assert ids.tolist() == [a1, a2]  # ordered by asset id
        assert matrix.shape == (2, _SIGLIP_DIM)
        assert matrix[0, 2] == pytest.approx(1.0)

        ids, matrix = load_embedding_matrix(db_conn, CLIP_L14_MODEL_KEY)
        assert ids.tolist() == [a1]
        assert matrix.shape == (1, _CLIP_DIM)

        assert list_asset_ids_without_image_embedding(db_conn, CLIP_L14_MODEL_KEY) == [a2]
        assert list_asset_ids_without_image_embedding(db_conn, SIGLIP2_MODEL_KEY) == []

    def test_upsert_replaces_and_records_dim(self, db_conn) -> None:
        a1 = _insert_asset(db_conn, "a.jpg")
        bulk_upsert_image_embeddings(db_conn, SIGLIP2_MODEL_KEY, [(a1, _blob(np.ones(4)))])
        bulk_upsert_image_embeddings(db_conn, SIGLIP2_MODEL_KEY, [(a1, _blob(np.zeros(4)))])
        row = db_conn.execute("SELECT COUNT(*), MAX(dim) FROM image_embeddings").fetchone()
        assert tuple(row) == (1, 4)

    def test_upsert_rejects_mixed_dims(self, db_conn) -> None:
        a1 = _insert_asset(db_conn, "a.jpg")
        a2 = _insert_asset(db_conn, "b.jpg")
        with pytest.raises(ValueError):
            bulk_upsert_image_embeddings(
                db_conn, SIGLIP2_MODEL_KEY, [(a1, _blob(np.ones(4))), (a2, _blob(np.ones(8)))]
            )
        bulk_upsert_image_embeddings(db_conn, SIGLIP2_MODEL_KEY, [(a1, _blob(np.ones(4)))])
        with pytest.raises(ValueError):
            bulk_upsert_image_embeddings(db_conn, SIGLIP2_MODEL_KEY, [(a2, _blob(np.ones(8)))])

    def test_load_skips_bad_rows(self, db_conn) -> None:
        a1 = _insert_asset(db_conn, "a.jpg")
        a2 = _insert_asset(db_conn, "b.jpg")
        a3 = _insert_asset(db_conn, "c.jpg")
        bulk_upsert_image_embeddings(
            db_conn, SIGLIP2_MODEL_KEY, [(a1, _blob(np.ones(4))), (a3, _blob(np.ones(4)))]
        )
        db_conn.execute(
            "INSERT INTO image_embeddings VALUES (?, ?, 3, ?, 0)",
            (a2, SIGLIP2_MODEL_KEY, b"\0" * 12),
        )
        db_conn.commit()
        ids, matrix = load_embedding_matrix(db_conn, SIGLIP2_MODEL_KEY)
        assert ids.tolist() == [a1, a3]
        assert matrix.shape == (2, 4)

    def test_clip_key_rejected_for_image_embeddings(self, db_conn) -> None:
        with pytest.raises(ValueError):
            bulk_upsert_image_embeddings(db_conn, CLIP_L14_MODEL_KEY, [(1, b"\0" * 4)])


class TestMigration:
    def test_v14_database_gains_image_embeddings(self, tmp_path: Path) -> None:
        from takeout_rater.db.connection import open_library_db
        from takeout_rater.db.schema import CURRENT_SCHEMA_VERSION, migrate

        conn = open_library_db(tmp_path)
        conn.execute("DROP INDEX idx_image_embeddings_asset_id")
        conn.execute("DROP TABLE image_embeddings")
        conn.execute("PRAGMA user_version = 14")
        conn.commit()

        migrate(conn)

        assert conn.execute("PRAGMA user_version").fetchone()[0] == CURRENT_SCHEMA_VERSION
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "image_embeddings" in tables


# ---------------------------------------------------------------------------
# Search API
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_text_encoder(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Encode any query as 'axis 0' in the backbone's dimension; record calls."""
    calls: list[tuple[str, str]] = []

    def _encode(key: str, text: str) -> np.ndarray:
        calls.append((key, text))
        return _axis(search_backbones.get_backbone(key).dim, 0)

    monkeypatch.setattr(search_backbones, "encode_text", _encode)
    return calls


def _seed_two_models(db_conn) -> tuple[int, int, int]:
    """Three assets; SigLIP ranks a3 first, CLIP ranks a1 first."""
    a1 = _insert_asset(db_conn, "a.jpg")
    a2 = _insert_asset(db_conn, "b.jpg")
    a3 = _insert_asset(db_conn, "c.jpg")
    bulk_upsert_clip_embeddings(
        db_conn,
        [
            (a1, _blob(_axis(_CLIP_DIM, 0, 0.1, 1))),
            (a2, _blob(_axis(_CLIP_DIM, 5, 0.1, 2))),
            (a3, _blob(_axis(_CLIP_DIM, 0, 0.6, 3))),
        ],
    )
    bulk_upsert_image_embeddings(
        db_conn,
        SIGLIP2_MODEL_KEY,
        [
            (a1, _blob(_axis(_SIGLIP_DIM, 0, 0.6, 4))),
            (a2, _blob(_axis(_SIGLIP_DIM, 7, 0.1, 5))),
            (a3, _blob(_axis(_SIGLIP_DIM, 0, 0.1, 6))),
        ],
    )
    return a1, a2, a3


class TestSearchBackends:
    def test_auto_falls_back_to_clip(self, client, db_conn, fake_text_encoder) -> None:
        a1 = _insert_asset(db_conn, "a.jpg")
        bulk_upsert_clip_embeddings(db_conn, [(a1, _blob(_axis(_CLIP_DIM, 0)))])
        data = client.get("/api/search?q=dog").json()
        assert data["model"] == CLIP_L14_MODEL_KEY
        assert data["results"][0]["asset_id"] == a1
        assert fake_text_encoder == [(CLIP_L14_MODEL_KEY, "dog")]

    def test_auto_prefers_siglip(self, client, db_conn, fake_text_encoder) -> None:
        a1, a2, a3 = _seed_two_models(db_conn)
        data = client.get("/api/search?q=dog").json()
        assert data["model"] == SIGLIP2_MODEL_KEY
        assert [r["asset_id"] for r in data["results"]] == [a3, a1, a2]
        assert data["total"] == 3

    def test_explicit_clip(self, client, db_conn, fake_text_encoder) -> None:
        a1, a2, a3 = _seed_two_models(db_conn)
        data = client.get(f"/api/search?q=dog&model={CLIP_L14_MODEL_KEY}").json()
        assert data["model"] == CLIP_L14_MODEL_KEY
        assert [r["asset_id"] for r in data["results"]] == [a1, a3, a2]

    def test_explicit_model_without_embeddings(self, client, db_conn, fake_text_encoder) -> None:
        a1 = _insert_asset(db_conn, "a.jpg")
        bulk_upsert_clip_embeddings(db_conn, [(a1, _blob(_axis(_CLIP_DIM, 0)))])
        data = client.get(f"/api/search?q=dog&model={SIGLIP2_MODEL_KEY}").json()
        assert data["results"] == []
        assert data["model"] == SIGLIP2_MODEL_KEY
        assert "error" in data
        assert fake_text_encoder == []  # no model load when there is nothing to search

    def test_pagination(self, client, db_conn, fake_text_encoder) -> None:
        a1, a2, a3 = _seed_two_models(db_conn)
        p1 = client.get("/api/search?q=dog&limit=2&page=1").json()
        p2 = client.get("/api/search?q=dog&limit=2&page=2").json()
        p3 = client.get("/api/search?q=dog&limit=2&page=3").json()
        assert [r["asset_id"] for r in p1["results"]] == [a3, a1]
        assert [r["asset_id"] for r in p2["results"]] == [a2]
        assert p3["results"] == []

    def test_partial_reports_model(self, client, db_conn, fake_text_encoder) -> None:
        _seed_two_models(db_conn)
        resp = client.get("/api/search?q=dog&partial=1")
        assert resp.status_code == 200
        assert f'data-model="{SIGLIP2_MODEL_KEY}"'.encode() in resp.content

    def test_siglip_index_refreshes_when_rows_change(
        self, client, db_conn, fake_text_encoder
    ) -> None:
        _seed_two_models(db_conn)
        assert client.get("/api/search?q=dog").json()["total"] == 3

        # Rows written by a running (or cancelled) embed job are picked up
        # without explicit invalidation.
        a4 = _insert_asset(db_conn, "d.jpg")
        bulk_upsert_image_embeddings(
            db_conn, SIGLIP2_MODEL_KEY, [(a4, _blob(_axis(_SIGLIP_DIM, 0)))]
        )
        data = client.get(f"/api/search?q=dog&model={SIGLIP2_MODEL_KEY}").json()
        assert data["total"] == 4
        assert data["results"][0]["asset_id"] == a4

    def test_clip_index_is_cached_until_invalidated(
        self, client, db_conn, fake_text_encoder
    ) -> None:
        from takeout_rater.api.search import invalidate_search_indexes

        a1 = _insert_asset(db_conn, "a.jpg")
        bulk_upsert_clip_embeddings(db_conn, [(a1, _blob(_axis(_CLIP_DIM, 0)))])
        assert client.get("/api/search?q=dog").json()["total"] == 1
        a2 = _insert_asset(db_conn, "b.jpg")
        bulk_upsert_clip_embeddings(db_conn, [(a2, _blob(_axis(_CLIP_DIM, 1)))])
        assert client.get("/api/search?q=dog").json()["total"] == 1  # cached
        invalidate_search_indexes(client.app.state, CLIP_L14_MODEL_KEY)
        assert client.get("/api/search?q=dog").json()["total"] == 2

    def test_auto_ignores_partial_siglip(self, client, db_conn, fake_text_encoder) -> None:
        ids = [_insert_asset(db_conn, f"{i}.jpg") for i in range(20)]
        bulk_upsert_clip_embeddings(db_conn, [(a, _blob(_axis(_CLIP_DIM, 0))) for a in ids])
        bulk_upsert_image_embeddings(
            db_conn, SIGLIP2_MODEL_KEY, [(a, _blob(_axis(_SIGLIP_DIM, 0))) for a in ids[:18]]
        )
        assert client.get("/api/search?q=dog").json()["model"] == CLIP_L14_MODEL_KEY
        bulk_upsert_image_embeddings(
            db_conn, SIGLIP2_MODEL_KEY, [(ids[18], _blob(_axis(_SIGLIP_DIM, 0)))]
        )
        assert client.get("/api/search?q=dog").json()["model"] == SIGLIP2_MODEL_KEY  # 95%

    def test_unavailable_siglip_falls_back(
        self, client, db_conn, fake_text_encoder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed_two_models(db_conn)
        _set_available(monkeypatch, SIGLIP2_MODEL_KEY, False)
        assert client.get("/api/search?q=dog").json()["model"] == CLIP_L14_MODEL_KEY
        resp = client.get(f"/api/search?q=dog&model={SIGLIP2_MODEL_KEY}")
        assert resp.status_code == 503
        assert "open-clip-torch" in resp.json()["detail"]

    def test_backbones_endpoint(self, client, db_conn) -> None:
        a1 = _insert_asset(db_conn, "a.jpg")
        _insert_asset(db_conn, "b.jpg")
        bulk_upsert_clip_embeddings(db_conn, [(a1, _blob(_axis(_CLIP_DIM, 0)))])
        data = client.get("/api/search/backbones").json()
        assert data["total_assets"] == 2
        assert data["auto"] == CLIP_L14_MODEL_KEY
        by_key = {b["key"]: b for b in data["backbones"]}
        assert by_key[CLIP_L14_MODEL_KEY]["count"] == 1
        assert by_key[SIGLIP2_MODEL_KEY]["count"] == 0
        assert by_key[SIGLIP2_MODEL_KEY]["dim"] == _SIGLIP_DIM

    def test_search_page_has_model_selector(self, client) -> None:
        resp = client.get("/search")
        assert resp.status_code == 200
        assert b'id="search-model"' in resp.content
        assert b"btn-siglip-compute" in resp.content


def test_resolve_backbone_key() -> None:
    resolve = search_backbones.resolve_backbone_key
    clip, sig = CLIP_L14_MODEL_KEY, SIGLIP2_MODEL_KEY
    assert resolve("auto", {}) == clip
    assert resolve("auto", {sig: 0, clip: 5}) == clip
    assert resolve("auto", {sig: 1}) == sig  # no CLIP embeddings at all
    assert resolve("auto", {sig: 94, clip: 100}) == clip
    assert resolve("auto", {sig: 95, clip: 100}) == sig
    assert resolve("auto", {sig: 100, clip: 100}, {sig: False, clip: True}) == clip
    assert resolve(clip, {sig: 100}) == clip


# ---------------------------------------------------------------------------
# Embed job
# ---------------------------------------------------------------------------


class _SyncThread(threading.Thread):
    def start(self) -> None:  # type: ignore[override]
        self.run()


class TestEmbedJobForBackbone:
    def test_unknown_model_400(self, client) -> None:
        resp = client.post("/api/jobs/embed/start", json={"model": "nope"})
        assert resp.status_code == 400

    def test_siglip_job_fills_image_embeddings(
        self, client, db_conn, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from PIL import Image

        from takeout_rater.api.jobs import _configured_thumbs_dir
        from takeout_rater.indexing.thumbnailer import thumb_path_for_id

        a1 = _insert_asset(db_conn, "a.jpg")
        a2 = _insert_asset(db_conn, "b.jpg")  # no thumbnail → skipped
        thumbs = _configured_thumbs_dir(client.app)
        thumb = thumb_path_for_id(thumbs, a1)
        thumb.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 48), (200, 10, 10)).save(thumb)

        loaded: list[bool] = []
        fake = search_backbones.SearchBackbone(
            key=SIGLIP2_MODEL_KEY,
            label="SigLIP 2 (fake)",
            description="",
            dim=_SIGLIP_DIM,
            loader=lambda: loaded.append(True),  # type: ignore[arg-type,return-value]
            available=lambda: True,
        )
        monkeypatch.setitem(search_backbones.BACKBONES, SIGLIP2_MODEL_KEY, fake)
        monkeypatch.setattr(
            search_backbones,
            "encode_images",
            lambda key, images: np.stack([_axis(_SIGLIP_DIM, 3) for _ in images]),
        )
        monkeypatch.setattr("takeout_rater.api.jobs.threading.Thread", _SyncThread)

        resp = client.post("/api/jobs/embed/start", json={"model": SIGLIP2_MODEL_KEY})
        assert resp.status_code == 200

        status = client.get("/api/jobs/status?job_type=embed").json()
        assert status["error"] is None, status
        assert status["done"] is True
        assert status["diagnostics"]["model"] == SIGLIP2_MODEL_KEY
        assert loaded == [True]

        ids, matrix = load_embedding_matrix(db_conn, SIGLIP2_MODEL_KEY)
        assert ids.tolist() == [a1]
        assert matrix[0, 3] == pytest.approx(1.0)
        assert count_image_embeddings(db_conn, CLIP_L14_MODEL_KEY) == 0
        assert a2 in list_asset_ids_without_image_embedding(db_conn, SIGLIP2_MODEL_KEY)

    def test_failed_clip_job_still_invalidates_cache(
        self, client, db_conn, fake_text_encoder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from PIL import Image

        from takeout_rater.api.jobs import _configured_thumbs_dir
        from takeout_rater.indexing.thumbnailer import thumb_path_for_id

        a1 = _insert_asset(db_conn, "a.jpg")
        a2 = _insert_asset(db_conn, "b.jpg")
        bulk_upsert_clip_embeddings(db_conn, [(a1, _blob(_axis(_CLIP_DIM, 0)))])
        thumb = thumb_path_for_id(_configured_thumbs_dir(client.app), a2)
        thumb.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32)).save(thumb)
        client.get("/api/search?q=dog")
        assert CLIP_L14_MODEL_KEY in client.app.state.search_indexes

        def _boom(key, images):
            raise RuntimeError("GPU fell over")

        fake = search_backbones.SearchBackbone(
            key=CLIP_L14_MODEL_KEY,
            label="CLIP (fake)",
            description="",
            dim=_CLIP_DIM,
            loader=lambda: None,  # type: ignore[arg-type,return-value]
            available=lambda: True,
        )
        monkeypatch.setitem(search_backbones.BACKBONES, CLIP_L14_MODEL_KEY, fake)
        monkeypatch.setattr(search_backbones, "encode_images", _boom)
        monkeypatch.setattr("takeout_rater.api.jobs.threading.Thread", _SyncThread)
        assert client.post("/api/jobs/embed/start", json={}).status_code == 200
        assert "GPU fell over" in client.get("/api/jobs/status?job_type=embed").json()["error"]
        assert CLIP_L14_MODEL_KEY not in client.app.state.search_indexes

    def test_cancel_without_job_404(self, client) -> None:
        assert client.post("/api/jobs/embed/cancel").status_code == 404


def test_clip_l14_loader_uses_importer_preprocessing(monkeypatch) -> None:
    """clip_embeddings rows from the embed job must match the indexer's preprocessing."""
    from takeout_rater.embeddings import search_backbones
    from takeout_rater.scoring.scorers import clip_backbone

    monkeypatch.setattr(clip_backbone, "get_clip_model", lambda: ("m", "open_clip", "t", "d"))
    model, preprocess, tokenizer, device = search_backbones._load_clip()
    assert (model, tokenizer, device) == ("m", "t", "d")
    assert preprocess is clip_backbone.preprocess_clip_image_fast
