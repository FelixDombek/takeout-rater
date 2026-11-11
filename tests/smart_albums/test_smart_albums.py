"""Tests for smart album suggestions: clustering, naming, storage and UI."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from takeout_rater.db import smart_album_queries as sq
from takeout_rater.db.queries import upsert_asset, upsert_clip_embedding
from takeout_rater.db.schema import CURRENT_SCHEMA_VERSION, migrate
from takeout_rater.smart_albums.service import run_suggestions
from takeout_rater.smart_albums.suggest import (
    SuggestParams,
    effective_k,
    format_name,
    rank_terms,
    suggest_albums,
    unique_names,
)
from takeout_rater.ui.app import create_app

DIM = 768
_rng = np.random.default_rng(0)
# Three "topics" plus a direction shared by every photo (think "a photo").
TOPICS = {name: _rng.normal(size=DIM) for name in ("beach", "dog", "mountain")}
COMMON = _rng.normal(size=DIM)
GROUP_SIZES = {"beach": 40, "dog": 25, "mountain": 15}


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _photo_embeddings() -> tuple[list[int], np.ndarray, list[str]]:
    """Synthetic image embeddings: topic + strong common part + noise."""
    ids: list[int] = []
    rows: list[np.ndarray] = []
    labels: list[str] = []
    rng = np.random.default_rng(1)
    for topic, n in GROUP_SIZES.items():
        for _ in range(n):
            ids.append(len(ids) + 1)
            rows.append(_unit(TOPICS[topic] + 3.0 * COMMON + 0.3 * rng.normal(size=DIM)))
            labels.append(topic)
    return ids, np.vstack(rows).astype(np.float32), labels


VOCAB_TERMS = ["beach", "dog", "mountain", "photo", "sunset"]


def _fake_encoder(terms: list[str]) -> np.ndarray:
    """Text embeddings: topic terms point at their topic, "photo" at the common part."""
    vecs = []
    rng = np.random.default_rng(2)
    for t in terms:
        if t in TOPICS:
            v = TOPICS[t] + COMMON
        elif t == "photo":
            v = COMMON + 0.05 * rng.normal(size=DIM)  # matches every photo best
        else:
            v = rng.normal(size=DIM)
        vecs.append(_unit(v))
    return np.vstack(vecs).astype(np.float32)


# ── Pure logic ────────────────────────────────────────────────────────────────


def test_effective_k() -> None:
    assert effective_k(1, 40) == 1
    assert effective_k(10, 40) == 2
    assert effective_k(100, 40) == 5
    assert effective_k(100_000, 40) == 40


def test_names_are_title_cased_and_unique() -> None:
    assert format_name(["beach", "golden hour"]) == "Beach · Golden hour"
    assert format_name([]) == "Untitled album"
    ranked = [
        [("beach", 0.1), ("sunset", 0.05), ("surfing", 0.01)],
        [("beach", 0.1), ("sunset", 0.04), ("sand", 0.02)],
        [("beach", 0.1), ("sunset", 0.04)],
    ]
    assert unique_names(ranked, 2) == [
        "Beach · Sunset",
        "Beach · Sunset · Sand",
        "Beach · Sunset (2)",
    ]


def test_rank_terms_prefers_distinctive_over_generic() -> None:
    ids, mat, labels = _photo_embeddings()
    vocab = _fake_encoder(VOCAB_TERMS)
    centroids = np.vstack(
        [_unit(mat[[lbl == t for lbl in labels]].mean(axis=0)) for t in GROUP_SIZES]
    )
    # Raw similarity: "photo" is the best match for every cluster …
    raw = centroids @ vocab.T
    assert all(VOCAB_TERMS[int(i)] == "photo" for i in raw.argmax(axis=1))
    # … but the distinctive topic term wins the name.
    ranked = rank_terms(centroids, vocab, VOCAB_TERMS, max_terms=2)
    assert [r[0][0] for r in ranked] == list(GROUP_SIZES)


def test_rank_terms_skips_near_synonyms() -> None:
    centroid = _unit(TOPICS["beach"])[None, :]
    vocab = np.vstack(
        [_unit(TOPICS["beach"]), _unit(TOPICS["beach"] + 0.01), _unit(TOPICS["dog"])]
    ).astype(np.float32)
    ranked = rank_terms(centroid, vocab, ["beach", "beaches", "dog"], max_terms=2)
    assert [t for t, _ in ranked[0]] == ["beach", "dog"]


def test_lead_terms_put_subjects_first() -> None:
    centroid = _unit(TOPICS["beach"] + 0.5 * TOPICS["dog"])[None, :]
    vocab = _fake_encoder(["beach", "dog"])
    assert rank_terms(centroid, vocab, ["beach", "dog"])[0][0][0] == "beach"
    ranked = rank_terms(centroid, vocab, ["beach", "dog"], lead_terms={"dog"})
    assert [t for t, _ in ranked[0]] == ["dog", "beach"]


def test_suggest_albums_recovers_groups() -> None:
    ids, mat, labels = _photo_embeddings()
    vocab = _fake_encoder(VOCAB_TERMS)
    out = suggest_albums(
        ids, mat, vocab, VOCAB_TERMS, SuggestParams(n_albums=3, min_photos=5, terms_per_name=1)
    )
    assert [s.name for s in out] == ["Beach", "Dog", "Mountain"]  # biggest first
    by_id = dict(zip(ids, labels, strict=True))
    for s in out:
        assert {by_id[a] for a in s.asset_ids} == {s.name.lower()}
        assert s.similarities == sorted(s.similarities, reverse=True)
        assert 0 < s.cohesion <= 1


def test_min_photos_drops_small_clusters() -> None:
    ids, mat, _ = _photo_embeddings()
    out = suggest_albums(
        ids, mat, _fake_encoder(VOCAB_TERMS), VOCAB_TERMS, SuggestParams(n_albums=3, min_photos=20)
    )
    assert len(out) == 2


# ── Database + service ────────────────────────────────────────────────────────


def _populate(conn: sqlite3.Connection) -> None:
    ids, mat, _ = _photo_embeddings()
    for aid, vec in zip(ids, mat, strict=True):
        asset_id = upsert_asset(
            conn,
            {
                "relpath": f"Photos/img{aid}.jpg",
                "filename": f"img{aid}.jpg",
                "ext": ".jpg",
                "mime": "image/jpeg",
                "taken_at": 1_700_000_000 + aid * 60,
                "indexed_at": int(time.time()),
            },
        )
        upsert_clip_embedding(conn, asset_id, vec.astype(np.float32).tobytes())
    conn.commit()


def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    return conn


_PARAMS = SuggestParams(n_albums=3, min_photos=5, terms_per_name=1)


def test_migration_from_v18_adds_tables() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript("PRAGMA user_version = 18;")
    migrate(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == CURRENT_SCHEMA_VERSION
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"smart_albums", "smart_album_assets"} <= tables


def test_run_suggestions_stores_albums_and_keeps_kept_ones() -> None:
    conn = _make_db()
    _populate(conn)
    summary = run_suggestions(conn, _PARAMS, encode_texts=_fake_encoder)
    assert summary["n_albums"] == 3
    albums = sq.list_smart_albums(conn)
    assert [a["name"] for a in albums] == ["Beach", "Dog", "Mountain"]
    assert [a["n_assets"] for a in albums] == [40, 25, 15]
    assert albums[0]["terms"][0]["term"] == "beach"
    assert len(albums[0]["preview_ids"]) == 3
    assert albums[0]["cover_asset_id"] == albums[0]["preview_ids"][0]

    dog = albums[1]
    assert sq.rename_smart_album(conn, dog["id"], "  Rex  ")
    dog = sq.get_smart_album(conn, dog["id"])
    assert dog["name"] == "Rex" and dog["kept"]  # renaming keeps it

    summary = run_suggestions(conn, _PARAMS, encode_texts=_fake_encoder)
    assert summary["n_skipped_duplicates"] == 1  # dog cluster is not re-suggested
    names = [a["name"] for a in sq.list_smart_albums(conn)]
    assert names == ["Rex", "Beach", "Mountain"]
    member = sq.get_smart_album_assets(conn, dog["id"], limit=1, offset=0)[0]["id"]
    assert [a["id"] for a in sq.get_smart_albums_for_asset(conn, member)] == [dog["id"]]


def test_album_ids_are_not_reused_across_runs() -> None:
    conn = _make_db()
    _populate(conn)
    run_suggestions(conn, _PARAMS, encode_texts=_fake_encoder)
    first = {a["id"] for a in sq.list_smart_albums(conn)}
    run_suggestions(conn, _PARAMS, encode_texts=_fake_encoder)
    assert first.isdisjoint(a["id"] for a in sq.list_smart_albums(conn))


def test_run_suggestions_without_embeddings_explains() -> None:
    conn = _make_db()
    with pytest.raises(ValueError, match="No CLIP embeddings"):
        run_suggestions(conn, _PARAMS, encode_texts=_fake_encoder)


# ── Web UI ────────────────────────────────────────────────────────────────────


@pytest.fixture()
def client(tmp_path: Path) -> tuple[TestClient, sqlite3.Connection]:
    from takeout_rater.db.connection import open_library_db

    conn = open_library_db(tmp_path)  # file DB: the job opens its own connection
    _populate(conn)
    app = create_app(tmp_path, conn, db_root=tmp_path)
    app.state.smart_albums_text_encoder = _fake_encoder
    return TestClient(app, follow_redirects=True), conn


@pytest.fixture(autouse=True)
def _small_vocab(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "takeout_rater.smart_albums.service.vocabulary",
        lambda conn: (list(VOCAB_TERMS), set(VOCAB_TERMS)),
    )


def _suggest(c: TestClient, body: dict | None = None) -> dict:
    resp = c.post(
        "/api/jobs/smart_albums/start",
        json=body or {"n_albums": 3, "min_photos": 5, "terms_per_name": 1},
    )
    assert resp.status_code == 200, resp.text
    for _ in range(200):
        status = c.get("/api/jobs/status?job_type=smart_albums").json()
        if status["done"]:
            return status
        time.sleep(0.05)
    raise AssertionError("smart_albums job did not finish")


def test_smart_albums_page_flow(client: tuple[TestClient, sqlite3.Connection]) -> None:
    c, _ = client
    assert 'href="/smart-albums"' in c.get("/albums").text  # reachable from Albums
    resp = c.get("/smart-albums")
    assert resp.status_code == 200
    assert "No smart albums yet" in resp.text

    status = _suggest(c)
    assert status["error"] is None, status
    assert status["diagnostics"]["n_albums"] == 3

    resp = c.get("/smart-albums")
    assert "Suggestions" in resp.text
    assert "Beach" in resp.text and "Mountain" in resp.text

    items = c.get("/api/smart-albums").json()["items"]
    beach = items[0]
    resp = c.get(f"/smart-albums/{beach['id']}")
    assert resp.status_code == 200
    assert "40 photos" in resp.text
    assert resp.text.count("data-lb-id=") == 40
    assert c.get(f"/smart-albums/{beach['id']}?sort=date").status_code == 200

    assert c.post(f"/api/smart-albums/{beach['id']}/keep", json={"kept": True}).json()["kept"]
    assert "★ Kept" in c.get("/smart-albums").text
    resp = c.post(f"/api/smart-albums/{beach['id']}/rename", json={"name": "Holidays"})
    assert resp.json()["name"] == "Holidays"
    resp = c.post(f"/api/smart-albums/{beach['id']}/rename", json={"name": ""})
    assert resp.json()["name"] == "Beach"

    dog = items[1]
    assert c.delete(f"/api/smart-albums/{dog['id']}").status_code == 200
    assert c.get(f"/smart-albums/{dog['id']}").status_code == 404
    assert c.delete(f"/api/smart-albums/{dog['id']}").status_code == 404
    assert c.post("/api/smart-albums/9999/keep", json={"kept": True}).status_code == 404
