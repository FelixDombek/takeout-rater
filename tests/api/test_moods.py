"""Tests for the mood timeline (issue #110): numpy core and API routes."""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from takeout_rater.db.queries import (
    CLIP_L14_MODEL_KEY,
    bulk_upsert_clip_embeddings,
    load_dated_asset_ids,
)
from takeout_rater.embeddings import search_backbones
from takeout_rater.moods import timeline as mt
from takeout_rater.ui.app import create_app

_DIM = 768


def _ts(y: int, m: int, d: int = 15) -> int:
    return int(dt.datetime(y, m, d, 12, tzinfo=dt.UTC).timestamp())


def _axis(i: int, noise: float = 0.05, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(_DIM).astype(np.float32) * noise
    v[i] += 1.0
    return (v / np.linalg.norm(v)).astype(np.float32)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


class TestNormalizeMoods:
    def test_cleans_and_dedupes(self) -> None:
        assert mt.normalize_moods([" happy ", "Happy", "", "very   calm"]) == ["happy", "very calm"]

    def test_caps_count_and_length(self) -> None:
        moods = mt.normalize_moods([f"m{i}" for i in range(20)] + ["x" * 100])
        assert len(moods) == mt.MAX_MOODS
        assert all(len(m) <= mt.MAX_MOOD_LENGTH for m in mt.normalize_moods(["x" * 100]))


class TestTextMatrix:
    def test_prompt_ensemble_is_normalised(self) -> None:
        seen: list[str] = []

        def encode(prompt: str) -> np.ndarray:
            seen.append(prompt)
            return _axis(0 if "happy" in prompt else 1, seed=len(seen))

        matrix = mt.mood_text_matrix(["happy", "sad"], encode)
        assert matrix.shape == (2, _DIM)
        np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0, rtol=1e-5)
        assert len(seen) == 2 * len(mt.PROMPT_TEMPLATES)
        assert "a photo that feels happy." in seen


class TestProbabilities:
    def test_rows_sum_to_one_and_pick_closest(self) -> None:
        text = np.stack([_axis(0, 0), _axis(1, 0)])
        emb = np.stack([_axis(0, seed=1), _axis(1, seed=2)])
        probs = mt.mood_probabilities(emb, text)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)
        assert probs[0, 0] > 0.9 and probs[1, 1] > 0.9


class TestBuckets:
    def test_month_buckets_fill_gaps(self) -> None:
        b = mt.assign_buckets(np.array([_ts(2020, 11), _ts(2021, 2)]), "month")
        assert b.keys == ["2020-11", "2020-12", "2021-01", "2021-02"]
        assert b.index.tolist() == [0, 3]
        assert b.starts[1] == _ts(2020, 12, 1) - 12 * 3600
        assert b.ends[1] == b.starts[2]

    def test_quarter_and_year(self) -> None:
        ts = np.array([_ts(2019, 12), _ts(2020, 4)])
        assert mt.assign_buckets(ts, "quarter").keys == ["2019-Q4", "2020-Q1", "2020-Q2"]
        assert mt.assign_buckets(ts, "year").keys == ["2019", "2020"]

    def test_weeks_start_on_monday(self) -> None:
        # 2024-01-03 is a Wednesday; its week starts Monday 2024-01-01.
        b = mt.assign_buckets(np.array([_ts(2024, 1, 3)]), "week")
        assert b.keys == ["2024-01-01"]
        assert b.ends[0] - b.starts[0] == 7 * 86400

    def test_pre_1970_dates(self) -> None:
        b = mt.assign_buckets(np.array([_ts(1965, 6)]), "month")
        assert b.keys == ["1965-06"]
        b = mt.assign_buckets(np.array([_ts(1965, 6)]), "week")
        assert dt.date.fromisoformat(b.keys[0]).weekday() == 0

    def test_unknown_granularity(self) -> None:
        with pytest.raises(ValueError):
            mt.assign_buckets(np.array([0]), "decade")


class TestTimeline:
    def test_share_and_lift(self) -> None:
        probs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        ts = np.array([_ts(2020, 1), _ts(2020, 1), _ts(2020, 3), _ts(2020, 3)])
        out = mt.mood_timeline(probs, ts, ["a", "b"], "month")
        assert out["overall"] == {"count": 4, "share": [0.5, 0.5]}
        jan, feb, mar = out["buckets"]
        assert jan["share"] == [1.0, 0.0] and jan["lift"] == [2.0, 0.0]
        assert feb["count"] == 0 and feb["share"] == [None, None]
        assert mar["share"] == [0.0, 1.0]

    def test_empty(self) -> None:
        out = mt.mood_timeline(np.zeros((0, 2), np.float32), np.zeros(0, np.int64), ["a", "b"])
        assert out["buckets"] == [] and out["overall"]["count"] == 0


class TestTopPhotos:
    def test_range_and_order(self) -> None:
        probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.95, 0.05]], dtype=np.float32)
        ids = np.array([10, 11, 12, 13])
        ts = np.array([100, 200, 300, 400])
        top = mt.top_photos(probs, ids, ts, 0, limit=2)
        assert [p["asset_id"] for p in top] == [13, 10]
        top = mt.top_photos(probs, ids, ts, 0, start=200, end=400)
        assert [p["asset_id"] for p in top] == [12, 11]


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_conn(tmp_path: Path):
    from takeout_rater.db.connection import open_library_db

    return open_library_db(tmp_path)


def _insert(conn, relpath: str, taken_at: int | None, sha: str | None = None, trashed=0) -> int:
    cur = conn.execute(
        "INSERT INTO assets (relpath, filename, ext, indexed_at, taken_at, sha256, trashed)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (relpath, relpath, ".jpg", int(time.time()), taken_at, sha, trashed),
    )
    conn.commit()
    return cur.lastrowid


@pytest.fixture()
def library(db_conn):
    """Happy photos in 2020, calm photos in 2021, plus excluded rows."""
    rows = []
    happy = [_insert(db_conn, f"h{i}.jpg", _ts(2020, 1 + i)) for i in range(3)]
    calm = [_insert(db_conn, f"c{i}.jpg", _ts(2021, 1 + i)) for i in range(3)]
    undated = _insert(db_conn, "u.jpg", None)
    trashed = _insert(db_conn, "t.jpg", _ts(2020, 6), trashed=1)
    dup_a = _insert(db_conn, "d1.jpg", _ts(2020, 6), sha="abc")
    dup_b = _insert(db_conn, "d2.jpg", _ts(2020, 6), sha="abc")
    for i, aid in enumerate(happy + [dup_a, dup_b, trashed]):
        rows.append((aid, _axis(0, seed=i).astype("<f4").tobytes()))
    for i, aid in enumerate(calm + [undated]):
        rows.append((aid, _axis(1, seed=10 + i).astype("<f4").tobytes()))
    bulk_upsert_clip_embeddings(db_conn, rows, model_id="test")
    return {"happy": happy, "calm": calm, "dup": (dup_a, dup_b)}


@pytest.fixture()
def fake_text(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def encode(key: str, text: str) -> np.ndarray:
        calls.append(text)
        if "happy" in text:
            return _axis(0, 0)
        if "calm" in text:
            return _axis(1, 0)
        return _axis(5, 0)

    monkeypatch.setattr(search_backbones, "encode_text", encode)
    from takeout_rater.api import moods as moods_api

    moods_api._text_cache.clear()
    return calls


@pytest.fixture()
def client(tmp_path: Path, db_conn) -> TestClient:
    return TestClient(create_app(tmp_path, db_conn, db_root=tmp_path), follow_redirects=False)


def test_load_dated_asset_ids_skips_undated_trashed_and_duplicates(db_conn, library) -> None:
    ids, ts = load_dated_asset_ids(db_conn)
    assert len(ids) == len(ts) == 7  # 3 happy + 3 calm + one of the two duplicates
    assert library["dup"][0] in ids and library["dup"][1] not in ids
    assert list(ids) == sorted(ids)


def test_timeline_api(client, library, fake_text) -> None:
    r = client.get("/api/moods/timeline", params={"moods": "happy,calm", "granularity": "year"})
    assert r.status_code == 200
    d = r.json()
    assert d["model"] == CLIP_L14_MODEL_KEY
    assert d["moods"] == ["happy", "calm"]
    assert d["overall"]["count"] == 7
    y2020, y2021 = d["buckets"]
    assert y2020["key"] == "2020" and y2020["count"] == 4
    assert y2020["share"][0] > 0.9 and y2021["share"][1] > 0.9
    assert y2020["lift"][0] > 1.5

    # Second call is served from the cache (no new text encodes).
    n_calls = len(fake_text)
    assert client.get("/api/moods/timeline", params={"moods": "happy,calm"}).status_code == 200
    assert len(fake_text) == n_calls


def test_timeline_validation(client, library, fake_text) -> None:
    assert client.get("/api/moods/timeline", params={"moods": "happy"}).status_code == 400
    r = client.get("/api/moods/timeline", params={"moods": "a,b", "granularity": "decade"})
    assert r.status_code == 400


def test_timeline_without_embeddings(client, fake_text) -> None:
    d = client.get("/api/moods/timeline").json()
    assert d["error"] and d["buckets"] == []
    assert d["moods"] == list(mt.DEFAULT_MOODS)


def test_photos_api(client, library, fake_text) -> None:
    params = {"moods": "happy,calm", "mood": "Calm", "limit": 2}
    d = client.get("/api/moods/photos", params=params).json()
    assert d["mood"] == "calm"
    assert len(d["photos"]) == 2
    assert all(p["asset_id"] in library["calm"] for p in d["photos"])

    params.update(start=_ts(2020, 1, 1), end=_ts(2021, 1, 1), mood="happy", limit=10)
    d = client.get("/api/moods/photos", params=params).json()
    assert {p["asset_id"] for p in d["photos"]} == set(library["happy"]) | {library["dup"][0]}

    params["mood"] = "angry"
    assert client.get("/api/moods/photos", params=params).status_code == 400


def test_moods_page_and_nav(client) -> None:
    r = client.get("/moods")
    assert r.status_code == 200
    assert "Mood timeline" in r.text
    assert 'href="/moods"' in r.text
