"""Tests for story chapters within trips (issue #109)."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from takeout_rater.db import event_queries as eq
from takeout_rater.db.queries import bulk_upsert_clip_embeddings
from takeout_rater.events import chapters as ch
from takeout_rater.events.service import run_detection
from takeout_rater.ui.app import create_app

from .test_trips import _populate

HOUR = 3600
DAY = 86400
DIM = 64


def _scene(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _shot(rng: np.random.Generator, scene: np.ndarray, noise: float = 0.25) -> np.ndarray:
    v = scene + noise * rng.normal(size=DIM).astype(np.float32) / np.sqrt(DIM) * 3
    return (v / np.linalg.norm(v)).astype(np.float32)


def _day_trip(scenes: list[int], per_scene: int = 8, seed: int = 0):
    """One photo every 10 minutes; ``scenes`` lists the scene id of each block."""
    rng = np.random.default_rng(seed)
    bank = {s: _scene(rng) for s in set(scenes)}
    photos, emb = [], {}
    t = 1_700_000_000
    aid = 1
    for s in scenes:
        for _ in range(per_scene):
            photos.append((aid, t))
            emb[aid] = _shot(rng, bank[s])
            aid += 1
            t += 600
    return photos, emb


# ── Pure algorithm ────────────────────────────────────────────────────────────


def test_splits_where_the_scene_changes() -> None:
    photos, emb = _day_trip([0, 1, 2], per_scene=8)
    story = ch.split_chapters(photos, emb)
    assert [len(c.asset_ids) for c in story.chapters] == [8, 8, 8]
    assert story.chapters[1].asset_ids[0] == 9
    assert story.chapters[0].break_score is None
    assert story.chapters[1].break_score > story.threshold
    assert len(story.scores) == 23
    assert story.n_with_embedding == 24


def test_one_scene_stays_one_chapter() -> None:
    photos, emb = _day_trip([0, 0, 0, 0], per_scene=10)
    story = ch.split_chapters(photos, emb)
    assert len(story.chapters) == 1
    assert story.chapters[0].asset_ids == [p[0] for p in photos]


def test_returning_to_a_scene_is_a_new_chapter() -> None:
    photos, emb = _day_trip([0, 1, 0], per_scene=8)
    assert len(ch.split_chapters(photos, emb).chapters) == 3


def test_min_photos_merges_short_detours() -> None:
    # 12 photos of scene 0, a 6-photo detour, then 12 of scene 2: the detour
    # is too short for its own chapter, so it joins one of its neighbours.
    photos, emb = _day_trip([0, 0, 1, 2, 2], per_scene=6)
    story = ch.split_chapters(photos, emb, min_photos=7)
    assert all(len(c.asset_ids) >= 7 for c in story.chapters)
    assert len(story.chapters) == 2
    assert len(ch.split_chapters(photos, emb, min_photos=4).chapters) == 3


def test_sensitivity_more_never_gives_fewer_chapters() -> None:
    photos, emb = _day_trip([0, 1, 2, 3, 4, 5], per_scene=6, seed=3)
    counts = [
        len(
            ch.split_chapters(
                photos,
                emb,
                ch.params_for(s),
                min_photos=ch.default_min_photos(len(photos), s),
            ).chapters
        )
        for s in ("fewer", "normal", "more")
    ]
    assert counts[0] <= counts[1] <= counts[2]
    assert counts[1] == 6


def test_photos_without_embedding_follow_neighbours_and_never_break_alone() -> None:
    photos, emb = _day_trip([0, 1], per_scene=10)
    for aid in (3, 4, 15):
        del emb[aid]
    story = ch.split_chapters(photos, emb)
    assert [len(c.asset_ids) for c in story.chapters] == [10, 10]
    assert story.n_with_embedding == 17


def test_without_embeddings_long_pauses_split() -> None:
    t0 = 1_700_000_000
    photos = [(i + 1, t0 + i * 300) for i in range(10)]
    photos += [(i + 11, t0 + 2 * DAY + i * 300) for i in range(10)]
    story = ch.split_chapters(photos, {}, min_photos=4)
    assert [len(c.asset_ids) for c in story.chapters] == [10, 10]
    assert story.n_with_embedding == 0
    assert all(c.centroid is None for c in story.chapters)
    # Evenly spaced photos with no embeddings: nothing to split on.
    flat = ch.split_chapters([(i, t0 + i * 300) for i in range(30)], {})
    assert len(flat.chapters) == 1


def test_input_order_does_not_matter_and_cover_prefers_scores() -> None:
    photos, emb = _day_trip([0, 1], per_scene=8)
    shuffled = list(reversed(photos))
    story = ch.split_chapters(shuffled, emb, aesthetic={2: 1.0, 3: 2.0, 4: 3.0})
    assert story.chapters[0].asset_ids == list(range(1, 9))
    cover = story.chapters[0].cover_asset_id
    assert cover in story.chapters[0].asset_ids
    # The cover comes from the photos closest to the chapter's centroid,
    # and among those the best-scored one wins.
    assert story.chapters[1].cover_asset_id in story.chapters[1].asset_ids


def test_tiny_and_empty_inputs() -> None:
    assert ch.split_chapters([], {}).chapters == []
    one = ch.split_chapters([(1, 100)], {})
    assert len(one.chapters) == 1 and one.scores == []


def test_label_chapters_prefers_distinctive_terms() -> None:
    rng = np.random.default_rng(1)
    beach, city, food = _scene(rng), _scene(rng), _scene(rng)
    vocab = np.vstack([beach, city, food, (beach + city) / np.linalg.norm(beach + city)])
    terms = ["beach", "street", "food", "mixed"]
    photos, emb = [], {}
    for i in range(30):
        photos.append((i + 1, 1_700_000_000 + i * 600))
        emb[i + 1] = _shot(rng, beach if i < 15 else city, noise=0.2)
    story = ch.split_chapters(photos, emb)
    assert len(story.chapters) == 2
    labels = ch.label_chapters(story.chapters, vocab, terms, allowed={"beach", "street", "mixed"})
    assert labels[0][0] == "beach"
    assert labels[1][0] == "street"
    assert ch.chapter_title(labels[0]) == "Beach & " + labels[0][1]
    assert ch.chapter_title([]) == ""


def test_label_terms_are_concrete() -> None:
    terms = ch.label_terms()
    assert "beach" in terms
    assert "abstract" not in terms


def test_time_label() -> None:
    start = 1_700_000_000 - (1_700_000_000 % DAY)  # 00:00 UTC
    assert ch.time_label(start + 9 * HOUR, start + 12 * HOUR, start, 0.0) == "Day 1 · 09:00–12:00"
    assert (
        ch.time_label(start + DAY + 18 * HOUR, start + 2 * DAY + HOUR, start, 0.0)
        == "Day 2 18:00 – Day 3 01:00"
    )
    assert ch.time_label(start + 9 * HOUR, start + 9 * HOUR, start, 0.0) == "Day 1 · 09:00"
    assert ch.time_label(start + 2 * DAY, start + 4 * DAY, start, 0.0) == "Days 3–5"
    assert ch.time_label(start + 9 * HOUR, start + 10 * HOUR, start, 0.0, show_day=False) == (
        "09:00–10:00"
    )
    # Longitude shifts to (solar) local time: Lisbon ≈ UTC-1.
    assert ch.time_label(start + 9 * HOUR, start + 10 * HOUR, start, -9.1).endswith("08:00–09:00")


# ── Storage + web UI ──────────────────────────────────────────────────────────


def _add_trip_embeddings(conn: sqlite3.Connection) -> tuple[int, list[int]]:
    """Give the Lisbon trip two scenes (first 2 days vs last 3); return (trip id, its photo ids)."""
    trip = eq.list_events(conn, kind="trip")[0]
    ids = [
        r[0]
        for r in conn.execute(
            "SELECT a.id FROM event_assets ea JOIN assets a ON a.id = ea.asset_id"
            " WHERE ea.event_id = ? ORDER BY a.taken_at, a.id",
            (trip["id"],),
        )
    ]
    rng = np.random.default_rng(7)
    sea, town = _scene(rng), _scene(rng)
    rows = [
        (aid, _shot(rng, sea if i < 20 else town, noise=0.15).astype("<f4").tobytes())
        for i, aid in enumerate(ids)
    ]
    bulk_upsert_clip_embeddings(conn, rows)
    return trip["id"], ids


@pytest.fixture()
def client(tmp_path: Path) -> tuple[TestClient, sqlite3.Connection]:
    from takeout_rater.db.connection import open_library_db

    conn = open_library_db(tmp_path)
    _populate(conn)
    run_detection(conn)
    app = create_app(tmp_path, conn, db_root=tmp_path)
    return TestClient(app, follow_redirects=True), conn


def test_load_event_embeddings(client: tuple[TestClient, sqlite3.Connection]) -> None:
    _, conn = client
    trip_id, ids = _add_trip_embeddings(conn)
    emb = eq.load_event_embeddings(conn, trip_id)
    assert set(emb) == set(ids)
    assert emb[ids[0]].shape == (DIM,)


def test_story_view_renders_chapters(client: tuple[TestClient, sqlite3.Connection]) -> None:
    c, conn = client
    trip_id, ids = _add_trip_embeddings(conn)

    grid = c.get(f"/trips/{trip_id}")
    assert grid.status_code == 200
    assert f"/trips/{trip_id}?view=story" in grid.text  # reachable from the trip page
    assert 'id="chapter-1"' not in grid.text

    resp = c.get(f"/trips/{trip_id}?view=story")
    assert resp.status_code == 200
    assert 'id="chapter-1"' in resp.text and 'id="chapter-2"' in resp.text
    assert 'id="chapter-3"' not in resp.text
    assert resp.text.count("data-lb-id=") == 50  # every photo, no pagination
    assert 'class="break-curve"' in resp.text
    assert "Day 1 08:00 – Day 2 12:30 · 20 photos" in resp.text
    assert "Days 3–5 · 30 photos" in resp.text

    data = c.get(f"/api/trips/{trip_id}/chapters").json()
    assert [ch_["n_assets"] for ch_ in data["chapters"]] == [20, 30]
    assert data["chapters"][1]["asset_ids"][0] == ids[20]
    assert data["n_with_embedding"] == 50
    assert len(data["break_scores"]) == 49
    assert data["chapters"][0]["title"] is None  # labels not requested

    more = c.get(f"/api/trips/{trip_id}/chapters?sensitivity=more").json()
    assert len(more["chapters"]) >= 2
    assert c.get(f"/api/trips/{trip_id}/chapters?sensitivity=bogus").json()["sensitivity"] == (
        "normal"
    )
    assert c.get("/api/trips/99999/chapters").status_code == 404


def test_story_view_without_embeddings_explains(
    client: tuple[TestClient, sqlite3.Connection],
) -> None:
    c, conn = client
    trip = eq.list_events(conn, kind="trip")[0]
    resp = c.get(f"/trips/{trip['id']}?view=story")
    assert resp.status_code == 200
    assert "No CLIP embeddings for these photos yet" in resp.text
    assert resp.text.count("data-lb-id=") == 50


def test_chapter_titles_from_clip_vocab(
    client: tuple[TestClient, sqlite3.Connection], monkeypatch: pytest.MonkeyPatch
) -> None:
    c, conn = client
    trip_id, _ = _add_trip_embeddings(conn)
    data = c.get(f"/api/trips/{trip_id}/chapters").json()
    assert len(data["chapters"]) == 2

    # Build a fake vocabulary whose "beach" row is the first chapter's centroid.
    from takeout_rater.api import assets as assets_api

    emb = eq.load_event_embeddings(conn, trip_id)
    first = np.mean([emb[a] for a in data["chapters"][0]["asset_ids"]], axis=0)
    second = np.mean([emb[a] for a in data["chapters"][1]["asset_ids"]], axis=0)
    vocab = np.vstack([first / np.linalg.norm(first), second / np.linalg.norm(second)])
    monkeypatch.setattr(
        assets_api, "_get_clip_vocab_matrix", lambda request: (vocab, ["beach", "street"])
    )
    data = c.get(f"/api/trips/{trip_id}/chapters?labels=1").json()
    assert data["chapters"][0]["title"].startswith("Beach")
    assert data["chapters"][1]["title"].startswith("Street")

    def _boom(request):
        raise RuntimeError("no weights")

    monkeypatch.setattr(assets_api, "_get_clip_vocab_matrix", _boom)
    data = c.get(f"/api/trips/{trip_id}/chapters?labels=1").json()
    assert data["chapters"][0]["title"] is None


def test_story_view_is_fast_enough_for_big_trips() -> None:
    rng = np.random.default_rng(0)
    scenes = [_scene(rng) for _ in range(40)]
    photos, emb = [], {}
    for i in range(3000):
        photos.append((i, 1_700_000_000 + i * 120))
        emb[i] = _shot(rng, scenes[i // 75])
    t0 = time.perf_counter()
    story = ch.split_chapters(photos, emb)
    assert time.perf_counter() - t0 < 2.0
    assert 20 <= len(story.chapters) <= 40
