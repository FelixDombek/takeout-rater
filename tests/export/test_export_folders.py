"""Tests for export folder layouts (category / trip / place) and the export runner."""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from takeout_rater.db.connection import open_library_db
from takeout_rater.db.queries import (
    bulk_insert_cluster_members,
    insert_cluster,
    insert_clustering_run,
    upsert_asset,
    upsert_asset_places,
)
from takeout_rater.export.folders import (
    DEFAULT_CATEGORIES,
    NO_PLACE_FOLDER,
    NO_TRIP_FOLDER,
    OTHER_FOLDER,
    UNCLASSIFIED_FOLDER,
    Category,
    assign_folders,
    category_folders,
    classify_embeddings,
    format_categories,
    parse_categories,
    place_folders,
    sanitize_folder,
    sanitize_segment,
    trip_folders,
)
from takeout_rater.export.run import export_picks, select_best_assets

DIM = 8


def _ts(y: int, m: int, d: int) -> int:
    return int(datetime(y, m, d, 12, tzinfo=UTC).timestamp())


@pytest.fixture()
def conn(tmp_path: Path):
    c = open_library_db(tmp_path)
    yield c
    c.close()


def _add_asset(conn, i: int, taken_at: int | None = None) -> int:
    return upsert_asset(
        conn,
        {
            "relpath": f"Photos/img{i}.jpg",
            "filename": f"img{i}.jpg",
            "ext": ".jpg",
            "mime": "image/jpeg",
            "taken_at": taken_at,
            "indexed_at": int(time.time()),
        },
    )


def _unit(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def _add_embedding(conn, asset_id: int, vec: np.ndarray) -> None:
    conn.execute(
        "INSERT INTO clip_embeddings (asset_id, embedding, computed_at) VALUES (?, ?, ?)",
        (asset_id, vec.astype("<f4").tobytes(), 0),
    )
    conn.commit()


# Fake text encoder: a prompt "axis N ..." maps to the unit vector on axis N.
def _fake_encode(texts: Sequence[str]) -> np.ndarray:
    return np.stack([_unit(int(t.split()[1])) for t in texts])


# ---------------------------------------------------------------------------
# Names and category parsing
# ---------------------------------------------------------------------------


def test_sanitize_segment_strips_invalid_characters() -> None:
    assert sanitize_segment('a<b>:c"d|e?f*') == "a_b__c_d_e_f_"
    assert sanitize_segment("  trailing dots... ") == "trailing dots"
    assert sanitize_segment("CON") == "_CON"
    assert sanitize_segment("") == "_"
    assert len(sanitize_segment("x" * 200)) == 80


def test_sanitize_folder_keeps_nesting_but_not_traversal() -> None:
    assert sanitize_folder("Nature/Beach") == "Nature/Beach"
    assert sanitize_folder("../../etc") == "_/_/etc"
    assert sanitize_folder("/abs\\path/") == "abs/path"


def test_parse_categories_roundtrip_defaults() -> None:
    parsed = parse_categories(format_categories(DEFAULT_CATEGORIES))
    assert parsed == list(DEFAULT_CATEGORIES)


def test_parse_categories_default_prompt_and_comments() -> None:
    cats = parse_categories("# my labels\nTravel/Beach\n\nFood: pizza; pasta on a plate\n")
    assert cats == [
        Category("Travel/Beach", ("a photo of beach",)),
        Category("Food", ("pizza", "pasta on a plate")),
    ]


@pytest.mark.parametrize("text", ["", "Only one", ": prompt\nOther: x"])
def test_parse_categories_rejects_bad_input(text: str) -> None:
    with pytest.raises(ValueError):
        parse_categories(text)


# ---------------------------------------------------------------------------
# Zero-shot classification
# ---------------------------------------------------------------------------


def test_classify_embeddings_picks_most_similar_and_respects_threshold() -> None:
    text = np.stack([_unit(0), _unit(1)])
    near0 = _unit(0)
    between = (_unit(0) + _unit(1)) / np.sqrt(2)
    best, conf = classify_embeddings(np.stack([near0, between]), text, min_confidence=0.6)
    assert best[0] == 0 and conf[0] > 0.99
    assert best[1] == -1  # 50/50 → below 0.6 → Other


def test_category_folders_uses_embeddings(conn) -> None:
    a = _add_asset(conn, 1)
    b = _add_asset(conn, 2)
    c = _add_asset(conn, 3)  # no embedding
    d = _add_asset(conn, 4)
    _add_embedding(conn, a, _unit(0))
    _add_embedding(conn, b, _unit(1))
    _add_embedding(conn, d, _unit(5))  # equally far from both → 50/50 → Other
    cats = [
        Category("Nature/Beach", ("axis 0 beach",)),
        Category("Food & Drink", ("axis 1 food", "axis 1 meal")),
    ]
    folders = category_folders(
        conn, [a, b, c, d], cats, min_confidence=0.6, encode_texts=_fake_encode
    )
    assert folders == {
        a: "Nature/Beach",
        b: "Food & Drink",
        c: UNCLASSIFIED_FOLDER,
        d: OTHER_FOLDER,
    }


# ---------------------------------------------------------------------------
# Trips and places
# ---------------------------------------------------------------------------


def _add_event(conn, eid: int, kind: str, name: str, start: int, parent: int | None = None):
    conn.execute(
        "INSERT INTO events (id, kind, parent_id, auto_name, start_ts, end_ts, n_assets,"
        " n_days, detected_at) VALUES (?, ?, ?, ?, ?, ?, 0, 1, 0)",
        (eid, kind, parent, name, start, start),
    )


def test_trip_folders(conn) -> None:
    in_trip = _add_asset(conn, 1, _ts(2024, 5, 13))
    in_nested = _add_asset(conn, 2, _ts(2024, 5, 14))
    in_event = _add_asset(conn, 3, _ts(2023, 1, 2))
    loose = _add_asset(conn, 4, _ts(2022, 7, 1))
    undated = _add_asset(conn, 5, None)
    _add_event(conn, 1, "trip", "Lisbon", _ts(2024, 5, 12))
    _add_event(conn, 2, "event", "Wedding: Ana & Rui", _ts(2024, 5, 14), parent=1)
    _add_event(conn, 3, "event", "Birthday", _ts(2023, 1, 2))
    conn.execute("UPDATE events SET custom_name = 'Portugal' WHERE id = 1")
    conn.executemany(
        "INSERT INTO event_assets (event_id, asset_id) VALUES (?, ?)",
        [(1, in_trip), (1, in_nested), (2, in_nested), (3, in_event)],
    )
    conn.commit()

    folders = trip_folders(conn, [in_trip, in_nested, in_event, loose, undated])
    assert folders == {
        in_trip: "2024-05-12 Portugal",
        in_nested: "2024-05-12 Portugal/2024-05-14 Wedding_ Ana & Rui",
        in_event: "2023-01-02 Birthday",
        loose: f"{NO_TRIP_FOLDER}/2022",
        undated: f"{NO_TRIP_FOLDER}/Unknown date",
    }


def test_place_folders(conn) -> None:
    city = _add_asset(conn, 1)
    region_only = _add_asset(conn, 2)
    remote = _add_asset(conn, 3)  # geocoded, but far from any place
    no_gps = _add_asset(conn, 4)
    upsert_asset_places(
        conn,
        [
            {
                "asset_id": city,
                "lat": 38.7,
                "lon": -9.1,
                "geocoder_version": 1,
                "country_code": "PT",
                "country": "Portugal",
                "admin1": "Lisbon",
                "city": "Lisbon",
            },
            {
                "asset_id": region_only,
                "lat": 37.0,
                "lon": -8.0,
                "geocoder_version": 1,
                "country_code": "PT",
                "country": "Portugal",
                "admin1": "Faro",
            },
            {"asset_id": remote, "lat": 0.0, "lon": -30.0, "geocoder_version": 1},
        ],
    )
    folders = place_folders(conn, [city, region_only, remote, no_gps])
    assert folders == {
        city: "Portugal/Lisbon",
        region_only: "Portugal/Faro",
        remote: NO_PLACE_FOLDER,
        no_gps: NO_PLACE_FOLDER,
    }


def test_assign_folders_flat_and_unknown_mode(conn) -> None:
    a = _add_asset(conn, 1)
    assert assign_folders(conn, [a], "flat") == {a: ""}
    with pytest.raises(ValueError):
        assign_folders(conn, [a], "bogus")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def test_export_picks_copies_into_place_folders(conn, tmp_path: Path) -> None:
    photos = tmp_path / "photos"
    (photos / "Photos").mkdir(parents=True)
    ids = []
    for i in range(3):
        ids.append(_add_asset(conn, i))
        (photos / "Photos" / f"img{i}.jpg").write_bytes(b"jpeg")
    run_id = insert_clustering_run(conn, "test", None)
    c1 = insert_cluster(conn, "test", None, run_id=run_id)
    bulk_insert_cluster_members(conn, c1, [(ids[0], 0.0, 1), (ids[1], 1.0, 0)])
    c2 = insert_cluster(conn, "test", None, run_id=run_id)
    bulk_insert_cluster_members(conn, c2, [(ids[2], 0.0, 1)])
    upsert_asset_places(
        conn,
        [
            {
                "asset_id": ids[0],
                "lat": 1.0,
                "lon": 1.0,
                "geocoder_version": 1,
                "country_code": "PT",
                "country": "Portugal",
                "city": "Lisbon",
            }
        ],
    )

    picks = select_best_assets(conn)
    assert sorted(picks) == sorted([(c1, ids[0]), (c2, ids[2])])

    out = tmp_path / "out"
    result = export_picks(conn, picks, photos, out, folder_mode="place")
    assert result.copied == 2
    assert (out / "Portugal" / "Lisbon" / f"cluster{c1:06d}_img0.jpg").is_file()
    assert (out / NO_PLACE_FOLDER / f"cluster{c2:06d}_img2.jpg").is_file()
    assert result.folders == {"Portugal/Lisbon": 1, NO_PLACE_FOLDER: 1}


def test_rerun_with_another_layout_replaces_previous_copies(conn, tmp_path: Path) -> None:
    photos = tmp_path / "photos"
    (photos / "Photos").mkdir(parents=True)
    aid = _add_asset(conn, 0)
    (photos / "Photos" / "img0.jpg").write_bytes(b"jpeg")
    upsert_asset_places(
        conn,
        [
            {
                "asset_id": aid,
                "lat": 1.0,
                "lon": 1.0,
                "geocoder_version": 1,
                "country_code": "PT",
                "country": "Portugal",
                "city": "Lisbon",
            }
        ],
    )
    out = tmp_path / "out"
    (out / "Portugal").mkdir(parents=True)
    (out / "Portugal" / "mine.txt").write_text("user file")
    (out / "keep.jpg").write_bytes(b"user")

    export_picks(conn, [(1, aid)], photos, out, folder_mode="place")
    assert (out / "Portugal" / "Lisbon" / "cluster000001_img0.jpg").is_file()

    result = export_picks(conn, [(1, aid)], photos, out, folder_mode="flat")
    assert result.removed == 1
    assert (out / "cluster000001_img0.jpg").is_file()
    assert not (out / "Portugal" / "Lisbon").exists()  # emptied folder pruned
    assert (out / "Portugal" / "mine.txt").read_text() == "user file"
    assert (out / "keep.jpg").is_file()
