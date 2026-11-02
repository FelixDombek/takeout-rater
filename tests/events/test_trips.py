"""Tests for trip/event detection, storage and the /trips UI."""

from __future__ import annotations

import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from takeout_rater.db import event_queries as eq
from takeout_rater.db.queries import upsert_asset, upsert_asset_scores
from takeout_rater.db.schema import CURRENT_SCHEMA_VERSION, migrate
from takeout_rater.events.detector import (
    DetectionParams,
    PhotoPoint,
    detect,
    estimate_homes,
    format_date_range,
    haversine_km,
    place_label,
)
from takeout_rater.events.service import run_detection
from takeout_rater.ui.app import create_app

BERLIN = (52.52, 13.40)
LISBON = (38.72, -9.14)
HOUR = 3600
DAY = 86400


def _ts(y: int, m: int, d: int, h: int = 12) -> int:
    return int(datetime(y, m, d, h, tzinfo=UTC).timestamp())


def _scenario() -> list[tuple[int, float | None, float | None]]:
    """(ts, lat, lon) rows: 60 days at home, a 5-day Lisbon trip, a home birthday."""
    rows: list[tuple[int, float | None, float | None]] = []
    start = _ts(2023, 4, 1)
    for i in range(60):  # one everyday photo per day at home
        rows.append((start + i * DAY, BERLIN[0] + 0.001 * (i % 5), BERLIN[1]))
    trip_start = _ts(2023, 6, 3, 9)
    for d in range(5):  # Lisbon, 10 photos per day; day 2 has no GPS at all
        for k in range(10):
            ts = trip_start + d * DAY + k * HOUR // 2
            if d == 2:
                rows.append((ts, None, None))
            else:
                rows.append((ts, LISBON[0] + 0.01 * k, LISBON[1]))
    party = _ts(2023, 5, 10, 18)
    for k in range(20):  # birthday at home: 20 photos in ~2 hours
        rows.append((party + k * 360, BERLIN[0], BERLIN[1]))
    return rows


def _points() -> list[PhotoPoint]:
    return [PhotoPoint(i + 1, ts, lat, lon) for i, (ts, lat, lon) in enumerate(_scenario())]


# ── Pure detector ─────────────────────────────────────────────────────────────


def test_haversine_berlin_lisbon() -> None:
    assert 2300 < haversine_km(*BERLIN, *LISBON) < 2400


def test_home_is_most_frequent_place() -> None:
    homes, default = estimate_homes(_points())
    assert default is not None
    assert haversine_km(*default, *BERLIN) < 5
    assert haversine_km(*homes[2023], *BERLIN) < 5


def test_detects_trip_bridging_day_without_gps() -> None:
    result = detect(_points())
    assert len(result.trips) == 1
    trip = result.trips[0]
    assert trip.n_days == 5
    assert len(trip.asset_ids) == 50  # includes the 10 photos without GPS
    assert haversine_km(trip.center_lat, trip.center_lon, *LISBON) < 20
    assert trip.max_distance_km > 2000


def test_home_day_splits_trips() -> None:
    pts = [p for p in _points() if not (p.lat is None and p.lon is None)]
    # Add geotagged home photos on the middle day → two separate trips
    mid = _ts(2023, 6, 5, 12)
    pts += [PhotoPoint(10_000 + k, mid + k * 60, *BERLIN) for k in range(3)]
    result = detect(pts, DetectionParams(trip_min_photos=5))
    assert len(result.trips) == 2


def test_events_are_split_by_gap_and_nested_in_trips() -> None:
    result = detect(_points())
    # The birthday is a stand-alone event; trip days are nested events.
    assert len(result.events) == 1
    party = result.events[0]
    assert len(party.asset_ids) == 20
    assert all(c.kind == "event" for c in result.trips[0].children)
    # Trip days have only 10 photos (< 15 minimum), so no nested events by default
    assert result.trips[0].children == []
    nested = detect(_points(), DetectionParams(event_min_photos=10)).trips[0].children
    assert len(nested) == 5


def test_no_geo_means_no_trips_but_events_still_work() -> None:
    pts = [PhotoPoint(i, 1_700_000_000 + i * 60) for i in range(30)]
    result = detect(pts)
    assert result.trips == []
    assert len(result.events) == 1
    assert result.default_home is None


def test_place_label_variants() -> None:
    assert place_label([("Lisbon", "Lisbon", "Portugal")] * 5) == "Lisbon, Portugal"
    two = [("Lisbon", "Lisbon", "Portugal")] * 5 + [("Porto", "Porto", "Portugal")] * 3
    assert place_label(two) == "Lisbon & Porto, Portugal"
    countries = [("Lisbon", None, "Portugal")] * 5 + [("Seville", None, "Spain")] * 4
    assert place_label(countries) == "Portugal & Spain"
    assert place_label([]) is None


def test_format_date_range() -> None:
    assert format_date_range(_ts(2024, 3, 12, 9), _ts(2024, 3, 12, 20)) == "12 Mar 2024"
    assert format_date_range(_ts(2023, 6, 3), _ts(2023, 6, 9)) == "3–9 Jun 2023"
    assert format_date_range(_ts(2023, 6, 28), _ts(2023, 7, 2)) == "28 Jun – 2 Jul 2023"
    assert format_date_range(_ts(2023, 12, 30), _ts(2024, 1, 2)) == "30 Dec 2023 – 2 Jan 2024"


# ── Database + service ────────────────────────────────────────────────────────


def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    return conn


def _populate(conn: sqlite3.Connection) -> None:
    for i, (ts, lat, lon) in enumerate(_scenario()):
        upsert_asset(
            conn,
            {
                "relpath": f"Photos/img{i}.jpg",
                "filename": f"img{i}.jpg",
                "ext": ".jpg",
                "mime": "image/jpeg",
                "taken_at": ts,
                # Takeout stores 0/0 for unknown geoData
                "geo_lat": lat if lat is not None else 0.0,
                "geo_lon": lon if lon is not None else 0.0,
                "indexed_at": int(time.time()),
            },
        )
    conn.commit()


def test_migration_from_v14_adds_tables() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE clustering_runs (id INTEGER PRIMARY KEY); PRAGMA user_version = 13;"
    )
    migrate(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == CURRENT_SCHEMA_VERSION
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"events", "event_assets"} <= tables


def test_run_detection_stores_trip_with_cover_and_keeps_custom_name() -> None:
    conn = _make_db()
    _populate(conn)
    # Give one Lisbon photo the best aesthetic score → it must become the cover.
    lisbon_ids = [
        r[0] for r in conn.execute("SELECT id FROM assets WHERE geo_lat BETWEEN 38 AND 39")
    ]
    upsert_asset_scores(
        conn,
        "aesthetic",
        "default",
        [(aid, "aesthetic", 5.0) for aid in lisbon_ids[:-1]] + [(lisbon_ids[-1], "aesthetic", 9.0)],
    )

    summary = run_detection(conn)
    assert summary["n_trips"] == 1
    assert summary["n_events"] == 1
    trips = eq.list_events(conn, kind="trip")
    assert len(trips) == 1
    trip = trips[0]
    assert trip["cover_asset_id"] == lisbon_ids[-1]
    assert trip["n_assets"] == 50
    assert trip["name"].startswith("Trip · Jun 2023")

    assert eq.rename_event(conn, trip["id"], "Lisbon with friends")
    run_detection(conn)
    trip = eq.list_events(conn, kind="trip")[0]
    assert trip["name"] == "Lisbon with friends"
    assert trip["auto_name"].startswith("Trip")

    member = conn.execute(
        "SELECT asset_id FROM event_assets WHERE event_id = ? LIMIT 1", (trip["id"],)
    ).fetchone()[0]
    assert [e["id"] for e in eq.get_events_for_asset(conn, member)] == [trip["id"]]


def test_run_detection_uses_asset_places_when_present() -> None:
    conn = _make_db()
    _populate(conn)
    conn.execute(
        "CREATE TABLE asset_places (asset_id INTEGER PRIMARY KEY, country_code TEXT,"
        " country TEXT, admin1 TEXT, city TEXT)"
    )
    conn.execute(
        "INSERT INTO asset_places SELECT id, 'PT', 'Portugal', 'Lisbon', 'Lisbon'"
        " FROM assets WHERE geo_lat BETWEEN 38 AND 39"
    )
    conn.commit()
    run_detection(conn)
    trip = eq.list_events(conn, kind="trip")[0]
    assert trip["name"] == "Lisbon, Portugal"
    assert trip["place_label"] == "Lisbon, Portugal"


# ── Web UI ────────────────────────────────────────────────────────────────────


@pytest.fixture()
def client(tmp_path: Path) -> tuple[TestClient, sqlite3.Connection]:
    conn = _make_db()
    _populate(conn)
    app = create_app(tmp_path, conn, db_root=tmp_path)
    return TestClient(app, follow_redirects=True), conn


def test_trips_page_empty_then_detect(client: tuple[TestClient, sqlite3.Connection]) -> None:
    c, _ = client
    resp = c.get("/trips")
    assert resp.status_code == 200
    assert "No trips or events detected yet" in resp.text
    assert 'href="/trips"' in resp.text  # nav link

    resp = c.post("/api/trips/detect", json={"trip_min_distance_km": 100})
    assert resp.status_code == 200
    assert resp.json()["n_trips"] == 1

    resp = c.get("/trips")
    assert "Trip · Jun 2023" in resp.text
    assert "3–7 Jun 2023" in resp.text
    assert "/map?from=" in resp.text

    resp = c.get("/trips?show=events")
    assert "Trip · Jun 2023" not in resp.text
    assert "Event · 10 May 2023" in resp.text


def test_trip_detail_and_rename(client: tuple[TestClient, sqlite3.Connection]) -> None:
    c, conn = client
    c.post("/api/trips/detect")
    trip = eq.list_events(conn, kind="trip")[0]
    resp = c.get(f"/trips/{trip['id']}")
    assert resp.status_code == 200
    assert "50 photos" in resp.text
    assert resp.text.count("data-lb-id=") == 50

    resp = c.post(f"/api/trips/{trip['id']}/rename", json={"name": "  Lisboa  "})
    assert resp.json()["name"] == "Lisboa"
    resp = c.post(f"/api/trips/{trip['id']}/rename", json={"name": ""})
    assert resp.json()["name"].startswith("Trip")

    assert c.get("/trips/99999").status_code == 404
    assert c.post("/api/trips/99999/rename", json={"name": "x"}).status_code == 404


def test_api_list_trips(client: tuple[TestClient, sqlite3.Connection]) -> None:
    c, _ = client
    c.post("/api/trips/detect")
    data = c.get("/api/trips?kind=trip").json()
    assert len(data["items"]) == 1
    item = data["items"][0]
    assert item["map_url"] == f"/map?from={item['start_ts']}&to={item['end_ts']}"
    assert c.get("/api/trips?kind=bogus").status_code == 400


def test_auto_name_for_events_includes_date() -> None:
    from takeout_rater.events.detector import DetectedGroup, auto_name

    ev = DetectedGroup("event", _ts(2024, 12, 24, 16), _ts(2024, 12, 24, 20), [1], 52.5, 13.4)
    assert auto_name(ev, "Berlin, Germany") == "Berlin · 24 Dec 2024"
    assert auto_name(ev, None) == "Event · 24 Dec 2024"
    trip = DetectedGroup("trip", _ts(2023, 6, 3), _ts(2023, 6, 9), [1], 38.7, -9.1)
    assert auto_name(trip, "Lisbon, Portugal") == "Lisbon, Portugal"
    assert auto_name(trip, None) == "Trip · Jun 2023"
