"""Tests for the map view page and its /api/map/points endpoint."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from takeout_rater.db.queries import list_geotagged_points, upsert_asset
from takeout_rater.db.schema import migrate
from takeout_rater.ui.app import create_app


def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    return conn


def _add(conn: sqlite3.Connection, name: str, **fields: object) -> int:
    return upsert_asset(
        conn,
        {
            "relpath": f"Photos/{name}.jpg",
            "filename": f"{name}.jpg",
            "ext": ".jpg",
            "indexed_at": int(time.time()),
            **fields,
        },
    )


@pytest.fixture()
def geo_conn() -> sqlite3.Connection:
    conn = _make_db()
    _add(conn, "paris", geo_lat=48.8566, geo_lon=2.3522, taken_at=2000, favorited=1)
    _add(conn, "tokyo", geo_lat=35.6762, geo_lon=139.6503, taken_at=1000)
    # geoData empty (0/0) but EXIF location present → falls back to EXIF
    _add(conn, "exif", geo_lat=0.0, geo_lon=0.0, geo_exif_lat=-33.86, geo_exif_lon=151.2)
    # No location at all
    _add(conn, "nowhere", geo_lat=0.0, geo_lon=0.0, taken_at=1500)
    _add(conn, "nulls")
    # Trashed photos are hidden
    _add(conn, "trashed", geo_lat=1.0, geo_lon=1.0, trashed=1)
    return conn


def test_list_geotagged_points_uses_exif_fallback_and_skips_missing(
    geo_conn: sqlite3.Connection,
) -> None:
    rows = list_geotagged_points(geo_conn)
    by_lat = {round(r[1], 2): r for r in rows}
    assert set(by_lat) == {48.86, 35.68, -33.86}
    # Sorted by taken_at with NULLs first
    assert [r[3] for r in rows] == [None, 1000, 2000]
    assert by_lat[-33.86][2] == pytest.approx(151.2)


def test_list_geotagged_points_favorited_filter(geo_conn: sqlite3.Connection) -> None:
    rows = list_geotagged_points(geo_conn, favorited=True)
    assert len(rows) == 1
    assert rows[0][4] == 1


def test_list_geotagged_points_dedupes_by_sha(geo_conn: sqlite3.Connection) -> None:
    geo_conn.execute(
        "UPDATE assets SET sha256 = 'same' WHERE filename IN ('paris.jpg', 'tokyo.jpg')"
    )
    assert len(list_geotagged_points(geo_conn, deduped=True)) == 2
    assert len(list_geotagged_points(geo_conn, deduped=False)) == 3


def test_dedupe_keeps_eligible_copy_when_lowest_id_is_filtered_out(
    geo_conn: sqlite3.Connection,
) -> None:
    # The lowest-id copy (paris) is trashed; the other copy (exif) must still show.
    geo_conn.execute("UPDATE assets SET trashed = 1 WHERE filename = 'paris.jpg'")
    geo_conn.execute("UPDATE assets SET sha256 = 'dup' WHERE filename IN ('paris.jpg', 'exif.jpg')")
    ids = {r[0] for r in list_geotagged_points(geo_conn, deduped=True)}
    exif = geo_conn.execute("SELECT id FROM assets WHERE filename = 'exif.jpg'").fetchone()[0]
    assert exif in ids


def test_map_points_endpoint(geo_conn: sqlite3.Connection, tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path, geo_conn, db_root=tmp_path))
    resp = client.get("/api/map/points")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 3
    assert data["total_assets"] == 5
    assert len(data["ids"]) == len(data["lat"]) == len(data["lon"]) == len(data["t"]) == 3
    fav = client.get("/api/map/points?favorited=1").json()
    assert fav["count"] == 1


def test_map_page_renders_and_is_in_nav(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path, _make_db(), db_root=tmp_path))
    resp = client.get("/map")
    assert resp.status_code == 200
    assert 'id="map"' in resp.text
    assert "/static/vendor/leaflet/leaflet.js" in resp.text
    assert 'href="/map"' in client.get("/assets").text


def test_map_static_assets_are_served(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path, _make_db(), db_root=tmp_path))
    for path in (
        "/static/map-view.js",
        "/static/map-view.css",
        "/static/vendor/leaflet/leaflet.js",
        "/static/vendor/world/countries-110m.geojson",
    ):
        assert client.get(path).status_code == 200, path
