"""Tests for offline reverse geocoding and the asset_places pipeline."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from takeout_rater.db.queries import (
    PlaceFilter,
    count_asset_places,
    count_assets,
    count_assets_deduped,
    get_asset_place,
    list_assets,
    list_assets_deduped,
    list_place_facets,
    upsert_asset,
)
from takeout_rater.db.schema import migrate
from takeout_rater.geo import places
from takeout_rater.geo.library import geocode_library
from takeout_rater.geo.places import Place, asset_coords, reverse_geocode, reverse_geocode_many

# A tiny gazetteer: a big city with a district, a separate town in another
# region close by, and a remote village.
_LOCATIONS = [
    {"country_code": "XA", "city": "Bigcity", "latitude": 50.0, "longitude": 10.0,
     "population": 2_000_000, "state": "North"},
    {"country_code": "XA", "city": "Bigcity-Centre", "latitude": 50.05, "longitude": 10.05,
     "population": 90_000, "state": "North"},
    {"country_code": "XA", "city": "Bordertown", "latitude": 50.0, "longitude": 10.15,
     "population": 20_000, "state": "South", "county": "South County"},
    {"country_code": "XB", "city": "Farvillage", "latitude": 60.0, "longitude": 20.0,
     "population": 1_200, "state": "Wild"},
]  # fmt: skip
_COUNTRIES = {"XA": "Examplia", "XB": "Otherland"}


@pytest.fixture()
def fake_gazetteer(monkeypatch: pytest.MonkeyPatch) -> places._Gazetteer:  # noqa: SLF001
    g = places._Gazetteer(_LOCATIONS, _COUNTRIES)  # noqa: SLF001
    monkeypatch.setattr(places, "_GAZETTEER", g)
    return g


# ── asset_coords ─────────────────────────────────────────────────────────────


def test_asset_coords_prefers_geodata() -> None:
    assert asset_coords(1.5, 2.5, 3.0, 4.0) == (1.5, 2.5)


def test_asset_coords_falls_back_to_exif_when_zero() -> None:
    assert asset_coords(0.0, 0.0, 3.0, 4.0) == (3.0, 4.0)


def test_asset_coords_none_when_unknown() -> None:
    assert asset_coords(0.0, 0.0, 0.0, 0.0) is None
    assert asset_coords(None, None) is None
    assert asset_coords(95.0, 10.0) is None  # out of range


# ── lookup ───────────────────────────────────────────────────────────────────


def test_district_snaps_to_city(fake_gazetteer: object) -> None:
    place = reverse_geocode(50.051, 10.049)
    assert place is not None
    assert place.city == "Bigcity"
    assert place.country == "Examplia"
    assert place.label == "Bigcity, Examplia"


def test_town_in_other_region_is_not_merged(fake_gazetteer: object) -> None:
    place = reverse_geocode(50.0, 10.151)
    assert place is not None
    assert place.city == "Bordertown"
    assert place.admin1 == "South"
    assert place.admin2 == "South County"


def test_far_from_any_city_keeps_region_only(fake_gazetteer: object) -> None:
    # ~110 km north of Farvillage: too far for a city, close enough for a country.
    place = reverse_geocode(61.0, 20.0)
    assert place is not None
    assert place.city is None
    assert place.country_code == "XB"
    assert place.label == "Wild, Otherland"


def test_open_sea_returns_none(fake_gazetteer: object) -> None:
    assert reverse_geocode(10.0, -40.0) is None


def test_many_handles_invalid_entries(fake_gazetteer: object) -> None:
    out = reverse_geocode_many([(0.0, 0.0), (50.0, 10.0), (200.0, 0.0)])
    assert out[0] is None
    assert out[1] is not None and out[1].city == "Bigcity"
    assert out[2] is None


def test_label_fallbacks() -> None:
    assert Place("XA", "Examplia").label == "Examplia"
    assert Place("XA", "", city="Town").label == "Town"


def test_real_gazetteer_merges_city_districts() -> None:
    pytest.importorskip("reverse_geocode")
    lisbon, berlin_kreuzberg, newark = reverse_geocode_many(
        [(38.7223, -9.1393), (52.49, 13.42), (40.735, -74.172)]
    )
    assert lisbon is not None and lisbon.label == "Lisbon, Portugal"
    assert berlin_kreuzberg is not None and berlin_kreuzberg.city == "Berlin"
    assert newark is not None and newark.city == "Newark"  # not snapped into New York


# ── library pipeline + queries ───────────────────────────────────────────────


def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    return conn


def _add(conn: sqlite3.Connection, name: str, lat: float, lon: float, **extra: object) -> int:
    return upsert_asset(
        conn,
        {
            "relpath": f"Photos/{name}.jpg",
            "filename": f"{name}.jpg",
            "ext": ".jpg",
            "geo_lat": lat,
            "geo_lon": lon,
            "taken_at": int(time.time()),
            "indexed_at": int(time.time()),
            **extra,
        },
    )


@pytest.fixture()
def geo_db(fake_gazetteer: object) -> sqlite3.Connection:
    conn = _make_db()
    _add(conn, "city1", 50.0, 10.0)
    _add(conn, "city2", 50.05, 10.05)
    _add(conn, "border", 50.0, 10.15)
    _add(conn, "nogps", 0.0, 0.0)
    _add(conn, "exif", 0.0, 0.0, geo_exif_lat=60.0, geo_exif_lon=20.0)
    return conn


def test_geocode_library_fills_places(geo_db: sqlite3.Connection) -> None:
    result = geocode_library(geo_db)
    assert result.candidates == 4
    assert result.processed == 4
    assert result.named == 4
    assert count_asset_places(geo_db) == 4
    exif_id = geo_db.execute("SELECT id FROM assets WHERE filename='exif.jpg'").fetchone()[0]
    place = get_asset_place(geo_db, exif_id)
    assert place is not None and place.city == "Farvillage"


def test_geocode_library_missing_only_skips_done_and_redoes_moved(
    geo_db: sqlite3.Connection,
) -> None:
    geocode_library(geo_db)
    assert geocode_library(geo_db).processed == 0
    geo_db.execute("UPDATE assets SET geo_lat = 50.0, geo_lon = 10.151 WHERE filename='city1.jpg'")
    again = geocode_library(geo_db)
    assert again.processed == 1
    assert geocode_library(geo_db, missing_only=False).processed == 4


def test_geocode_library_removes_place_when_gps_cleared(geo_db: sqlite3.Connection) -> None:
    geocode_library(geo_db)
    aid = geo_db.execute("SELECT id FROM assets WHERE filename='city1.jpg'").fetchone()[0]
    geo_db.execute("UPDATE assets SET geo_lat = 0, geo_lon = 0 WHERE id = ?", (aid,))
    result = geocode_library(geo_db)
    assert result.removed == 1
    assert get_asset_place(geo_db, aid) is None
    assert count_asset_places(geo_db) == 3


def test_place_filter_on_listing_queries(geo_db: sqlite3.Connection) -> None:
    geocode_library(geo_db)
    xa = PlaceFilter(country_code="XA")
    assert count_assets(geo_db, place=xa) == 3
    assert count_assets_deduped(geo_db, place=xa) == 3
    city = PlaceFilter(country_code="XA", admin1="North", city="Bigcity")
    assert {a.filename for a in list_assets(geo_db, place=city)} == {"city1.jpg", "city2.jpg"}
    assert len(list_assets_deduped(geo_db, place=PlaceFilter(city="Bordertown"))) == 1
    assert count_assets(geo_db, place=PlaceFilter(country_code="ZZ")) == 0


def test_place_facets(geo_db: sqlite3.Connection) -> None:
    geocode_library(geo_db)
    facets = list_place_facets(geo_db)
    assert facets["countries"][0] == {"code": "XA", "name": "Examplia", "count": 3}
    assert "regions" not in facets
    facets = list_place_facets(geo_db, country_code="XA", admin1="North")
    assert {r["name"] for r in facets["regions"]} == {"North", "South"}
    assert facets["cities"] == [{"name": "Bigcity", "count": 2}]


# ── web UI ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def client(geo_db: sqlite3.Connection, tmp_path: Path) -> TestClient:
    from takeout_rater.ui.app import create_app

    geocode_library(geo_db)
    app = create_app(tmp_path, geo_db, db_root=tmp_path)
    return TestClient(app, follow_redirects=True)


def test_browse_shows_place_filter(client: TestClient) -> None:
    html = client.get("/assets").text
    assert 'id="place-filter"' in html
    assert "Examplia (3)" in html


def test_browse_filters_by_city(client: TestClient) -> None:
    html = client.get("/assets?country=XA&region=North&city=Bigcity").text
    assert "2 unique photos" in html
    assert "in Bigcity, North, Examplia" in html
    assert "Bigcity (2)" in html


def test_browse_hint_when_no_places_yet(fake_gazetteer: object, tmp_path: Path) -> None:
    from takeout_rater.ui.app import create_app

    conn = _make_db()
    _add(conn, "a", 50.0, 10.0)
    html = TestClient(create_app(tmp_path, conn, db_root=tmp_path)).get("/assets").text
    assert "Look up place names" in html
    assert 'id="place-filter"' not in html


def test_api_places(client: TestClient) -> None:
    data = client.get("/api/places?country=xa").json()
    assert data["total"] == 4
    assert data["countries"][0]["code"] == "XA"
    assert data["cities"][0] == {"name": "Bigcity", "count": 2}


def test_detail_shows_place_link(client: TestClient, geo_db: sqlite3.Connection) -> None:
    aid = geo_db.execute("SELECT id FROM assets WHERE filename='border.jpg'").fetchone()[0]
    html = client.get(f"/assets/{aid}?partial=1").text
    assert "Bordertown, South, Examplia" in html
    assert "city=Bordertown" in html


def test_timeline_seek_accepts_place(client: TestClient) -> None:
    resp = client.get("/api/timeline/seek?timestamp=0&country=XA")
    assert resp.json() == {"page": 1}


def test_geocode_job_endpoint(fake_gazetteer: object, tmp_path: Path) -> None:
    from takeout_rater.db.connection import open_library_db
    from takeout_rater.ui.app import create_app

    conn = open_library_db(tmp_path)
    _add(conn, "a", 50.0, 10.0)
    client = TestClient(create_app(tmp_path, conn, db_root=tmp_path))
    assert client.post("/api/jobs/geocode/start", json={"mode": "bogus"}).status_code == 400
    assert client.post("/api/jobs/geocode/start", json={"mode": "full"}).status_code == 200
    deadline = time.time() + 10
    status: dict = {}
    while time.time() < deadline:
        status = client.get("/api/jobs/status?job_type=geocode").json()
        if status["done"]:
            break
        time.sleep(0.05)
    assert status["done"] and status["error"] is None
    assert count_asset_places(conn) == 1


def test_post_index_geocode_keeps_message_and_counts_named(geo_db: sqlite3.Connection) -> None:
    from takeout_rater.api.jobs import JobProgress, _geocode_after_index

    progress = JobProgress(job_type="index", message="Indexed 5 photo(s).")
    assert _geocode_after_index(geo_db, progress, {"index": progress}) == 4
    assert progress.message == "Indexed 5 photo(s)."


def test_post_index_geocode_skips_when_manual_job_running(geo_db: sqlite3.Connection) -> None:
    from takeout_rater.api.jobs import JobProgress, _geocode_after_index

    progress = JobProgress(job_type="index")
    manual = JobProgress(job_type="geocode", running=True)
    assert _geocode_after_index(geo_db, progress, {"geocode": manual}) == 0
    assert count_asset_places(geo_db) == 0


def test_post_index_geocode_honours_cancel(geo_db: sqlite3.Connection) -> None:
    from takeout_rater.api.jobs import JobProgress, _geocode_after_index

    progress = JobProgress(job_type="index")
    progress.cancel_event.set()
    assert _geocode_after_index(geo_db, progress, {}) == 0
