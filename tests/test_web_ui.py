"""Tests for the FastAPI browse UI routes."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from takeout_rater.db.queries import upsert_asset  # noqa: E402
from takeout_rater.db.schema import migrate  # noqa: E402
from takeout_rater.ui.app import create_app  # noqa: E402


def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    return conn


def _add_asset(conn: sqlite3.Connection, relpath: str = "Photos/img.jpg") -> int:
    return upsert_asset(
        conn,
        {
            "relpath": relpath,
            "filename": Path(relpath).name,
            "ext": Path(relpath).suffix.lower(),
            "size_bytes": 512,
            "mime": "image/jpeg",
            "title": Path(relpath).stem,
            "indexed_at": int(time.time()),
        },
    )


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    conn = _make_db()
    app = create_app(tmp_path, conn)
    return TestClient(app, follow_redirects=True)


@pytest.fixture()
def client_with_assets(tmp_path: Path) -> TestClient:
    conn = _make_db()
    for i in range(3):
        _add_asset(conn, f"Photos/img{i}.jpg")
    app = create_app(tmp_path, conn)
    return TestClient(app, follow_redirects=True)


# ── GET / → redirect to /assets ───────────────────────────────────────────────


def test_root_redirects_to_assets(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "/assets" in str(resp.url)


# ── GET /assets ───────────────────────────────────────────────────────────────


def test_browse_empty_db_returns_200(client: TestClient) -> None:
    resp = client.get("/assets")
    assert resp.status_code == 200


def test_browse_returns_html(client: TestClient) -> None:
    resp = client.get("/assets")
    assert "text/html" in resp.headers["content-type"]


def test_browse_shows_asset_count(client_with_assets: TestClient) -> None:
    resp = client_with_assets.get("/assets")
    assert "3 photos" in resp.text


def test_browse_shows_empty_message_when_no_assets(client: TestClient) -> None:
    resp = client.get("/assets")
    assert "No photos" in resp.text


def test_browse_assets_page2(client_with_assets: TestClient) -> None:
    resp = client_with_assets.get("/assets?page=2")
    assert resp.status_code == 200


# ── GET /assets/{id} ─────────────────────────────────────────────────────────


def test_asset_detail_returns_200(client_with_assets: TestClient) -> None:
    resp = client_with_assets.get("/assets/1")
    assert resp.status_code == 200


def test_asset_detail_not_found_returns_404(client: TestClient) -> None:
    resp = client.get("/assets/99999")
    assert resp.status_code == 404


def test_asset_detail_contains_filename(client_with_assets: TestClient) -> None:
    resp = client_with_assets.get("/assets/1")
    assert "img0.jpg" in resp.text


# ── GET /thumbs/{id} ─────────────────────────────────────────────────────────


def test_thumbnail_not_found_returns_404(client: TestClient) -> None:
    resp = client.get("/thumbs/1")
    assert resp.status_code == 404


def test_thumbnail_serves_jpeg(tmp_path: Path) -> None:
    """When a thumbnail file exists, it should be served with image/jpeg content-type."""
    pytest.importorskip("PIL")
    from PIL import Image  # noqa: PLC0415

    from takeout_rater.indexing.thumbnailer import thumb_path_for_id  # noqa: PLC0415

    conn = _make_db()
    asset_id = _add_asset(conn)
    thumbs_dir = tmp_path / "takeout-rater" / "thumbs"
    thumbs_dir.mkdir(parents=True)
    thumb = thumb_path_for_id(thumbs_dir, asset_id)
    thumb.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(100, 150, 200)).save(thumb, "JPEG")

    app = create_app(tmp_path, conn)
    client = TestClient(app)
    resp = client.get(f"/thumbs/{asset_id}")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
