"""Tests for the FastAPI browse UI routes."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from takeout_rater.clustering.builder import build_clusters  # noqa: E402
from takeout_rater.db.queries import upsert_asset, upsert_phash  # noqa: E402
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


@pytest.fixture()
def client_with_clusters(tmp_path: Path) -> TestClient:
    conn = _make_db()
    id1 = _add_asset(conn, "Photos/a.jpg")
    id2 = _add_asset(conn, "Photos/b.jpg")
    upsert_phash(conn, id1, "0000000000000000")
    upsert_phash(conn, id2, "0000000000000001")  # 1 bit diff
    build_clusters(conn, threshold=5)
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
    # Default dedupe mode shows "3 unique photos" (assets have no sha256 → 3 groups)
    assert "3 unique photos" in resp.text


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


# ── GET /clusters ─────────────────────────────────────────────────────────────


def test_clusters_empty_db_returns_200(client: TestClient) -> None:
    resp = client.get("/clusters")
    assert resp.status_code == 200


def test_clusters_returns_html(client: TestClient) -> None:
    resp = client.get("/clusters")
    assert "text/html" in resp.headers["content-type"]


def test_clusters_empty_shows_no_clusters_message(client: TestClient) -> None:
    resp = client.get("/clusters")
    assert "No clusters" in resp.text


def test_clusters_shows_cluster_count(client_with_clusters: TestClient) -> None:
    resp = client_with_clusters.get("/clusters")
    assert resp.status_code == 200
    assert "1 cluster" in resp.text


def test_clusters_page2_returns_200(client_with_clusters: TestClient) -> None:
    resp = client_with_clusters.get("/clusters?page=2")
    assert resp.status_code == 200


# ── GET /clusters/{id} ───────────────────────────────────────────────────────


def test_cluster_detail_returns_200(client_with_clusters: TestClient) -> None:
    resp = client_with_clusters.get("/clusters/1")
    assert resp.status_code == 200


def test_cluster_detail_not_found_returns_404(client: TestClient) -> None:
    resp = client.get("/clusters/99999")
    assert resp.status_code == 404


def test_cluster_detail_shows_members(client_with_clusters: TestClient) -> None:
    resp = client_with_clusters.get("/clusters/1")
    assert "a.jpg" in resp.text or "b.jpg" in resp.text


def test_cluster_detail_shows_rep_badge(client_with_clusters: TestClient) -> None:
    resp = client_with_clusters.get("/clusters/1")
    assert "REP" in resp.text


# ── nav link ─────────────────────────────────────────────────────────────────


def test_browse_page_has_clusters_nav_link(client: TestClient) -> None:
    resp = client.get("/assets")
    assert "/clusters" in resp.text


# ── score range filter ────────────────────────────────────────────────────────


@pytest.fixture()
def client_with_scores(tmp_path: Path) -> TestClient:
    from takeout_rater.db.queries import (  # noqa: PLC0415
        bulk_insert_asset_scores,
        finish_scorer_run,
        insert_scorer_run,
    )

    conn = _make_db()
    ids = [_add_asset(conn, f"Photos/img{i}.jpg") for i in range(5)]
    run_id = insert_scorer_run(conn, "blur", "default")
    bulk_insert_asset_scores(
        conn, run_id, [(aid, "sharpness", float(i * 20)) for i, aid in enumerate(ids)]
    )
    finish_scorer_run(conn, run_id)
    app = create_app(tmp_path, conn)
    return TestClient(app, follow_redirects=True)


def test_browse_with_sort_by_score_returns_200(client_with_scores: TestClient) -> None:
    resp = client_with_scores.get("/assets?sort_by=blur:sharpness")
    assert resp.status_code == 200


def test_browse_min_score_filter_reduces_count(client_with_scores: TestClient) -> None:
    client_with_scores.get("/assets?sort_by=blur:sharpness")
    resp_filtered = client_with_scores.get("/assets?sort_by=blur:sharpness&min_score=40")
    # Page with min_score should mention fewer photos
    assert resp_filtered.status_code == 200
    # Check that the filtered page doesn't have the full count
    assert "5 photos" not in resp_filtered.text


def test_browse_max_score_filter_reduces_count(client_with_scores: TestClient) -> None:
    resp_filtered = client_with_scores.get("/assets?sort_by=blur:sharpness&max_score=20")
    assert resp_filtered.status_code == 200
    assert "5 photos" not in resp_filtered.text


def test_browse_score_range_shows_filter_indicator(client_with_scores: TestClient) -> None:
    resp = client_with_scores.get("/assets?sort_by=blur:sharpness&min_score=40")
    assert "filtered by range" in resp.text


def test_browse_blank_min_score_returns_200(client_with_scores: TestClient) -> None:
    """Blank min_score= should not cause a 422 validation error."""
    resp = client_with_scores.get("/assets?sort_by=blur:sharpness&min_score=")
    assert resp.status_code == 200


def test_browse_blank_max_score_returns_200(client_with_scores: TestClient) -> None:
    """Blank max_score= should not cause a 422 validation error."""
    resp = client_with_scores.get("/assets?sort_by=blur:sharpness&max_score=")
    assert resp.status_code == 200


def test_browse_both_blank_scores_returns_200(client_with_scores: TestClient) -> None:
    """Both blank score params together should not cause a 422 validation error."""
    resp = client_with_scores.get("/assets?sort_by=blur:sharpness&min_score=&max_score=")
    assert resp.status_code == 200


def test_browse_blank_score_treats_as_unfiltered(client_with_scores: TestClient) -> None:
    """Blank min_score should produce the same result as omitting the parameter."""
    resp_no_filter = client_with_scores.get("/assets?sort_by=blur:sharpness")
    resp_blank = client_with_scores.get("/assets?sort_by=blur:sharpness&min_score=")
    assert resp_no_filter.status_code == 200
    assert resp_blank.status_code == 200
    # Both should show the same total count (blank is same as absent)
    assert "5 photos" in resp_blank.text


def test_browse_invalid_score_returns_200(client_with_scores: TestClient) -> None:
    """Non-numeric min_score should be silently ignored, not produce a 422."""
    resp = client_with_scores.get("/assets?sort_by=blur:sharpness&min_score=abc")
    assert resp.status_code == 200


# ── preset API ────────────────────────────────────────────────────────────────


def test_api_presets_list_empty(client: TestClient) -> None:
    resp = client.get("/api/presets")
    assert resp.status_code == 200
    assert resp.json() == []


def test_api_presets_create(client: TestClient) -> None:
    resp = client.post(
        "/api/presets",
        json={"name": "High Aesthetic", "sort_by": "aesthetic:aesthetic", "min_score": 7.0},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "High Aesthetic"
    assert data["sort_by"] == "aesthetic:aesthetic"
    assert data["min_score"] == pytest.approx(7.0)
    assert "id" in data


def test_api_presets_list_after_create(client: TestClient) -> None:
    client.post("/api/presets", json={"name": "P1"})
    client.post("/api/presets", json={"name": "P2"})
    resp = client.get("/api/presets")
    names = [p["name"] for p in resp.json()]
    assert "P1" in names
    assert "P2" in names


def test_api_presets_upsert_updates_existing(client: TestClient) -> None:
    r1 = client.post("/api/presets", json={"name": "X", "sort_by": "blur:sharpness"})
    r2 = client.post("/api/presets", json={"name": "X", "sort_by": "aesthetic:aesthetic"})
    assert r1.json()["id"] == r2.json()["id"]
    assert r2.json()["sort_by"] == "aesthetic:aesthetic"


def test_api_presets_delete(client: TestClient) -> None:
    create_resp = client.post("/api/presets", json={"name": "To Delete"})
    preset_id = create_resp.json()["id"]
    del_resp = client.delete(f"/api/presets/{preset_id}")
    assert del_resp.status_code == 204
    list_resp = client.get("/api/presets")
    assert all(p["id"] != preset_id for p in list_resp.json())


def test_api_presets_delete_not_found(client: TestClient) -> None:
    resp = client.delete("/api/presets/99999")
    assert resp.status_code == 404


def test_api_presets_blank_name_rejected(client: TestClient) -> None:
    resp = client.post("/api/presets", json={"name": "   "})
    assert resp.status_code == 422


def test_browse_shows_preset_dropdown_when_presets_exist(client: TestClient) -> None:
    client.post("/api/presets", json={"name": "My Preset", "sort_by": "blur:sharpness"})
    resp = client.get("/assets")
    assert "My Preset" in resp.text


def test_browse_shows_save_preset_button_when_sort_active(
    client_with_scores: TestClient,
) -> None:
    resp = client_with_scores.get("/assets?sort_by=blur:sharpness")
    assert "Save as" in resp.text


# ── Unconfigured (setup) state ────────────────────────────────────────────────


@pytest.fixture()
def client_unconfigured(tmp_path: Path) -> TestClient:
    """Client with no DB connection, simulating the initial setup state."""
    app = create_app(library_root=None, db_conn=None)
    return TestClient(app, follow_redirects=False)


def test_clusters_returns_503_when_not_configured(client_unconfigured: TestClient) -> None:
    resp = client_unconfigured.get("/clusters")
    assert resp.status_code == 503


def test_cluster_detail_returns_503_when_not_configured(client_unconfigured: TestClient) -> None:
    resp = client_unconfigured.get("/clusters/1")
    assert resp.status_code == 503


def test_clusters_redirects_to_setup_for_html_browser(client_unconfigured: TestClient) -> None:
    """Browser navigation to /clusters while unconfigured should redirect to /setup."""
    resp = client_unconfigured.get(
        "/clusters", headers={"Accept": "text/html,application/xhtml+xml,*/*"}
    )
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/setup"


def test_assets_redirects_to_setup_for_html_browser(client_unconfigured: TestClient) -> None:
    """Browser navigation to /assets while unconfigured should redirect to /setup."""
    resp = client_unconfigured.get(
        "/assets", headers={"Accept": "text/html,application/xhtml+xml,*/*"}
    )
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/setup"


# ── deduplication (SHA-256) ───────────────────────────────────────────────────


def _add_asset_with_sha256(conn: sqlite3.Connection, relpath: str, sha256: str) -> int:
    return upsert_asset(
        conn,
        {
            "relpath": relpath,
            "filename": Path(relpath).name,
            "ext": Path(relpath).suffix.lower(),
            "size_bytes": 512,
            "mime": "image/jpeg",
            "sha256": sha256,
            "indexed_at": int(time.time()),
        },
    )


@pytest.fixture()
def client_with_duplicates(tmp_path: Path) -> TestClient:
    """DB with 3 files: two share a sha256 (duplicates) + one unique."""
    conn = _make_db()
    _add_asset_with_sha256(conn, "Photos/album1/img.jpg", "cafebabe")
    _add_asset_with_sha256(conn, "Photos/album2/img.jpg", "cafebabe")
    _add_asset_with_sha256(conn, "Photos/unique.jpg", "deadbeef")
    app = create_app(tmp_path, conn)
    return TestClient(app, follow_redirects=True)


def test_browse_dedupe_default_shows_unique_count(
    client_with_duplicates: TestClient,
) -> None:
    """Default (dedupe=1) should show 2 unique photos, not 3."""
    resp = client_with_duplicates.get("/assets")
    assert resp.status_code == 200
    assert "2" in resp.text
    assert "unique" in resp.text


def test_browse_dedupe_off_shows_all_files(client_with_duplicates: TestClient) -> None:
    """dedupe=0 should show all 3 physical files."""
    resp = client_with_duplicates.get("/assets?dedupe=0")
    assert resp.status_code == 200
    assert "3" in resp.text


def test_browse_dedupe_toggle_link_present(client_with_duplicates: TestClient) -> None:
    """Default view should include a link to show duplicates."""
    resp = client_with_duplicates.get("/assets")
    assert "dedupe=0" in resp.text or "Show duplicates" in resp.text


def test_browse_dedupe_off_toggle_link_present(client_with_duplicates: TestClient) -> None:
    """Non-dedupe view should include a link to hide duplicates."""
    resp = client_with_duplicates.get("/assets?dedupe=0")
    assert "Hide duplicates" in resp.text


def test_asset_detail_shows_duplicate_paths(tmp_path: Path) -> None:
    """Detail page for a duplicate image should list all physical paths."""
    conn = _make_db()
    id1 = _add_asset_with_sha256(conn, "Photos/album1/img.jpg", "cafebabe")
    _add_asset_with_sha256(conn, "Photos/album2/img.jpg", "cafebabe")
    app = create_app(tmp_path, conn)
    client = TestClient(app, follow_redirects=True)
    resp = client.get(f"/assets/{id1}")
    assert resp.status_code == 200
    # Both paths should appear in the detail page
    assert "album1/img.jpg" in resp.text
    assert "album2/img.jpg" in resp.text


def test_asset_detail_no_duplicate_section_for_unique(tmp_path: Path) -> None:
    """Detail page for a unique image should NOT show the duplicates section."""
    conn = _make_db()
    id1 = _add_asset_with_sha256(conn, "Photos/unique.jpg", "deadbeef")
    app = create_app(tmp_path, conn)
    client = TestClient(app, follow_redirects=True)
    resp = client.get(f"/assets/{id1}")
    assert resp.status_code == 200
    assert "Also stored at" not in resp.text
