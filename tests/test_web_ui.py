"""Tests for the FastAPI browse UI routes."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from takeout_rater.clustering.builder import build_clusters
from takeout_rater.db.queries import upsert_asset, upsert_phash
from takeout_rater.db.schema import migrate
from takeout_rater.ui.app import create_app


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


# ── Infinite scroll (partial endpoint) ───────────────────────────────────────


def test_browse_partial_returns_200(client_with_assets: TestClient) -> None:
    """GET /assets?partial=1 should return 200 with card fragment HTML."""
    resp = client_with_assets.get("/assets?partial=1")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_browse_partial_contains_cards(client_with_assets: TestClient) -> None:
    """Partial response should contain card elements (not a full page)."""
    resp = client_with_assets.get("/assets?partial=1")
    assert 'class="card"' in resp.text
    # Should NOT contain the full page chrome
    assert "<html" not in resp.text
    assert "<header" not in resp.text


def test_browse_partial_has_lightbox_data_attrs(client_with_assets: TestClient) -> None:
    """Partial card elements must carry data-lb-* attributes for the lightbox."""
    resp = client_with_assets.get("/assets?partial=1")
    assert "data-lb-id=" in resp.text
    assert "data-lb-src=" in resp.text
    assert "data-lb-title=" in resp.text


def test_browse_partial_no_sentinel_when_single_page(tmp_path: Path) -> None:
    """When all assets fit on one page, partial should not include a sentinel."""
    from takeout_rater.api.assets import _PAGE_SIZE  # noqa: PLC0415

    conn = _make_db()
    # Add fewer assets than one full page so total_pages == 1
    for i in range(min(3, _PAGE_SIZE - 1)):
        _add_asset(conn, f"Photos/img{i}.jpg")
    app = create_app(tmp_path, conn)
    client = TestClient(app, follow_redirects=True)
    resp = client.get("/assets?partial=1")
    assert "scroll-sentinel" not in resp.text
    # New design: partial carries metadata instead of a sentinel element
    assert "data-partial-meta" in resp.text
    assert 'data-total-pages="1"' in resp.text


def test_browse_partial_has_meta_when_multi_page(tmp_path: Path) -> None:
    """When there are more pages, partial response should carry the total-pages meta."""
    from takeout_rater.api.assets import _PAGE_SIZE  # noqa: PLC0415

    conn = _make_db()
    # Add more assets than one page can hold
    for i in range(_PAGE_SIZE + 1):
        _add_asset(conn, f"Photos/img{i}.jpg")
    app = create_app(tmp_path, conn)
    client = TestClient(app, follow_redirects=True)
    resp = client.get("/assets?partial=1&page=1")
    assert "scroll-sentinel" not in resp.text
    assert "data-partial-meta" in resp.text
    assert 'data-total-pages="2"' in resp.text


def test_browse_full_page_has_lightbox_data_attrs(client_with_assets: TestClient) -> None:
    """Full browse page cards must carry data-lb-* attributes."""
    resp = client_with_assets.get("/assets")
    assert "data-lb-id=" in resp.text
    assert "data-lb-src=" in resp.text
    assert "data-lb-title=" in resp.text


def test_browse_full_page_has_lightbox_markup(client_with_assets: TestClient) -> None:
    """Full browse page must include the lightbox overlay element."""
    resp = client_with_assets.get("/assets")
    assert 'id="lightbox"' in resp.text
    assert 'id="lb-img"' in resp.text
    assert 'id="lb-link"' in resp.text


def test_browse_full_page_no_pagination_nav(client_with_assets: TestClient) -> None:
    """Browse page should use infinite scroll instead of old pagination nav."""
    resp = client_with_assets.get("/assets")
    assert 'class="pagination"' not in resp.text


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


def test_asset_detail_partial_returns_fragment(client_with_assets: TestClient) -> None:
    """detail_partial.html should return HTML fragment without full page chrome."""
    resp = client_with_assets.get("/assets/1?partial=1")
    assert resp.status_code == 200
    assert "img0.jpg" in resp.text
    # Fragment must not include base-page elements
    assert "<html" not in resp.text
    assert "lb-detail-inner" in resp.text


def test_asset_detail_partial_not_found_returns_404(client: TestClient) -> None:
    resp = client.get("/assets/99999?partial=1")
    assert resp.status_code == 404


def test_browse_lightbox_has_details_panel(client_with_assets: TestClient) -> None:
    """Browse page must include the lightbox details panel."""
    resp = client_with_assets.get("/assets")
    assert 'id="lb-details"' in resp.text


# ── GET /thumbs/{id} ─────────────────────────────────────────────────────────


def test_thumbnail_not_found_returns_404(client: TestClient) -> None:
    resp = client.get("/thumbs/1")
    assert resp.status_code == 404


def test_thumbnail_serves_jpeg(tmp_path: Path) -> None:
    """When a thumbnail file exists, it should be served with image/jpeg content-type."""
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


# ── GET /clusterings ──────────────────────────────────────────────────────────


def test_clusterings_empty_db_returns_200(client: TestClient) -> None:
    resp = client.get("/clusterings")
    assert resp.status_code == 200


def test_clusterings_returns_html(client: TestClient) -> None:
    resp = client.get("/clusterings")
    assert "text/html" in resp.headers["content-type"]


def test_clusterings_empty_shows_no_clusters_message(client: TestClient) -> None:
    resp = client.get("/clusterings")
    assert "No clustering" in resp.text


def test_clusterings_shows_cluster_count(client_with_clusters: TestClient) -> None:
    resp = client_with_clusters.get("/clusterings")
    assert resp.status_code == 200
    assert "1 cluster" in resp.text


def test_clusterings_page2_returns_200(client_with_clusters: TestClient) -> None:
    # Extra query params are ignored; page should still return 200.
    resp = client_with_clusters.get("/clusterings?page=2")
    assert resp.status_code == 200


# ── GET /clusterings/{run_id} ────────────────────────────────────────────────


def test_clustering_detail_returns_200(client_with_clusters: TestClient) -> None:
    resp = client_with_clusters.get("/clusterings/1")
    assert resp.status_code == 200


def test_clustering_detail_not_found_returns_404(client: TestClient) -> None:
    resp = client.get("/clusterings/99999")
    assert resp.status_code == 404


def test_clustering_detail_shows_clusters(client_with_clusters: TestClient) -> None:
    resp = client_with_clusters.get("/clusterings/1")
    assert "a.jpg" in resp.text or "b.jpg" in resp.text


def test_clustering_detail_has_back_link(client_with_clusters: TestClient) -> None:
    resp = client_with_clusters.get("/clusterings/1")
    assert "/clusters" in resp.text


# ── DELETE /api/clusterings/{run_id} ─────────────────────────────────────────


def test_delete_clustering_run_returns_200(client_with_clusters: TestClient) -> None:
    resp = client_with_clusters.delete("/api/clusterings/1")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 1


def test_delete_clustering_run_not_found_returns_404(client: TestClient) -> None:
    resp = client.delete("/api/clusterings/99999")
    assert resp.status_code == 404


def test_delete_clustering_run_removes_from_list(client_with_clusters: TestClient) -> None:
    client_with_clusters.delete("/api/clusterings/1")
    resp = client_with_clusters.get("/clusterings")
    assert "No clustering" in resp.text


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


def test_cluster_detail_back_link_points_to_run(client_with_clusters: TestClient) -> None:
    resp = client_with_clusters.get("/clusters/1")
    assert "/clusterings/1" in resp.text


# ── nav link ─────────────────────────────────────────────────────────────────


def test_browse_page_has_clusters_nav_link(client: TestClient) -> None:
    resp = client.get("/assets")
    assert "/clusterings" in resp.text


# ── score range filter ────────────────────────────────────────────────────────


@pytest.fixture()
def client_with_scores(tmp_path: Path) -> TestClient:
    from takeout_rater.db.queries import upsert_asset_scores  # noqa: PLC0415

    conn = _make_db()
    ids = [_add_asset(conn, f"Photos/img{i}.jpg") for i in range(5)]
    upsert_asset_scores(
        conn,
        "blur",
        "default",
        [(aid, "sharpness", float(i * 20)) for i, aid in enumerate(ids)],
    )
    app = create_app(tmp_path, conn)
    return TestClient(app, follow_redirects=True)


def test_browse_with_sort_by_score_returns_200(client_with_scores: TestClient) -> None:
    resp = client_with_scores.get("/assets?sort_by=blur:default:sharpness")
    assert resp.status_code == 200


def test_browse_min_score_filter_reduces_count(client_with_scores: TestClient) -> None:
    client_with_scores.get("/assets?sort_by=blur:default:sharpness")
    resp_filtered = client_with_scores.get("/assets?sort_by=blur:default:sharpness&min_score=40")
    # Page with min_score should mention fewer photos
    assert resp_filtered.status_code == 200
    # Check that the filtered page doesn't have the full count
    assert "5 photos" not in resp_filtered.text


def test_browse_max_score_filter_reduces_count(client_with_scores: TestClient) -> None:
    resp_filtered = client_with_scores.get("/assets?sort_by=blur:default:sharpness&max_score=20")
    assert resp_filtered.status_code == 200
    assert "5 photos" not in resp_filtered.text


def test_browse_score_range_shows_filter_indicator(client_with_scores: TestClient) -> None:
    resp = client_with_scores.get("/assets?sort_by=blur:default:sharpness&min_score=40")
    assert "filtered by range" in resp.text


def test_browse_blank_min_score_returns_200(client_with_scores: TestClient) -> None:
    """Blank min_score= should not cause a 422 validation error."""
    resp = client_with_scores.get("/assets?sort_by=blur:default:sharpness&min_score=")
    assert resp.status_code == 200


def test_browse_blank_max_score_returns_200(client_with_scores: TestClient) -> None:
    """Blank max_score= should not cause a 422 validation error."""
    resp = client_with_scores.get("/assets?sort_by=blur:default:sharpness&max_score=")
    assert resp.status_code == 200


def test_browse_both_blank_scores_returns_200(client_with_scores: TestClient) -> None:
    """Both blank score params together should not cause a 422 validation error."""
    resp = client_with_scores.get("/assets?sort_by=blur:default:sharpness&min_score=&max_score=")
    assert resp.status_code == 200


def test_browse_blank_score_treats_as_unfiltered(client_with_scores: TestClient) -> None:
    """Blank min_score should produce the same result as omitting the parameter."""
    resp_no_filter = client_with_scores.get("/assets?sort_by=blur:default:sharpness")
    resp_blank = client_with_scores.get("/assets?sort_by=blur:default:sharpness&min_score=")
    assert resp_no_filter.status_code == 200
    assert resp_blank.status_code == 200
    # Both should show the same total count (blank is same as absent)
    assert "5 photos" in resp_blank.text


def test_browse_invalid_score_returns_200(client_with_scores: TestClient) -> None:
    """Non-numeric min_score should be silently ignored, not produce a 422."""
    resp = client_with_scores.get("/assets?sort_by=blur:default:sharpness&min_score=abc")
    assert resp.status_code == 200


def test_sort_options_only_include_scored_metrics(tmp_path: Path) -> None:
    """Sort dropdown should only list metrics that have scores."""
    conn = _make_db()
    for i in range(3):
        _add_asset(conn, f"Photos/img{i}.jpg")
    # No scores at all — dropdown should have no scorer options
    app = create_app(tmp_path, conn)
    client = TestClient(app, follow_redirects=True)
    resp = client.get("/assets")
    assert resp.status_code == 200
    # The sort dropdown should only have the default "date taken" option
    assert "simple:sharpness" not in resp.text


def test_sort_options_appear_after_scoring(tmp_path: Path) -> None:
    """Sort dropdown should include metrics once scores exist."""
    from takeout_rater.db.queries import upsert_asset_scores  # noqa: PLC0415

    conn = _make_db()
    ids = [_add_asset(conn, f"Photos/img{i}.jpg") for i in range(3)]
    upsert_asset_scores(
        conn,
        "simple",
        "blur",
        [(aid, "sharpness", float(i * 20)) for i, aid in enumerate(ids)],
    )
    app = create_app(tmp_path, conn)
    client = TestClient(app, follow_redirects=True)
    resp = client.get("/assets")
    assert resp.status_code == 200
    assert "simple:blur:sharpness" in resp.text


def test_sort_by_unscored_metric_shows_helpful_message(tmp_path: Path) -> None:
    """Manually requesting an unscored metric should show a helpful message, not 'No photos indexed'."""
    conn = _make_db()
    for i in range(3):
        _add_asset(conn, f"Photos/img{i}.jpg")
    app = create_app(tmp_path, conn)
    client = TestClient(app, follow_redirects=True)
    # Force sort_by via URL for a metric with no scores
    resp = client.get("/assets?sort_by=blur:default:sharpness")
    assert resp.status_code == 200
    assert "No photos indexed yet" not in resp.text
    assert "No scored photos for this metric" in resp.text


# ── preset API ────────────────────────────────────────────────────────────────


def test_api_presets_list_empty(client: TestClient) -> None:
    resp = client.get("/api/presets")
    assert resp.status_code == 200
    assert resp.json() == []


def test_api_presets_create(client: TestClient) -> None:
    resp = client.post(
        "/api/presets",
        json={
            "name": "High Aesthetic",
            "sort_by": "aesthetic:default:aesthetic_quality",
            "min_score": 7.0,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "High Aesthetic"
    assert data["sort_by"] == "aesthetic:default:aesthetic_quality"
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
    r1 = client.post("/api/presets", json={"name": "X", "sort_by": "blur:default:sharpness"})
    r2 = client.post(
        "/api/presets", json={"name": "X", "sort_by": "aesthetic:default:aesthetic_quality"}
    )
    assert r1.json()["id"] == r2.json()["id"]
    assert r2.json()["sort_by"] == "aesthetic:default:aesthetic_quality"


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
    client.post("/api/presets", json={"name": "My Preset", "sort_by": "blur:default:sharpness"})
    resp = client.get("/assets")
    assert "My Preset" in resp.text


def test_browse_shows_save_preset_button_when_sort_active(
    client_with_scores: TestClient,
) -> None:
    resp = client_with_scores.get("/assets?sort_by=blur:default:sharpness")
    assert "Save as" in resp.text


# ── Unconfigured (setup) state ────────────────────────────────────────────────


@pytest.fixture()
def client_unconfigured(tmp_path: Path) -> TestClient:
    """Client with no DB connection, simulating the initial setup state."""
    app = create_app(library_root=None, db_conn=None)
    return TestClient(app, follow_redirects=False)


def test_clusterings_returns_503_when_not_configured(client_unconfigured: TestClient) -> None:
    resp = client_unconfigured.get("/clusterings")
    assert resp.status_code == 503


def test_cluster_detail_returns_503_when_not_configured(client_unconfigured: TestClient) -> None:
    resp = client_unconfigured.get("/clusters/1")
    assert resp.status_code == 503


def test_clustering_detail_returns_503_when_not_configured(
    client_unconfigured: TestClient,
) -> None:
    resp = client_unconfigured.get("/clusterings/1")
    assert resp.status_code == 503


def test_clusterings_redirects_to_setup_for_html_browser(
    client_unconfigured: TestClient,
) -> None:
    """Browser navigation to /clusterings while unconfigured should redirect to /setup."""
    resp = client_unconfigured.get(
        "/clusterings", headers={"Accept": "text/html,application/xhtml+xml,*/*"}
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


def test_browse_dedupe_toggle_removed(client_with_duplicates: TestClient) -> None:
    """Dedup now happens at write time; the Show/Hide duplicates toggle is gone."""
    resp = client_with_duplicates.get("/assets")
    assert resp.status_code == 200
    assert "Show duplicates" not in resp.text
    assert "Hide duplicates" not in resp.text


def test_browse_dedupe_off_toggle_removed(client_with_duplicates: TestClient) -> None:
    """Even with dedupe=0, the Hide duplicates toggle no longer appears."""
    resp = client_with_duplicates.get("/assets?dedupe=0")
    assert resp.status_code == 200
    assert "Hide duplicates" not in resp.text


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


def test_asset_detail_shows_sidecar_json_panel(tmp_path: Path) -> None:
    """Detail page should show the raw metadata panel when a sidecar file exists."""
    conn = _make_db()
    # Create a real sidecar file in the tmp directory
    sidecar_relpath = "Photos/img.jpg.supplemental-metadata.json"
    sidecar_path = tmp_path / sidecar_relpath
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(
        '{"title":"img.jpg","description":"","url":"https://photos.google.com/x",'
        '"creationTime":{"timestamp":"1686836581"},"imageViews":"7"}',
        encoding="utf-8",
    )
    asset_id = upsert_asset(
        conn,
        {
            "relpath": "Photos/img.jpg",
            "filename": "img.jpg",
            "ext": ".jpg",
            "size_bytes": 1024,
            "mime": "image/jpeg",
            "sidecar_relpath": sidecar_relpath,
            "indexed_at": int(time.time()),
        },
    )
    app = create_app(tmp_path, conn)
    client = TestClient(app, follow_redirects=True)
    resp = client.get(f"/assets/{asset_id}")
    assert resp.status_code == 200
    assert "imageViews" in resp.text
    assert "https://photos.google.com/x" in resp.text


def test_asset_detail_no_sidecar_panel_when_missing(tmp_path: Path) -> None:
    """Detail page should show 'No sidecar JSON available' when no sidecar is indexed."""
    conn = _make_db()
    asset_id = _add_asset(conn, "Photos/nosidecar.jpg")
    app = create_app(tmp_path, conn)
    client = TestClient(app, follow_redirects=True)
    resp = client.get(f"/assets/{asset_id}")
    assert resp.status_code == 200
    assert "No sidecar JSON available" in resp.text


def test_asset_detail_shows_only_canonical_sidecar(tmp_path: Path) -> None:
    """Detail page for a deduplicated image shows the canonical sidecar only.

    The alias path (album2) is listed in the 'Also stored at' section but its
    sidecar JSON is not loaded separately — only the canonical asset's sidecar
    appears.
    """
    conn = _make_db()

    # Create sidecar for the canonical copy
    sidecar1 = "Photos/album1/img.jpg.supplemental-metadata.json"
    sidecar1_path = tmp_path / sidecar1
    sidecar1_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar1_path.write_text(
        '{"title":"img.jpg","description":"album1 copy","url":"",'
        '"creationTime":{"timestamp":"1000"},"imageViews":"1"}',
        encoding="utf-8",
    )

    id1 = upsert_asset(
        conn,
        {
            "relpath": "Photos/album1/img.jpg",
            "filename": "img.jpg",
            "ext": ".jpg",
            "size_bytes": 512,
            "mime": "image/jpeg",
            "sha256": "aabbccdd",
            "sidecar_relpath": sidecar1,
            "indexed_at": int(time.time()),
        },
    )
    # This is a binary duplicate — goes to asset_paths, not a new assets row.
    upsert_asset(
        conn,
        {
            "relpath": "Photos/album2/img.jpg",
            "filename": "img.jpg",
            "ext": ".jpg",
            "size_bytes": 512,
            "mime": "image/jpeg",
            "sha256": "aabbccdd",
            "indexed_at": int(time.time()),
        },
    )

    app = create_app(tmp_path, conn)
    client = TestClient(app, follow_redirects=True)
    resp = client.get(f"/assets/{id1}")
    assert resp.status_code == 200
    # Canonical sidecar content is shown
    assert "album1 copy" in resp.text
    # Alias path is listed in 'Also stored at'
    assert "album2/img.jpg" in resp.text


# ── EXIF data display ─────────────────────────────────────────────────────────


def _make_jpeg_with_exif(path: Path, make: str = "TestCamera", model: str = "TestModel X1") -> None:
    """Create a minimal JPEG at *path* with Make and Model EXIF tags."""
    from PIL import ExifTags  # noqa: PLC0415
    from PIL import Image as _Image

    img = _Image.new("RGB", (4, 4), color=(100, 150, 200))
    exif = img.getexif()
    exif[ExifTags.Base.Make] = make
    exif[ExifTags.Base.Model] = model
    img.save(path, "JPEG", exif=exif.tobytes())


def test_asset_detail_shows_exif_data(tmp_path: Path) -> None:
    """Detail page should display EXIF tags when the image file has EXIF data."""
    conn = _make_db()
    relpath = "Photos/img.jpg"
    img_path = tmp_path / relpath
    img_path.parent.mkdir(parents=True, exist_ok=True)
    _make_jpeg_with_exif(img_path, make="CoolBrand", model="PhotoPro 9")
    asset_id = upsert_asset(
        conn,
        {
            "relpath": relpath,
            "filename": "img.jpg",
            "ext": ".jpg",
            "size_bytes": img_path.stat().st_size,
            "mime": "image/jpeg",
            "indexed_at": int(time.time()),
        },
    )
    app = create_app(tmp_path, conn)
    client = TestClient(app, follow_redirects=True)
    resp = client.get(f"/assets/{asset_id}")
    assert resp.status_code == 200
    assert "CoolBrand" in resp.text
    assert "PhotoPro 9" in resp.text
    assert "No EXIF data available" not in resp.text


def test_asset_detail_shows_exif_data_partial(tmp_path: Path) -> None:
    """Lightbox partial should also display EXIF data when the image has EXIF tags."""
    conn = _make_db()
    relpath = "Photos/img.jpg"
    img_path = tmp_path / relpath
    img_path.parent.mkdir(parents=True, exist_ok=True)
    _make_jpeg_with_exif(img_path, make="SnapBrand", model="QuickShot Z")
    asset_id = upsert_asset(
        conn,
        {
            "relpath": relpath,
            "filename": "img.jpg",
            "ext": ".jpg",
            "size_bytes": img_path.stat().st_size,
            "mime": "image/jpeg",
            "indexed_at": int(time.time()),
        },
    )
    app = create_app(tmp_path, conn)
    client = TestClient(app, follow_redirects=True)
    resp = client.get(f"/assets/{asset_id}?partial=1")
    assert resp.status_code == 200
    assert "SnapBrand" in resp.text
    assert "QuickShot Z" in resp.text
    assert "No EXIF data available" not in resp.text


def test_asset_detail_no_exif_when_image_missing(tmp_path: Path) -> None:
    """Detail page should show 'No EXIF data available' when the image file does not exist."""
    conn = _make_db()
    asset_id = _add_asset(conn, "Photos/ghost.jpg")
    app = create_app(tmp_path, conn)
    client = TestClient(app, follow_redirects=True)
    resp = client.get(f"/assets/{asset_id}")
    assert resp.status_code == 200
    assert "No EXIF data available" in resp.text


def test_asset_detail_no_exif_when_image_has_none(tmp_path: Path) -> None:
    """Detail page shows 'No EXIF data available' when the JPEG contains no EXIF metadata."""
    from PIL import Image as _Image  # noqa: PLC0415

    conn = _make_db()
    relpath = "Photos/no_exif.jpg"
    img_path = tmp_path / relpath
    img_path.parent.mkdir(parents=True, exist_ok=True)
    # Save without EXIF — the resulting file has no Exif APP1 segment.
    _Image.new("RGB", (2, 2)).save(str(img_path), "JPEG")
    asset_id = upsert_asset(
        conn,
        {
            "relpath": relpath,
            "filename": "no_exif.jpg",
            "ext": ".jpg",
            "size_bytes": img_path.stat().st_size,
            "mime": "image/jpeg",
            "indexed_at": int(time.time()),
        },
    )
    app = create_app(tmp_path, conn)
    client = TestClient(app, follow_redirects=True)
    resp = client.get(f"/assets/{asset_id}")
    assert resp.status_code == 200
    assert "No EXIF data available" in resp.text


def test_asset_detail_sidecar_with_nested_google_photos_dir(tmp_path: Path) -> None:
    """Detail page should find sidecar when library has a Takeout/Google Photos structure.

    Relpaths are stored relative to the Google Photos root, but app.state.takeout_root
    must point to the same root (not to library_root) for file reads to succeed.
    """
    conn = _make_db()
    # Simulate the real on-disk structure: library_root/Takeout/Google Photos/...
    google_photos = tmp_path / "Takeout" / "Google Photos"
    sidecar_relpath = "Photos from 2026/img.jpg.supplemental-metadata.json"
    sidecar_path = google_photos / sidecar_relpath
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(
        '{"title":"img.jpg","description":"","url":"https://photos.google.com/y",'
        '"creationTime":{"timestamp":"1771361057"},"imageViews":"3"}',
        encoding="utf-8",
    )
    asset_id = upsert_asset(
        conn,
        {
            "relpath": "Photos from 2026/img.jpg",
            "filename": "img.jpg",
            "ext": ".jpg",
            "size_bytes": 1024,
            "mime": "image/jpeg",
            "sidecar_relpath": sidecar_relpath,
            "indexed_at": int(time.time()),
        },
    )
    # Pass library_root (parent of Takeout/) — app must resolve photos_root itself.
    app = create_app(tmp_path, conn)
    client = TestClient(app, follow_redirects=True)
    resp = client.get(f"/assets/{asset_id}")
    assert resp.status_code == 200
    assert "imageViews" in resp.text
    assert "https://photos.google.com/y" in resp.text


def test_asset_detail_sidecar_partial_with_nested_google_photos_dir(tmp_path: Path) -> None:
    """Lightbox partial should also find sidecar with nested Takeout/Google Photos structure."""
    conn = _make_db()
    google_photos = tmp_path / "Takeout" / "Google Photos"
    sidecar_relpath = "Photos from 2026/img.jpg.supplemental-metadata.json"
    sidecar_path = google_photos / sidecar_relpath
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(
        '{"title":"img.jpg","description":"","url":"https://photos.google.com/z",'
        '"creationTime":{"timestamp":"1771361057"},"imageViews":"5"}',
        encoding="utf-8",
    )
    asset_id = upsert_asset(
        conn,
        {
            "relpath": "Photos from 2026/img.jpg",
            "filename": "img.jpg",
            "ext": ".jpg",
            "size_bytes": 1024,
            "mime": "image/jpeg",
            "sidecar_relpath": sidecar_relpath,
            "indexed_at": int(time.time()),
        },
    )
    app = create_app(tmp_path, conn)
    client = TestClient(app, follow_redirects=True)
    resp = client.get(f"/assets/{asset_id}?partial=1")
    assert resp.status_code == 200
    assert "imageViews" in resp.text
    assert "https://photos.google.com/z" in resp.text


def test_timeline_no_data_returns_has_data_false(client: TestClient) -> None:
    """With no indexed photos the timeline endpoint reports has_data=False."""
    resp = client.get("/api/timeline")
    assert resp.status_code == 200
    data = resp.json()
    assert data["has_data"] is False


def test_timeline_returns_year_range(tmp_path: Path) -> None:
    """Timeline endpoint should report the correct min/max years from taken_at."""
    import calendar

    conn = _make_db()
    # 2018-06-15
    ts_2018 = int(calendar.timegm((2018, 6, 15, 0, 0, 0, 0, 0, 0)))
    # 2022-01-01
    ts_2022 = int(calendar.timegm((2022, 1, 1, 0, 0, 0, 0, 0, 0)))

    from takeout_rater.db.queries import upsert_asset  # noqa: F811

    upsert_asset(
        conn,
        {
            "relpath": "a/img1.jpg",
            "filename": "img1.jpg",
            "ext": ".jpg",
            "size_bytes": 1,
            "mime": "image/jpeg",
            "indexed_at": int(time.time()),
            "taken_at": ts_2018,
        },
    )
    upsert_asset(
        conn,
        {
            "relpath": "a/img2.jpg",
            "filename": "img2.jpg",
            "ext": ".jpg",
            "size_bytes": 1,
            "mime": "image/jpeg",
            "indexed_at": int(time.time()),
            "taken_at": ts_2022,
        },
    )

    app = create_app(tmp_path, conn)
    cl = TestClient(app, follow_redirects=True)
    resp = cl.get("/api/timeline")
    assert resp.status_code == 200
    data = resp.json()
    assert data["has_data"] is True
    assert data["min_year"] == 2018
    assert data["max_year"] == 2022


# ── GET /api/timeline/seek ────────────────────────────────────────────────────


def test_timeline_seek_returns_page_1_for_newest_timestamp(tmp_path: Path) -> None:
    """Seeking to the newest photo's timestamp should return page 1."""
    import calendar

    conn = _make_db()
    ts_new = int(calendar.timegm((2023, 1, 1, 0, 0, 0, 0, 0, 0)))
    ts_old = int(calendar.timegm((2010, 1, 1, 0, 0, 0, 0, 0, 0)))

    from takeout_rater.db.queries import upsert_asset  # noqa: F811

    upsert_asset(
        conn,
        {
            "relpath": "a/new.jpg",
            "filename": "new.jpg",
            "ext": ".jpg",
            "size_bytes": 1,
            "mime": "image/jpeg",
            "indexed_at": int(time.time()),
            "taken_at": ts_new,
        },
    )
    upsert_asset(
        conn,
        {
            "relpath": "a/old.jpg",
            "filename": "old.jpg",
            "ext": ".jpg",
            "size_bytes": 1,
            "mime": "image/jpeg",
            "indexed_at": int(time.time()),
            "taken_at": ts_old,
        },
    )

    app = create_app(tmp_path, conn)
    cl = TestClient(app, follow_redirects=True)

    # Seeking to a time before all photos → no photos are newer → page 1
    resp = cl.get(f"/api/timeline/seek?timestamp={ts_old - 1}")
    assert resp.status_code == 200
    assert resp.json()["page"] == 1

    # Seeking to a very recent timestamp → all photos are older → page 1
    resp2 = cl.get(f"/api/timeline/seek?timestamp={ts_new + 1}")
    assert resp2.status_code == 200
    assert resp2.json()["page"] == 1


def test_timeline_seek_page_advances_for_older_timestamp(tmp_path: Path) -> None:
    """Seeking to an older timestamp should return a later page than a recent one."""
    import calendar

    conn = _make_db()

    from takeout_rater.db.queries import upsert_asset  # noqa: F811

    # Insert 60 photos spread across two years so they span two pages (page size 50).
    for i in range(60):
        year = 2020 if i < 30 else 2019
        ts = int(calendar.timegm((year, 6, i % 28 + 1, 0, 0, 0, 0, 0, 0)))
        upsert_asset(
            conn,
            {
                "relpath": f"a/img{i}.jpg",
                "filename": f"img{i}.jpg",
                "ext": ".jpg",
                "size_bytes": 1,
                "mime": "image/jpeg",
                "indexed_at": int(time.time()),
                "taken_at": ts,
            },
        )

    app = create_app(tmp_path, conn)
    cl = TestClient(app, follow_redirects=True)

    ts_recent = int(calendar.timegm((2020, 7, 1, 0, 0, 0, 0, 0, 0)))
    ts_older = int(calendar.timegm((2019, 1, 1, 0, 0, 0, 0, 0, 0)))

    resp_recent = cl.get(f"/api/timeline/seek?timestamp={ts_recent}")
    resp_older = cl.get(f"/api/timeline/seek?timestamp={ts_older}")
    assert resp_recent.status_code == 200
    assert resp_older.status_code == 200
    # Older timestamp → more photos are newer → higher page number
    assert resp_older.json()["page"] >= resp_recent.json()["page"]


# ── GET /albums ───────────────────────────────────────────────────────────────


@pytest.fixture()
def client_with_albums(tmp_path: Path) -> TestClient:
    from takeout_rater.db.queries import link_asset_to_album, upsert_album  # noqa: PLC0415

    conn = _make_db()
    id1 = _add_asset(conn, "Vacation 2023/img1.jpg")
    id2 = _add_asset(conn, "Vacation 2023/img2.jpg")
    id3 = _add_asset(conn, "Birthday Party/cake.jpg")
    album1 = upsert_album(conn, "Vacation 2023", "Vacation 2023")
    album2 = upsert_album(conn, "Birthday Party", "Birthday Party")
    link_asset_to_album(conn, album1, id1)
    link_asset_to_album(conn, album1, id2)
    link_asset_to_album(conn, album2, id3)
    app = create_app(tmp_path, conn)
    return TestClient(app, follow_redirects=True)


def test_albums_empty_db_returns_200(client: TestClient) -> None:
    resp = client.get("/albums")
    assert resp.status_code == 200


def test_albums_returns_html(client: TestClient) -> None:
    resp = client.get("/albums")
    assert "text/html" in resp.headers["content-type"]


def test_albums_empty_shows_no_albums_message(client: TestClient) -> None:
    resp = client.get("/albums")
    assert "No albums" in resp.text


def test_albums_shows_album_names(client_with_albums: TestClient) -> None:
    resp = client_with_albums.get("/albums")
    assert resp.status_code == 200
    assert "Vacation 2023" in resp.text
    assert "Birthday Party" in resp.text


def test_albums_shows_photo_counts(client_with_albums: TestClient) -> None:
    resp = client_with_albums.get("/albums")
    assert "2 photos" in resp.text
    assert "1 photo" in resp.text


def test_albums_nav_link_present(client: TestClient) -> None:
    resp = client.get("/assets")
    assert "/albums" in resp.text


# ── GET /albums/{album_id} ────────────────────────────────────────────────────


def test_album_detail_returns_200(client_with_albums: TestClient) -> None:
    resp = client_with_albums.get("/albums/1")
    assert resp.status_code == 200


def test_album_detail_not_found_returns_404(client: TestClient) -> None:
    resp = client.get("/albums/99999")
    assert resp.status_code == 404


def test_album_detail_shows_album_name(client_with_albums: TestClient) -> None:
    resp = client_with_albums.get("/albums/1")
    assert "Vacation 2023" in resp.text


def test_album_detail_shows_photos(client_with_albums: TestClient) -> None:
    resp = client_with_albums.get("/albums/1")
    assert "img1.jpg" in resp.text or "img2.jpg" in resp.text


def test_album_detail_has_lightbox_markup(client_with_albums: TestClient) -> None:
    resp = client_with_albums.get("/albums/1")
    assert 'id="lightbox"' in resp.text
    assert 'id="lb-img"' in resp.text
    assert 'id="lb-details"' in resp.text


def test_album_detail_cards_have_lightbox_data_attrs(client_with_albums: TestClient) -> None:
    resp = client_with_albums.get("/albums/1")
    assert "data-lb-id=" in resp.text
    assert "data-lb-src=" in resp.text


def test_album_detail_has_back_link(client_with_albums: TestClient) -> None:
    resp = client_with_albums.get("/albums/1")
    assert "/albums" in resp.text


def test_album_detail_page2_returns_200(client_with_albums: TestClient) -> None:
    resp = client_with_albums.get("/albums/1?page=2")
    assert resp.status_code == 200


# ── GET /api/assets/{id}/similar ─────────────────────────────────────────────


def test_similar_assets_not_found_returns_404(client: TestClient) -> None:
    resp = client.get("/api/assets/99999/similar")
    assert resp.status_code == 404


def test_similar_assets_no_embedding_returns_empty_with_error(
    client_with_assets: TestClient,
) -> None:
    """Asset exists but has no CLIP embedding → empty results with error key."""
    resp = client_with_assets.get("/api/assets/1/similar")
    assert resp.status_code == 200
    data = resp.json()
    assert data["asset_id"] == 1
    assert data["results"] == []
    assert data.get("error") == "no_embedding"


def test_similar_assets_no_phash_returns_empty_with_error(
    client_with_assets: TestClient,
) -> None:
    """Asset exists but has no pHash → empty results with no_phash error."""
    resp = client_with_assets.get("/api/assets/1/similar?method=phash")
    assert resp.status_code == 200
    data = resp.json()
    assert data["asset_id"] == 1
    assert data["results"] == []
    assert data.get("error") == "no_phash"
    assert data["method"] == "phash"


def test_similar_assets_with_embedding_returns_results(tmp_path: Path) -> None:
    """When CLIP embeddings exist, the endpoint returns similar assets."""
    import struct  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    from takeout_rater.db.queries import bulk_upsert_clip_embeddings  # noqa: PLC0415

    DIM = 768
    rng = np.random.default_rng(0)

    conn = _make_db()

    # Add two assets
    id1 = _add_asset(conn, "Photos/a.jpg")
    id2 = _add_asset(conn, "Photos/b.jpg")

    # Create a reference embedding for asset 1 and a very similar one for asset 2
    base = rng.standard_normal(DIM).astype(np.float32)
    base /= np.linalg.norm(base)
    similar = base + rng.standard_normal(DIM).astype(np.float32) * 0.01
    similar /= np.linalg.norm(similar)

    blob1 = struct.pack(f"{DIM}f", *base)
    blob2 = struct.pack(f"{DIM}f", *similar)

    bulk_upsert_clip_embeddings(conn, [(id1, blob1), (id2, blob2)])

    app = create_app(tmp_path, conn)
    from fastapi.testclient import TestClient as TC  # noqa: PLC0415

    client = TC(app)

    resp = client.get(f"/api/assets/{id1}/similar?method=clip&metric=cosine&threshold=0.80")
    assert resp.status_code == 200
    data = resp.json()
    assert data["asset_id"] == id1
    assert data["method"] == "clip"
    assert data["metric"] == "cosine"
    assert len(data["results"]) >= 1
    result = data["results"][0]
    assert result["asset_id"] == id2
    assert result["score"] >= 0.80
    assert "taken_at" in result
    assert result["filename"] == "b.jpg"


def test_similar_assets_euclidean_metric(tmp_path: Path) -> None:
    """CLIP euclidean metric returns distance score (lower = more similar)."""
    import struct  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    from takeout_rater.db.queries import bulk_upsert_clip_embeddings  # noqa: PLC0415

    DIM = 768
    rng = np.random.default_rng(1)

    conn = _make_db()
    id1 = _add_asset(conn, "Photos/a.jpg")
    id2 = _add_asset(conn, "Photos/b.jpg")

    base = rng.standard_normal(DIM).astype(np.float32)
    base /= np.linalg.norm(base)
    close = base + rng.standard_normal(DIM).astype(np.float32) * 0.01
    close /= np.linalg.norm(close)

    bulk_upsert_clip_embeddings(conn, [(id1, struct.pack(f"{DIM}f", *base)), (id2, struct.pack(f"{DIM}f", *close))])

    app = create_app(tmp_path, conn)
    from fastapi.testclient import TestClient as TC  # noqa: PLC0415

    client = TC(app)

    resp = client.get(f"/api/assets/{id1}/similar?method=clip&metric=euclidean&threshold=0.45")
    assert resp.status_code == 200
    data = resp.json()
    assert data["metric"] == "euclidean"
    assert len(data["results"]) >= 1
    # score is L2 distance; very similar embeddings should have small distance
    assert data["results"][0]["score"] < 0.45


def test_similar_assets_phash_returns_results(tmp_path: Path) -> None:
    """When pHash values exist, phash method returns similar assets."""
    conn = _make_db()
    id1 = _add_asset(conn, "Photos/a.jpg")
    id2 = _add_asset(conn, "Photos/b.jpg")

    # Identical hashes → Hamming distance 0
    upsert_phash(conn, id1, "0" * 64)
    upsert_phash(conn, id2, "0" * 63 + "1")  # last hex digit 0→1 = 1 bit change

    app = create_app(tmp_path, conn)
    from fastapi.testclient import TestClient as TC  # noqa: PLC0415

    client = TC(app)

    resp = client.get(f"/api/assets/{id1}/similar?method=phash&threshold=10")
    assert resp.status_code == 200
    data = resp.json()
    assert data["method"] == "phash"
    assert data["metric"] is None
    assert len(data["results"]) == 1
    assert data["results"][0]["asset_id"] == id2
    assert data["results"][0]["score"] == 1  # 1 bit differs


def test_similar_assets_threshold_filters_results(tmp_path: Path) -> None:
    """Using a very high threshold should exclude moderately similar assets."""
    import struct  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    from takeout_rater.db.queries import bulk_upsert_clip_embeddings  # noqa: PLC0415

    DIM = 768
    rng = np.random.default_rng(42)

    conn = _make_db()
    id1 = _add_asset(conn, "Photos/ref.jpg")
    id2 = _add_asset(conn, "Photos/similar.jpg")

    base = rng.standard_normal(DIM).astype(np.float32)
    base /= np.linalg.norm(base)
    moderate = base + rng.standard_normal(DIM).astype(np.float32) * 0.3
    moderate /= np.linalg.norm(moderate)

    blob1 = struct.pack(f"{DIM}f", *base)
    blob2 = struct.pack(f"{DIM}f", *moderate)
    bulk_upsert_clip_embeddings(conn, [(id1, blob1), (id2, blob2)])

    app = create_app(tmp_path, conn)
    from fastapi.testclient import TestClient as TC  # noqa: PLC0415

    client = TC(app)

    # Very high threshold — should exclude the moderately similar asset
    resp = client.get(f"/api/assets/{id1}/similar?threshold=0.99")
    assert resp.status_code == 200
    data = resp.json()
    assert data["results"] == []


def test_asset_detail_has_more_like_this_panel(client_with_assets: TestClient) -> None:
    """The asset detail page must include the 'More like this' panel."""
    resp = client_with_assets.get("/assets/1")
    assert resp.status_code == 200
    assert "mlt-panel" in resp.text
    assert "mlt-method" in resp.text
    assert "More like this" in resp.text
