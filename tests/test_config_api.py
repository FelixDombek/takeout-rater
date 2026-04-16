"""Tests for the config API endpoints and health probe."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from takeout_rater.ui.app import create_app


@pytest.fixture()
def client() -> TestClient:
    """App with no library configured (unconfigured mode)."""
    app = create_app(None, None)
    return TestClient(app, follow_redirects=False)


@pytest.fixture()
def client_configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """App that has a config file pointing at *tmp_path*."""
    cfg_file = tmp_path / ".takeout-rater.json"
    cfg_file.write_text(json.dumps({"photos_root": str(tmp_path)}))

    import takeout_rater.config as cfg_mod  # noqa: E402

    monkeypatch.setattr(cfg_mod, "_CONFIG_FILE", cfg_file)

    import takeout_rater.api.config_routes as routes_mod  # noqa: E402

    monkeypatch.setattr(routes_mod, "get_photos_root", cfg_mod.get_photos_root)
    monkeypatch.setattr(routes_mod, "set_photos_root", cfg_mod.set_photos_root)
    monkeypatch.setattr(routes_mod, "get_db_root", cfg_mod.get_db_root)
    monkeypatch.setattr(routes_mod, "set_db_root", cfg_mod.set_db_root)

    app = create_app(None, None)
    return TestClient(app, follow_redirects=False)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


def test_health_returns_200(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_returns_json_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# GET /api/config
# ---------------------------------------------------------------------------


def test_get_config_unconfigured(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import takeout_rater.api.config_routes as routes_mod  # noqa: E402

    monkeypatch.setattr(routes_mod, "get_photos_root", lambda: None)
    monkeypatch.setattr(routes_mod, "get_db_root", lambda: None)

    resp = client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["configured"] is False
    assert data["takeout_path"] is None
    assert data["photos_root"] is None


def test_get_config_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import takeout_rater.api.config_routes as routes_mod  # noqa: E402

    monkeypatch.setattr(routes_mod, "get_photos_root", lambda: tmp_path)
    monkeypatch.setattr(routes_mod, "get_db_root", lambda: None)

    resp = client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["configured"] is True
    assert data["takeout_path"] == str(tmp_path)
    assert data["photos_root"] == str(tmp_path)


def test_get_config_includes_db_root(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import takeout_rater.api.config_routes as routes_mod  # noqa: E402

    db_root = tmp_path / "state"
    monkeypatch.setattr(routes_mod, "get_photos_root", lambda: tmp_path)
    monkeypatch.setattr(routes_mod, "get_db_root", lambda: db_root)

    resp = client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["db_root"] == str(db_root)


# ---------------------------------------------------------------------------
# GET /api/library/status
# ---------------------------------------------------------------------------


def test_library_status_unconfigured(client: TestClient) -> None:
    resp = client.get("/api/library/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["configured"] is False
    assert data["library_path"] is None
    assert data["db_schema_version"] is None


def test_library_status_scan_version_present(client: TestClient) -> None:
    from takeout_rater.db.queries import CURRENT_INDEXER_VERSION  # noqa: E402

    resp = client.get("/api/library/status")
    assert resp.status_code == 200
    assert resp.json()["db_scan_version"] == CURRENT_INDEXER_VERSION


def test_library_status_configured(tmp_path: Path) -> None:
    from takeout_rater.db.connection import open_library_db  # noqa: E402
    from takeout_rater.db.queries import CURRENT_INDEXER_VERSION  # noqa: E402

    conn = open_library_db(tmp_path)
    app = create_app(tmp_path, conn)
    c = TestClient(app, follow_redirects=False)

    resp = c.get("/api/library/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["configured"] is True
    assert data["library_path"] == str(tmp_path)
    assert isinstance(data["db_schema_version"], int)
    assert data["db_schema_version"] > 0
    assert data["db_scan_version"] == CURRENT_INDEXER_VERSION
    conn.close()


# ---------------------------------------------------------------------------
# POST /api/config/takeout-path
# ---------------------------------------------------------------------------


def test_set_path_valid(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    saved: list[Path] = []

    import takeout_rater.api.config_routes as routes_mod  # noqa: E402

    monkeypatch.setattr(routes_mod, "set_photos_root", lambda p: saved.append(p))
    monkeypatch.setattr(routes_mod, "set_db_root", lambda p: None)

    resp = client.post("/api/config/takeout-path", json={"path": str(tmp_path)})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert len(saved) == 1
    assert saved[0] == tmp_path.resolve()


def test_set_path_with_db_root(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Supplying db_root stores it separately from photos_root."""
    photos_saved: list[Path] = []
    db_saved: list[Path | None] = []

    db_root_dir = tmp_path / "state"
    db_root_dir.mkdir()

    import takeout_rater.api.config_routes as routes_mod  # noqa: E402

    monkeypatch.setattr(routes_mod, "set_photos_root", lambda p: photos_saved.append(p))
    monkeypatch.setattr(routes_mod, "set_db_root", lambda p: db_saved.append(p))

    resp = client.post(
        "/api/config/takeout-path",
        json={"path": str(tmp_path), "db_root": str(db_root_dir)},
    )
    assert resp.status_code == 200
    assert len(photos_saved) == 1
    assert len(db_saved) == 1
    assert db_saved[0] == db_root_dir.resolve()


def test_set_path_nonexistent_returns_400(client: TestClient) -> None:
    resp = client.post("/api/config/takeout-path", json={"path": "/this/path/does/not/exist/xyz"})
    assert resp.status_code == 400


def test_set_path_not_a_directory(client: TestClient, tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("hello")
    resp = client.post("/api/config/takeout-path", json={"path": str(f)})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Root redirect when unconfigured
# ---------------------------------------------------------------------------


def test_root_redirects_to_setup_when_unconfigured(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/setup"


# ---------------------------------------------------------------------------
# Setup page
# ---------------------------------------------------------------------------


def test_setup_page_returns_200(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import takeout_rater.config as cfg_mod  # noqa: E402

    monkeypatch.setattr(cfg_mod, "get_photos_root", lambda: None)
    monkeypatch.setattr(cfg_mod, "get_db_root", lambda: None)

    resp = client.get("/setup")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_setup_page_contains_form_elements(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import takeout_rater.config as cfg_mod  # noqa: E402

    monkeypatch.setattr(cfg_mod, "get_photos_root", lambda: None)
    monkeypatch.setattr(cfg_mod, "get_db_root", lambda: None)

    resp = client.get("/setup")
    assert "path-input" in resp.text
    assert "Browse" in resp.text
    assert "Save" in resp.text


# ---------------------------------------------------------------------------
# Assets endpoint returns 503 when no DB
# ---------------------------------------------------------------------------


def test_assets_returns_503_when_no_db(client: TestClient) -> None:
    resp = client.get("/assets")
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Setting path initialises DB and updates app state immediately
# ---------------------------------------------------------------------------


def test_set_path_updates_app_state(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """After POST /api/config/takeout-path the app state must have a live DB connection."""
    import takeout_rater.api.config_routes as routes_mod  # noqa: E402

    monkeypatch.setattr(routes_mod, "set_photos_root", lambda p: None)
    monkeypatch.setattr(routes_mod, "set_db_root", lambda p: None)

    # Before setting the path the app is unconfigured
    assert client.app.state.db_conn is None  # type: ignore[union-attr]

    resp = client.post("/api/config/takeout-path", json={"path": str(tmp_path)})
    assert resp.status_code == 200

    # After setting the path the app state must be updated
    assert client.app.state.db_conn is not None  # type: ignore[union-attr]
    assert client.app.state.library_root == tmp_path.resolve()  # type: ignore[union-attr]


def test_set_path_then_assets_accessible(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """After configuring the path, /assets must return 200 (not 503)."""
    import takeout_rater.api.config_routes as routes_mod  # noqa: E402

    monkeypatch.setattr(routes_mod, "set_photos_root", lambda p: None)
    monkeypatch.setattr(routes_mod, "set_db_root", lambda p: None)

    client.post("/api/config/takeout-path", json={"path": str(tmp_path)})

    # Now /assets should be reachable (follow_redirects=False, so expect the HTML directly)
    resp = client.get("/assets")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Browser navigation to unconfigured routes redirects to /setup
# ---------------------------------------------------------------------------


def test_assets_redirects_to_setup_for_html_client(client: TestClient) -> None:
    """A browser-style request (Accept: text/html) to /assets redirects to /setup."""
    resp = client.get("/assets", headers={"Accept": "text/html,application/xhtml+xml,*/*"})
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/setup"


# ---------------------------------------------------------------------------
# Index job triggering after setting path
# ---------------------------------------------------------------------------


def test_set_path_triggers_index_job(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """After POST /api/config/takeout-path an index job is started."""
    import takeout_rater.api.config_routes as routes_mod  # noqa: E402

    monkeypatch.setattr(routes_mod, "set_photos_root", lambda p: None)
    monkeypatch.setattr(routes_mod, "set_db_root", lambda p: None)

    # Capture that _start_index_job is called
    calls: list[Path] = []

    def _fake_start(app: object, photos_root: Path, db_root: Path | None = None) -> None:
        calls.append(photos_root)
        # Don't actually spawn a thread in unit tests

    import takeout_rater.api.jobs as jobs_mod  # noqa: E402

    monkeypatch.setattr(jobs_mod, "_start_index_job", _fake_start)

    resp = client.post("/api/config/takeout-path", json={"path": str(tmp_path)})
    assert resp.status_code == 200
    assert len(calls) == 1
    assert calls[0] == tmp_path.resolve()


# ---------------------------------------------------------------------------
# Jobs API for index status
# ---------------------------------------------------------------------------


def test_index_status_via_jobs_api(client: TestClient) -> None:
    """GET /api/jobs/status?job_type=index returns index status."""
    resp = client.get("/api/jobs/status?job_type=index")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_type"] == "index"
    assert data["running"] is False
    assert data["done"] is False
    assert data["error"] is None
    assert "current_item" in data


def test_index_status_reflects_stored_job_progress(client: TestClient) -> None:
    """The jobs status endpoint reflects whatever is stored in app.state.jobs['index']."""
    from takeout_rater.api.jobs import JobProgress  # noqa: E402

    progress = JobProgress(
        job_type="index",
        running=False,
        done=True,
        processed=42,
        total=42,
        message="Indexed 42 photo(s).",
        current_item="",
    )
    from takeout_rater.api.jobs import _get_jobs  # noqa: E402

    _get_jobs(client.app)["index"] = progress  # type: ignore[union-attr]

    resp = client.get("/api/jobs/status?job_type=index")
    assert resp.status_code == 200
    data = resp.json()
    assert data["done"] is True
    assert data["processed"] == 42
    assert data["total"] == 42
    assert data["message"] == "Indexed 42 photo(s)."
    assert data["current_item"] == ""
