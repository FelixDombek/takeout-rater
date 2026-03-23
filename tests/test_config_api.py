"""Tests for the config API endpoints and health probe."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from takeout_rater.ui.app import create_app  # noqa: E402


@pytest.fixture()
def client() -> TestClient:
    """App with no library configured (unconfigured mode)."""
    app = create_app(None, None)
    return TestClient(app, follow_redirects=False)


@pytest.fixture()
def client_configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """App that has a config file pointing at *tmp_path*."""
    cfg_file = tmp_path / ".takeout-rater.json"
    cfg_file.write_text(json.dumps({"takeout_path": str(tmp_path)}))

    import takeout_rater.config as cfg_mod  # noqa: E402

    monkeypatch.setattr(cfg_mod, "_CONFIG_FILE", cfg_file)

    import takeout_rater.api.config_routes as routes_mod  # noqa: E402

    monkeypatch.setattr(routes_mod, "get_takeout_path", cfg_mod.get_takeout_path)
    monkeypatch.setattr(routes_mod, "set_takeout_path", cfg_mod.set_takeout_path)

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

    monkeypatch.setattr(routes_mod, "get_takeout_path", lambda: None)

    resp = client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["configured"] is False
    assert data["takeout_path"] is None


def test_get_config_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import takeout_rater.api.config_routes as routes_mod  # noqa: E402

    monkeypatch.setattr(routes_mod, "get_takeout_path", lambda: tmp_path)

    resp = client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["configured"] is True
    assert data["takeout_path"] == str(tmp_path)


# ---------------------------------------------------------------------------
# POST /api/config/takeout-path
# ---------------------------------------------------------------------------


def test_set_path_valid(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    saved: list[Path] = []

    import takeout_rater.api.config_routes as routes_mod  # noqa: E402

    monkeypatch.setattr(routes_mod, "set_takeout_path", lambda p: saved.append(p))

    resp = client.post("/api/config/takeout-path", json={"path": str(tmp_path)})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert len(saved) == 1
    assert saved[0] == tmp_path.resolve()


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

    monkeypatch.setattr(cfg_mod, "get_takeout_path", lambda: None)

    resp = client.get("/setup")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_setup_page_contains_form_elements(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import takeout_rater.config as cfg_mod  # noqa: E402

    monkeypatch.setattr(cfg_mod, "get_takeout_path", lambda: None)

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

    monkeypatch.setattr(routes_mod, "set_takeout_path", lambda p: None)

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

    monkeypatch.setattr(routes_mod, "set_takeout_path", lambda p: None)

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
# Index status endpoint
# ---------------------------------------------------------------------------


def test_index_status_before_any_indexing(client: TestClient) -> None:
    """GET /api/index/status returns a not-running, not-done response by default."""
    resp = client.get("/api/index/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["running"] is False
    assert data["done"] is False
    assert data["error"] is None


def test_index_status_after_setting_path_triggers_background_index(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """After POST /api/config/takeout-path a background index run is started."""
    import takeout_rater.api.config_routes as routes_mod  # noqa: E402

    monkeypatch.setattr(routes_mod, "set_takeout_path", lambda p: None)

    # Capture that _start_background_index is called
    calls: list[Path] = []

    def _fake_start(app: object, library_root: Path) -> None:
        calls.append(library_root)
        # Don't actually spawn a thread in unit tests

    monkeypatch.setattr(routes_mod, "_start_background_index", _fake_start)

    resp = client.post("/api/config/takeout-path", json={"path": str(tmp_path)})
    assert resp.status_code == 200
    assert len(calls) == 1
    assert calls[0] == tmp_path.resolve()


def test_index_status_returns_progress_fields(client: TestClient) -> None:
    """GET /api/index/status returns all expected progress fields including new phase fields."""
    resp = client.get("/api/index/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "running" in data
    assert "done" in data
    assert "error" in data
    assert "found" in data
    assert "indexed" in data
    assert "thumbs_ok" in data
    assert "thumbs_skip" in data
    assert "phase" in data
    assert "total_dirs" in data
    assert "dirs_scanned" in data
    assert "current_dir" in data


def test_index_status_default_phase_is_scanning(client: TestClient) -> None:
    """Before any indexing starts the phase must default to 'scanning'."""
    resp = client.get("/api/index/status")
    data = resp.json()
    assert data["phase"] == "scanning"
    assert data["total_dirs"] == 0
    assert data["dirs_scanned"] == 0
    assert data["current_dir"] == ""


def test_index_status_reflects_stored_progress(client: TestClient) -> None:
    """The status endpoint reflects whatever is stored in app.state.index_progress."""
    from takeout_rater.indexing.run import IndexProgress  # noqa: E402

    progress = IndexProgress(
        running=False,
        done=True,
        found=42,
        indexed=42,
        phase="indexing",
        total_dirs=5,
        dirs_scanned=5,
        current_dir="Photos from 2023",
    )
    client.app.state.index_progress = progress  # type: ignore[union-attr]

    resp = client.get("/api/index/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["done"] is True
    assert data["found"] == 42
    assert data["indexed"] == 42
    assert data["phase"] == "indexing"
    assert data["total_dirs"] == 5
    assert data["dirs_scanned"] == 5
    assert data["current_dir"] == "Photos from 2023"
