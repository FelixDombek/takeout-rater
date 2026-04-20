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
    monkeypatch.setattr(routes_mod, "set_current_library", cfg_mod.set_current_library)
    monkeypatch.setattr(routes_mod, "get_db_root", cfg_mod.get_db_root)

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
    assert data["photos_root"] == str(tmp_path)


def test_performance_config_round_trip(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import takeout_rater.config as cfg_mod  # noqa: E402

    cfg_file = tmp_path / "config.json"
    monkeypatch.setattr(cfg_mod, "_CONFIG_FILE", cfg_file)

    resp = client.get("/api/config/performance")
    assert resp.status_code == 200
    assert resp.json() == {
        "clip_accelerator": "torch",
        "clip_batch_size": 128,
        "clip_fp16": True,
    }

    resp = client.post(
        "/api/config/performance",
        json={"clip_accelerator": "torch", "clip_batch_size": 96, "clip_fp16": False},
    )
    assert resp.status_code == 200
    assert resp.json()["performance"] == {
        "clip_accelerator": "torch",
        "clip_batch_size": 96,
        "clip_fp16": False,
    }
    assert client.get("/api/config/performance").json()["clip_batch_size"] == 96


def test_performance_config_rejects_invalid_batch_size(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import takeout_rater.config as cfg_mod  # noqa: E402

    monkeypatch.setattr(cfg_mod, "_CONFIG_FILE", tmp_path / "config.json")
    resp = client.post(
        "/api/config/performance",
        json={"clip_accelerator": "auto", "clip_batch_size": 999, "clip_fp16": True},
    )
    assert resp.status_code == 400


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


def test_get_config_includes_known_libraries(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import takeout_rater.api.config_routes as routes_mod  # noqa: E402

    library = {
        "id": "photos-123",
        "name": "Photos",
        "photos_root": str(tmp_path / "Photos"),
        "db_root": str(tmp_path / "state"),
    }
    monkeypatch.setattr(routes_mod, "get_photos_root", lambda: None)
    monkeypatch.setattr(routes_mod, "get_db_root", lambda: None)
    monkeypatch.setattr(routes_mod, "list_libraries", lambda: [library])

    resp = client.get("/api/config")

    assert resp.status_code == 200
    assert resp.json()["libraries"] == [library]


# ---------------------------------------------------------------------------
# GET /api/library/status
# ---------------------------------------------------------------------------


def test_library_status_unconfigured(client: TestClient) -> None:
    resp = client.get("/api/library/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["photos_root"] is None
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
    app = create_app(tmp_path, conn, db_root=tmp_path)
    c = TestClient(app, follow_redirects=False)

    resp = c.get("/api/library/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["photos_root"] == str(tmp_path)
    assert isinstance(data["db_schema_version"], int)
    assert data["db_schema_version"] > 0
    assert data["db_scan_version"] == CURRENT_INDEXER_VERSION
    conn.close()


# ---------------------------------------------------------------------------
# POST /api/config/photos-root
# ---------------------------------------------------------------------------


def test_set_path_valid(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    saved: list[Path] = []
    db_root = tmp_path / "state"
    db_root.mkdir()

    import takeout_rater.api.config_routes as routes_mod  # noqa: E402

    monkeypatch.setattr(routes_mod, "set_current_library", lambda p, db_root=None: saved.append(p))

    resp = client.post(
        "/api/config/photos-root",
        json={"path": str(tmp_path), "db_root": str(db_root)},
    )
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

    def _capture_current_library(p: Path, db_root: Path | None = None) -> None:
        photos_saved.append(p)
        db_saved.append(db_root)

    monkeypatch.setattr(routes_mod, "set_current_library", _capture_current_library)

    resp = client.post(
        "/api/config/photos-root",
        json={"path": str(tmp_path), "db_root": str(db_root_dir)},
    )
    assert resp.status_code == 200
    assert len(photos_saved) == 1
    assert len(db_saved) == 1
    assert db_saved[0] == db_root_dir.resolve()


def test_set_path_without_db_root_uses_user_local_library_dir(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Omitting db_root stores the DB under the user-local app directory."""
    import takeout_rater.api.config_routes as routes_mod  # noqa: E402

    app_dir = tmp_path / ".takeout_rater"
    monkeypatch.setattr(routes_mod, "get_app_dir", lambda: app_dir)
    monkeypatch.setattr(
        routes_mod,
        "default_db_root_for_photos_root",
        lambda p: app_dir / f"{p.name}-id",
    )
    monkeypatch.setattr(routes_mod, "set_current_library", lambda p, db_root=None: None)

    import takeout_rater.api.jobs as jobs_mod  # noqa: E402

    monkeypatch.setattr(jobs_mod, "_start_index_job", lambda app, photos_root, db_root: None)

    resp = client.post("/api/config/photos-root", json={"path": str(tmp_path)})

    assert resp.status_code == 200
    db_root = Path(resp.json()["db_root"])
    assert db_root == app_dir / f"{tmp_path.name}-id"
    assert db_root.exists()


def test_set_path_nonexistent_returns_400(client: TestClient) -> None:
    resp = client.post(
        "/api/config/photos-root",
        json={"path": "/this/path/does/not/exist/xyz", "db_root": "/this/path/does/not/exist/xyz"},
    )
    assert resp.status_code == 400


def test_set_path_not_a_directory(client: TestClient, tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("hello")
    resp = client.post("/api/config/photos-root", json={"path": str(f), "db_root": str(tmp_path)})
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


def test_setup_page_contains_reset_button_when_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import takeout_rater.config as cfg_mod  # noqa: E402

    monkeypatch.setattr(cfg_mod, "get_photos_root", lambda: tmp_path)
    monkeypatch.setattr(cfg_mod, "get_db_root", lambda: tmp_path / "state")

    resp = client.get("/setup")

    assert resp.status_code == 200
    assert "btn-reset-library" in resp.text
    assert "Delete database &amp; re-index" in resp.text


def test_setup_page_lists_known_libraries(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import takeout_rater.config as cfg_mod  # noqa: E402

    monkeypatch.setattr(cfg_mod, "get_photos_root", lambda: None)
    monkeypatch.setattr(cfg_mod, "get_db_root", lambda: None)
    monkeypatch.setattr(cfg_mod, "get_app_dir", lambda: tmp_path / ".takeout_rater")
    monkeypatch.setattr(
        cfg_mod,
        "list_libraries",
        lambda: [
            {
                "id": "photos-123",
                "name": "Holiday Photos",
                "photos_root": str(tmp_path / "Holiday Photos"),
                "db_root": str(tmp_path / "state"),
            }
        ],
    )

    resp = client.get("/setup")

    assert resp.status_code == 200
    assert "Known libraries" in resp.text
    assert "Holiday Photos" in resp.text
    assert 'data-library-id="photos-123"' in resp.text


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
    """After POST /api/config/photos-root the app state must have a live DB connection."""
    import takeout_rater.api.config_routes as routes_mod  # noqa: E402

    monkeypatch.setattr(routes_mod, "set_current_library", lambda p, db_root=None: None)
    db_root = tmp_path / "state"
    db_root.mkdir()

    # Before setting the path the app is unconfigured
    assert client.app.state.db_conn is None  # type: ignore[union-attr]

    resp = client.post(
        "/api/config/photos-root",
        json={"path": str(tmp_path), "db_root": str(db_root)},
    )
    assert resp.status_code == 200

    # After setting the path the app state must be updated
    assert client.app.state.db_conn is not None  # type: ignore[union-attr]
    assert client.app.state.photos_root == tmp_path.resolve()  # type: ignore[union-attr]


def test_set_path_then_assets_accessible(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """After configuring the path, /assets must return 200 (not 503)."""
    import takeout_rater.api.config_routes as routes_mod  # noqa: E402

    monkeypatch.setattr(routes_mod, "set_current_library", lambda p, db_root=None: None)
    db_root = tmp_path / "state"
    db_root.mkdir()

    client.post(
        "/api/config/photos-root",
        json={"path": str(tmp_path), "db_root": str(db_root)},
    )

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
    """After POST /api/config/photos-root an index job is started."""
    import takeout_rater.api.config_routes as routes_mod  # noqa: E402

    monkeypatch.setattr(routes_mod, "set_current_library", lambda p, db_root=None: None)

    # Capture that _start_index_job is called
    calls: list[Path] = []

    def _fake_start(app: object, photos_root: Path, db_root: Path) -> None:
        calls.append(photos_root)
        # Don't actually spawn a thread in unit tests

    import takeout_rater.api.jobs as jobs_mod  # noqa: E402

    monkeypatch.setattr(jobs_mod, "_start_index_job", _fake_start)
    db_root = tmp_path / "state"
    db_root.mkdir()

    resp = client.post(
        "/api/config/photos-root",
        json={"path": str(tmp_path), "db_root": str(db_root)},
    )
    assert resp.status_code == 200
    assert len(calls) == 1
    assert calls[0] == tmp_path.resolve()


def test_reset_library_deletes_database_and_starts_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    from takeout_rater.db.connection import (  # noqa: E402
        library_db_path,
        library_state_dir,
        open_library_db,
    )

    photos_root = tmp_path / "photos"
    photos_root.mkdir()
    db_root = tmp_path / "state"
    conn = open_library_db(db_root)
    conn.execute("CREATE TABLE marker (id INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO marker (id) VALUES (1)")
    conn.commit()
    thumbs_dir = library_state_dir(db_root) / "thumbs"
    thumbs_dir.mkdir(parents=True)
    (thumbs_dir / "old-thumb.jpg").write_bytes(b"old")

    calls: list[Path] = []

    def _fake_start(app: object, photos_root: Path, db_root: Path) -> None:
        calls.append(photos_root)

    import takeout_rater.api.jobs as jobs_mod  # noqa: E402

    monkeypatch.setattr(jobs_mod, "_start_index_job", _fake_start)

    app = create_app(photos_root, conn, db_root=db_root)
    client = TestClient(app, follow_redirects=False)

    resp = client.post("/api/config/reset-library")

    assert resp.status_code == 200
    assert calls == [photos_root]
    db_path = library_db_path(db_root)
    assert db_path.exists()
    assert not thumbs_dir.exists()

    check = sqlite3.connect(db_path)
    try:
        marker = check.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'marker'"
        ).fetchone()
    finally:
        check.close()
    assert marker is None


def test_reset_library_rejects_while_job_running(tmp_path: Path) -> None:
    from takeout_rater.api.jobs import JobProgress  # noqa: E402
    from takeout_rater.db.connection import open_library_db  # noqa: E402

    photos_root = tmp_path / "photos"
    photos_root.mkdir()
    db_root = tmp_path / "state"
    app = create_app(photos_root, open_library_db(db_root), db_root=db_root)
    app.state.jobs["score"] = JobProgress(job_type="score", running=True)  # type: ignore[union-attr]
    client = TestClient(app, follow_redirects=False)

    resp = client.post("/api/config/reset-library")

    assert resp.status_code == 409


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
    assert data["diagnostics"] == {}


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
        diagnostics={"scan_seconds": 1.25, "assets_indexed": 42},
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
    assert data["diagnostics"]["scan_seconds"] == 1.25
    assert data["diagnostics"]["assets_indexed"] == 42
