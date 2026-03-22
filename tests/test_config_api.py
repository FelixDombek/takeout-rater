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
