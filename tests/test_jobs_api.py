"""Tests for the background jobs API endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from takeout_rater.ui.app import create_app  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client_no_db() -> TestClient:
    """App with no library configured."""
    app = create_app(None, None)
    return TestClient(app, follow_redirects=False)


@pytest.fixture()
def client_with_db(tmp_path: Path) -> TestClient:
    """App with an in-memory SQLite DB and a library root."""
    from takeout_rater.db.connection import open_library_db  # noqa: E402

    conn = open_library_db(tmp_path)
    app = create_app(tmp_path, conn)
    return TestClient(app, follow_redirects=False)


# ---------------------------------------------------------------------------
# GET /api/jobs/status — no library configured
# ---------------------------------------------------------------------------


def test_jobs_status_returns_200_unconfigured(client_no_db: TestClient) -> None:
    """Status endpoint is reachable even when no library is configured."""
    resp = client_no_db.get("/api/jobs/status")
    assert resp.status_code == 200


def test_jobs_status_returns_list_by_default(client_no_db: TestClient) -> None:
    resp = client_no_db.get("/api/jobs/status")
    data = resp.json()
    assert isinstance(data, list)
    job_types = {item["job_type"] for item in data}
    assert job_types == {"score", "cluster", "export", "rehash"}


def test_jobs_status_initial_all_not_running(client_no_db: TestClient) -> None:
    resp = client_no_db.get("/api/jobs/status")
    for item in resp.json():
        assert item["running"] is False
        assert item["done"] is False
        assert item["error"] is None


def test_jobs_status_specific_job_type(client_no_db: TestClient) -> None:
    resp = client_no_db.get("/api/jobs/status?job_type=score")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_type"] == "score"
    assert data["running"] is False


def test_jobs_status_unknown_job_type_returns_400(client_no_db: TestClient) -> None:
    resp = client_no_db.get("/api/jobs/status?job_type=bogus")
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /api/jobs/scorers
# ---------------------------------------------------------------------------


def test_list_scorers_returns_200(client_no_db: TestClient) -> None:
    resp = client_no_db.get("/api/jobs/scorers")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) > 0
    for item in data:
        assert "id" in item
        assert "name" in item
        assert "available" in item


# ---------------------------------------------------------------------------
# POST /api/jobs/*/start — no library configured returns 503
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("job_type", ["score", "cluster", "export", "rehash"])
def test_start_job_without_db_returns_503(client_no_db: TestClient, job_type: str) -> None:
    resp = client_no_db.post(f"/api/jobs/{job_type}/start", json={})
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# POST /api/jobs/score/start — with library configured
# ---------------------------------------------------------------------------


def test_start_score_job_returns_started(
    client_with_db: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /api/jobs/score/start must return {"status": "started"}."""

    # Patch _start_background_index equivalent: capture the thread target
    started: list = []

    original_thread = __import__("threading").Thread

    class _NoOpThread(original_thread):
        def start(self) -> None:
            started.append(True)
            # Don't actually run the worker

    monkeypatch.setattr("takeout_rater.api.jobs.threading.Thread", _NoOpThread)

    resp = client_with_db.post("/api/jobs/score/start", json={})
    assert resp.status_code == 200
    assert resp.json()["status"] == "started"
    assert len(started) == 1


def test_start_score_job_conflicts_when_running(
    client_with_db: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second start while already running must return 409."""
    from takeout_rater.api.jobs import JobProgress  # noqa: E402

    client_with_db.app.state.jobs["score"] = JobProgress(  # type: ignore[union-attr]
        job_type="score", running=True
    )

    resp = client_with_db.post("/api/jobs/score/start", json={})
    assert resp.status_code == 409


def test_start_score_job_sets_progress_in_app_state(
    client_with_db: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading  # noqa: E402

    original_thread = threading.Thread

    class _NoOpThread(original_thread):
        def start(self) -> None:
            pass  # Don't run worker

    monkeypatch.setattr("takeout_rater.api.jobs.threading.Thread", _NoOpThread)

    resp = client_with_db.post("/api/jobs/score/start", json={})
    assert resp.status_code == 200

    jobs = client_with_db.app.state.jobs  # type: ignore[union-attr]
    assert "score" in jobs
    assert jobs["score"].running is True


# ---------------------------------------------------------------------------
# POST /api/jobs/cluster/start
# ---------------------------------------------------------------------------


def test_start_cluster_job_returns_started(
    client_with_db: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading  # noqa: E402

    original_thread = threading.Thread

    class _NoOpThread(original_thread):
        def start(self) -> None:
            pass

    monkeypatch.setattr("takeout_rater.api.jobs.threading.Thread", _NoOpThread)

    resp = client_with_db.post(
        "/api/jobs/cluster/start",
        json={"threshold": 10, "window": 200, "min_size": 2},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "started"


def test_start_cluster_job_conflicts_when_running(
    client_with_db: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from takeout_rater.api.jobs import JobProgress  # noqa: E402

    client_with_db.app.state.jobs["cluster"] = JobProgress(  # type: ignore[union-attr]
        job_type="cluster", running=True
    )

    resp = client_with_db.post("/api/jobs/cluster/start", json={})
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# POST /api/jobs/export/start
# ---------------------------------------------------------------------------


def test_start_export_job_returns_started(
    client_with_db: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading  # noqa: E402

    original_thread = threading.Thread

    class _NoOpThread(original_thread):
        def start(self) -> None:
            pass

    monkeypatch.setattr("takeout_rater.api.jobs.threading.Thread", _NoOpThread)

    resp = client_with_db.post("/api/jobs/export/start", json={})
    assert resp.status_code == 200
    assert resp.json()["status"] == "started"


def test_start_export_job_requires_metric_when_scorer_given(
    client_with_db: TestClient,
) -> None:
    resp = client_with_db.post("/api/jobs/export/start", json={"scorer_id": "blur"})
    assert resp.status_code == 400


def test_start_export_job_conflicts_when_running(
    client_with_db: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from takeout_rater.api.jobs import JobProgress  # noqa: E402

    client_with_db.app.state.jobs["export"] = JobProgress(  # type: ignore[union-attr]
        job_type="export", running=True
    )

    resp = client_with_db.post("/api/jobs/export/start", json={})
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# POST /api/jobs/rehash/start
# ---------------------------------------------------------------------------


def test_start_rehash_job_returns_started(
    client_with_db: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading  # noqa: E402

    original_thread = threading.Thread

    class _NoOpThread(original_thread):
        def start(self) -> None:
            pass

    monkeypatch.setattr("takeout_rater.api.jobs.threading.Thread", _NoOpThread)

    resp = client_with_db.post("/api/jobs/rehash/start", json={})
    assert resp.status_code == 200
    assert resp.json()["status"] == "started"


def test_start_rehash_job_conflicts_when_running(
    client_with_db: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from takeout_rater.api.jobs import JobProgress  # noqa: E402

    client_with_db.app.state.jobs["rehash"] = JobProgress(  # type: ignore[union-attr]
        job_type="rehash", running=True
    )

    resp = client_with_db.post("/api/jobs/rehash/start", json={})
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# GET /jobs page
# ---------------------------------------------------------------------------


def test_jobs_page_returns_200_with_db(client_with_db: TestClient) -> None:
    resp = client_with_db.get("/jobs")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_jobs_page_contains_job_cards(client_with_db: TestClient) -> None:
    resp = client_with_db.get("/jobs")
    assert "Score" in resp.text
    assert "Cluster" in resp.text
    assert "Export" in resp.text
    assert "Rehash" in resp.text


def test_jobs_page_redirects_without_db(client_no_db: TestClient) -> None:
    resp = client_no_db.get("/jobs")
    assert resp.status_code in (302, 307)


# ---------------------------------------------------------------------------
# Job status reflects stored progress
# ---------------------------------------------------------------------------


def test_jobs_status_reflects_stored_progress(client_with_db: TestClient) -> None:
    from takeout_rater.api.jobs import JobProgress  # noqa: E402

    p = JobProgress(
        job_type="score",
        running=False,
        done=True,
        scored=100,
        total=100,
        message="Scoring complete.",
    )
    client_with_db.app.state.jobs = {"score": p}  # type: ignore[union-attr]

    resp = client_with_db.get("/api/jobs/status?job_type=score")
    assert resp.status_code == 200
    data = resp.json()
    assert data["done"] is True
    assert data["scored"] == 100
    assert data["total"] == 100
    assert data["message"] == "Scoring complete."


# ---------------------------------------------------------------------------
# Nav links presence in base template
# ---------------------------------------------------------------------------


def test_jobs_nav_link_present_in_browse(client_with_db: TestClient) -> None:
    """The /jobs link must appear in the navigation on the browse page."""
    resp = client_with_db.get("/assets")
    assert resp.status_code == 200
    assert "/jobs" in resp.text
