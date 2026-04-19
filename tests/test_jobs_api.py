"""Tests for the background jobs API endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from takeout_rater.ui.app import create_app

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
    app = create_app(tmp_path, conn, db_root=tmp_path)
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
    assert job_types == {
        "index",
        "score",
        "cluster",
        "export",
        "rescan",
        "embed",
        "detect_faces",
        "cluster_faces",
    }


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
        assert "description" in item
        assert "technical_description" in item


# ---------------------------------------------------------------------------
# POST /api/jobs/*/start — no library configured returns 503
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("job_type", ["index", "score", "cluster", "export", "rescan", "embed"])
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


def test_score_worker_runs_to_completion_with_empty_library(tmp_path: Path) -> None:
    """The score worker must complete without errors on an empty library.

    This test actually executes the worker thread (rather than patching it away)
    to ensure that every lazy import inside ``_worker`` resolves correctly and
    that the worker finishes cleanly when there are no assets to score.
    """
    import threading  # noqa: E402

    from takeout_rater.db.connection import open_library_db  # noqa: E402
    from takeout_rater.ui.app import create_app  # noqa: E402

    app = create_app(tmp_path, open_library_db(tmp_path), db_root=tmp_path)
    client = TestClient(app, follow_redirects=False)

    threads: list[threading.Thread] = []
    original_thread = threading.Thread

    class _TrackingThread(original_thread):
        def start(self) -> None:
            threads.append(self)
            super().start()

    import takeout_rater.api.jobs as jobs_mod  # noqa: E402

    orig = jobs_mod.threading.Thread
    jobs_mod.threading.Thread = _TrackingThread  # type: ignore[assignment]
    try:
        resp = client.post("/api/jobs/score/start", json={})
        assert resp.status_code == 200
    finally:
        jobs_mod.threading.Thread = orig  # type: ignore[assignment]

    # Wait for the worker thread to finish (up to 5 s)
    assert len(threads) == 1
    threads[0].join(timeout=5)

    jobs = app.state.jobs  # type: ignore[union-attr]
    assert jobs["score"].done is True
    assert jobs["score"].error is None


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


def test_start_detect_faces_rejects_unknown_accelerator(client_with_db: TestClient) -> None:
    resp = client_with_db.post("/api/jobs/detect_faces/start", json={"accelerator": "cpu"})
    assert resp.status_code == 400
    assert "accelerator" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# GET /jobs page
# ---------------------------------------------------------------------------


def test_jobs_page_returns_200_with_db(client_with_db: TestClient) -> None:
    resp = client_with_db.get("/jobs")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_jobs_page_contains_job_cards(client_with_db: TestClient) -> None:
    resp = client_with_db.get("/jobs")
    assert "Rescan" in resp.text
    assert "Rehash" not in resp.text
    assert "Run Clustering" not in resp.text
    assert "Run Export" not in resp.text
    assert "Run Detection" not in resp.text
    assert "Run Face Clustering" not in resp.text


def test_clusterings_page_contains_job_cards(client_with_db: TestClient) -> None:
    resp = client_with_db.get("/clusterings")
    assert resp.status_code == 200
    assert "Cluster" in resp.text
    assert "Export" in resp.text


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
        processed=100,
        total=100,
        message="Scoring complete.",
    )
    client_with_db.app.state.jobs = {"score": p}  # type: ignore[union-attr]

    resp = client_with_db.get("/api/jobs/status?job_type=score")
    assert resp.status_code == 200
    data = resp.json()
    assert data["done"] is True
    assert data["processed"] == 100
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


# ---------------------------------------------------------------------------
# POST /api/jobs/rescan/start
# ---------------------------------------------------------------------------


def test_start_rescan_job_returns_started(
    client_with_db: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /api/jobs/rescan/start must return {"status": "started"}."""
    import threading  # noqa: E402

    original_thread = threading.Thread

    class _NoOpThread(original_thread):
        def start(self) -> None:
            pass

    monkeypatch.setattr("takeout_rater.api.jobs.threading.Thread", _NoOpThread)

    resp = client_with_db.post("/api/jobs/rescan/start", json={})
    assert resp.status_code == 200
    assert resp.json()["status"] == "started"


def test_start_rescan_job_defaults_to_missing_only(
    client_with_db: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default mode is missing_only."""
    import threading  # noqa: E402

    original_thread = threading.Thread

    class _NoOpThread(original_thread):
        def start(self) -> None:
            pass

    monkeypatch.setattr("takeout_rater.api.jobs.threading.Thread", _NoOpThread)

    resp = client_with_db.post("/api/jobs/rescan/start", json={})
    assert resp.status_code == 200
    jobs = client_with_db.app.state.jobs  # type: ignore[union-attr]
    assert "rescan" in jobs
    assert jobs["rescan"].running is True


def test_start_rescan_job_full_mode(
    client_with_db: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """full mode is accepted without error."""
    import threading  # noqa: E402

    original_thread = threading.Thread

    class _NoOpThread(original_thread):
        def start(self) -> None:
            pass

    monkeypatch.setattr("takeout_rater.api.jobs.threading.Thread", _NoOpThread)

    resp = client_with_db.post("/api/jobs/rescan/start", json={"mode": "full"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "started"


def test_start_rescan_job_invalid_mode_returns_400(
    client_with_db: TestClient,
) -> None:
    """Unknown mode must return 400."""
    resp = client_with_db.post("/api/jobs/rescan/start", json={"mode": "bogus"})
    assert resp.status_code == 400


def test_start_rescan_job_conflicts_when_running(
    client_with_db: TestClient,
) -> None:
    """A second start while already running must return 409."""
    from takeout_rater.api.jobs import JobProgress  # noqa: E402

    client_with_db.app.state.jobs["rescan"] = JobProgress(  # type: ignore[union-attr]
        job_type="rescan", running=True
    )

    resp = client_with_db.post("/api/jobs/rescan/start", json={})
    assert resp.status_code == 409


def test_rescan_status_endpoint_returns_rescan(client_with_db: TestClient) -> None:
    """/api/jobs/status?job_type=rescan must return rescan status."""
    resp = client_with_db.get("/api/jobs/status?job_type=rescan")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_type"] == "rescan"
    assert data["running"] is False


# ---------------------------------------------------------------------------
# Rescan candidate selection (missing_only vs full)
# ---------------------------------------------------------------------------


def test_list_asset_ids_needing_rescan_missing_only(tmp_path: Path) -> None:
    """missing_only selects only NULL or outdated indexer_version rows."""
    from takeout_rater.db.connection import open_library_db  # noqa: E402
    from takeout_rater.db.queries import (  # noqa: E402
        CURRENT_INDEXER_VERSION,
        list_asset_ids_needing_rescan,
        upsert_asset,
    )

    conn = open_library_db(tmp_path)

    # Asset with indexer_version = NULL
    id_null = upsert_asset(
        conn,
        {
            "relpath": "a.jpg",
            "filename": "a.jpg",
            "ext": ".jpg",
            "mime": "image/jpeg",
            "size_bytes": 1,
        },
    )  # type: ignore[call-arg]
    # Asset with indexer_version = current (should be excluded in missing_only)
    id_current = upsert_asset(
        conn,
        {
            "relpath": "b.jpg",
            "filename": "b.jpg",
            "ext": ".jpg",
            "mime": "image/jpeg",
            "size_bytes": 1,
        },
    )  # type: ignore[call-arg]
    conn.execute(
        "UPDATE assets SET indexer_version = ? WHERE id = ?",
        (CURRENT_INDEXER_VERSION, id_current),
    )
    conn.commit()
    # Asset with indexer_version = 0 (old version)
    id_old = upsert_asset(
        conn,
        {
            "relpath": "c.jpg",
            "filename": "c.jpg",
            "ext": ".jpg",
            "mime": "image/jpeg",
            "size_bytes": 1,
        },
    )  # type: ignore[call-arg]
    conn.execute("UPDATE assets SET indexer_version = ? WHERE id = ?", (0, id_old))
    conn.commit()

    candidates = list_asset_ids_needing_rescan(conn, full=False)
    candidate_ids = {row[0] for row in candidates}

    assert id_null in candidate_ids
    assert id_old in candidate_ids
    assert id_current not in candidate_ids

    conn.close()


def test_list_asset_ids_needing_rescan_full(tmp_path: Path) -> None:
    """full mode returns all assets regardless of indexer_version."""
    from takeout_rater.db.connection import open_library_db  # noqa: E402
    from takeout_rater.db.queries import (  # noqa: E402
        CURRENT_INDEXER_VERSION,
        list_asset_ids_needing_rescan,
        upsert_asset,
    )

    conn = open_library_db(tmp_path)

    id1 = upsert_asset(
        conn,
        {
            "relpath": "x.jpg",
            "filename": "x.jpg",
            "ext": ".jpg",
            "mime": "image/jpeg",
            "size_bytes": 1,
        },
    )  # type: ignore[call-arg]
    id2 = upsert_asset(
        conn,
        {
            "relpath": "y.jpg",
            "filename": "y.jpg",
            "ext": ".jpg",
            "mime": "image/jpeg",
            "size_bytes": 1,
        },
    )  # type: ignore[call-arg]
    conn.execute(
        "UPDATE assets SET indexer_version = ? WHERE id = ?",
        (CURRENT_INDEXER_VERSION, id2),
    )
    conn.commit()

    candidates = list_asset_ids_needing_rescan(conn, full=True)
    candidate_ids = {row[0] for row in candidates}

    assert id1 in candidate_ids
    assert id2 in candidate_ids

    conn.close()


def test_rescan_worker_sets_indexer_version(tmp_path: Path) -> None:
    """Running the rescan worker updates indexer_version for all targeted assets."""
    import threading  # noqa: E402
    import time  # noqa: E402

    from takeout_rater.db.connection import open_library_db  # noqa: E402
    from takeout_rater.db.queries import (
        CURRENT_INDEXER_VERSION,
        upsert_asset,
    )  # noqa: E402
    from takeout_rater.ui.app import create_app  # noqa: E402

    conn = open_library_db(tmp_path)
    asset_id = upsert_asset(
        conn,
        {  # type: ignore[call-arg]
            "relpath": "scan_test.jpg",
            "filename": "scan_test.jpg",
            "ext": ".jpg",
            "mime": "image/jpeg",
            "size_bytes": 1,
        },
    )
    conn.commit()
    conn.close()

    app = create_app(tmp_path, open_library_db(tmp_path), db_root=tmp_path)
    from fastapi.testclient import TestClient  # noqa: E402

    client = TestClient(app, follow_redirects=False)

    done_event = threading.Event()
    original_thread = threading.Thread

    class _SyncThread(original_thread):
        def start(self) -> None:
            super().start()
            done_event.set()

    import takeout_rater.api.jobs as jobs_mod  # noqa: E402

    orig = jobs_mod.threading.Thread
    jobs_mod.threading.Thread = _SyncThread  # type: ignore[assignment]
    try:
        resp = client.post("/api/jobs/rescan/start", json={"mode": "missing_only"})
        assert resp.status_code == 200
        # Wait for the worker thread to finish (up to 5 s)
        done_event.wait(timeout=5)
        time.sleep(0.5)
    finally:
        jobs_mod.threading.Thread = orig  # type: ignore[assignment]

    # Verify indexer_version was updated
    check_conn = open_library_db(tmp_path)
    row = check_conn.execute(
        "SELECT indexer_version FROM assets WHERE id = ?", (asset_id,)
    ).fetchone()
    check_conn.close()
    assert row is not None
    assert row[0] == CURRENT_INDEXER_VERSION


def test_rescan_full_regenerates_thumbnail_for_direct_photos_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full rescan must use the configured photos root, not only root/Takeout."""
    import threading  # noqa: E402
    import time  # noqa: E402

    from PIL import Image  # noqa: E402

    from takeout_rater.db.connection import open_library_db  # noqa: E402
    from takeout_rater.db.queries import upsert_asset  # noqa: E402
    from takeout_rater.indexing.thumbnailer import thumb_path_for_id  # noqa: E402

    album_dir = tmp_path / "Album"
    album_dir.mkdir()
    image_path = album_dir / "scan_test.jpg"
    Image.new("RGB", (64, 64), color=(200, 100, 50)).save(image_path, "JPEG")

    conn = open_library_db(tmp_path)
    asset_id = upsert_asset(
        conn,
        {
            "relpath": "Album/scan_test.jpg",
            "filename": "scan_test.jpg",
            "ext": ".jpg",
            "mime": "image/jpeg",
            "size_bytes": image_path.stat().st_size,
        },
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(
        "takeout_rater.scoring.scorers.clip_backbone.get_clip_model",
        lambda: (_ for _ in ()).throw(ImportError("clip unavailable in test")),
    )

    app = create_app(tmp_path, open_library_db(tmp_path), db_root=tmp_path)
    client = TestClient(app, follow_redirects=False)

    done_event = threading.Event()
    original_thread = threading.Thread

    class _SyncThread(original_thread):
        def start(self) -> None:
            super().start()
            done_event.set()

    import takeout_rater.api.jobs as jobs_mod  # noqa: E402

    orig = jobs_mod.threading.Thread
    jobs_mod.threading.Thread = _SyncThread  # type: ignore[assignment]
    try:
        resp = client.post("/api/jobs/rescan/start", json={"mode": "full"})
        assert resp.status_code == 200
        done_event.wait(timeout=5)
        time.sleep(0.5)
    finally:
        jobs_mod.threading.Thread = orig  # type: ignore[assignment]

    thumb = thumb_path_for_id(tmp_path / "takeout-rater" / "thumbs", asset_id)
    assert thumb.exists()


# ---------------------------------------------------------------------------
# POST /api/jobs/index/start
# ---------------------------------------------------------------------------


def test_start_index_job_returns_started(client_with_db: TestClient) -> None:
    """POST /api/jobs/index/start must return {'status': 'started'}."""
    resp = client_with_db.post("/api/jobs/index/start")
    assert resp.status_code == 200
    assert resp.json()["status"] == "started"


def test_start_index_job_conflicts_when_running(client_with_db: TestClient) -> None:
    """A second start while already running must return 409."""
    from takeout_rater.api.jobs import JobProgress  # noqa: E402

    client_with_db.app.state.jobs["index"] = JobProgress(  # type: ignore[union-attr]
        job_type="index", running=True
    )

    resp = client_with_db.post("/api/jobs/index/start")
    assert resp.status_code == 409


def test_index_status_endpoint_returns_index(client_with_db: TestClient) -> None:
    """/api/jobs/status?job_type=index must return index status."""
    resp = client_with_db.get("/api/jobs/status?job_type=index")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_type"] == "index"
    assert data["running"] is False


def test_start_index_job_creates_job_progress(client_with_db: TestClient) -> None:
    """After POST /api/jobs/index/start, app.state.jobs must have an 'index' entry."""
    client_with_db.post("/api/jobs/index/start")
    jobs = client_with_db.app.state.jobs  # type: ignore[union-attr]
    assert "index" in jobs
    assert jobs["index"].job_type == "index"


# ---------------------------------------------------------------------------
# Rescan: album linking
# ---------------------------------------------------------------------------


def _run_rescan_worker(tmp_path: Path) -> None:
    """Helper: start and synchronously wait for the rescan worker to finish."""
    import threading  # noqa: E402
    import time  # noqa: E402

    import takeout_rater.api.jobs as jobs_mod  # noqa: E402
    from takeout_rater.db.connection import open_library_db  # noqa: E402
    from takeout_rater.ui.app import create_app  # noqa: E402

    app = create_app(tmp_path, open_library_db(tmp_path), db_root=tmp_path)
    client = TestClient(app, follow_redirects=False)

    done_event = threading.Event()
    original_thread = threading.Thread

    class _SyncThread(original_thread):
        def start(self) -> None:
            super().start()
            done_event.set()

    orig = jobs_mod.threading.Thread
    jobs_mod.threading.Thread = _SyncThread  # type: ignore[assignment]
    try:
        resp = client.post("/api/jobs/rescan/start", json={"mode": "missing_only"})
        assert resp.status_code == 200
        done_event.wait(timeout=5)
        time.sleep(0.5)
    finally:
        jobs_mod.threading.Thread = orig  # type: ignore[assignment]


def test_rescan_worker_links_asset_to_album(tmp_path: Path) -> None:
    """Rescan must link each asset to the album derived from its relpath."""
    from takeout_rater.db.connection import open_library_db  # noqa: E402
    from takeout_rater.db.queries import upsert_asset  # noqa: E402

    conn = open_library_db(tmp_path)
    upsert_asset(
        conn,
        {
            "relpath": "Summer Vacation 2023/IMG_001.jpg",
            "filename": "IMG_001.jpg",
            "ext": ".jpg",
            "mime": "image/jpeg",
            "size_bytes": 1,
        },
    )
    conn.commit()
    conn.close()

    _run_rescan_worker(tmp_path)

    check_conn = open_library_db(tmp_path)
    albums = check_conn.execute("SELECT name FROM albums").fetchall()
    album_names = [r[0] for r in albums]
    assert "Summer Vacation 2023" in album_names

    links = check_conn.execute(
        "SELECT COUNT(*) FROM album_assets aa"
        " JOIN albums al ON al.id = aa.album_id"
        " WHERE al.name = 'Summer Vacation 2023'"
    ).fetchone()[0]
    check_conn.close()
    assert links == 1


def test_rescan_worker_links_asset_to_all_albums_via_aliases(tmp_path: Path) -> None:
    """Rescan must link a canonical asset to every album it appears in, including
    albums derived from alias paths stored in asset_paths."""
    import time  # noqa: E402

    from takeout_rater.db.connection import open_library_db  # noqa: E402
    from takeout_rater.db.queries import upsert_asset  # noqa: E402

    conn = open_library_db(tmp_path)
    # Insert canonical asset in "Photos from 2023"
    asset_id = upsert_asset(
        conn,
        {
            "relpath": "Photos from 2023/IMG_001.jpg",
            "filename": "IMG_001.jpg",
            "ext": ".jpg",
            "mime": "image/jpeg",
            "size_bytes": 1,
        },
    )
    # Record alias in "Summer Vacation 2023" (same binary, different directory)
    conn.execute(
        "INSERT INTO asset_paths (asset_id, relpath, indexed_at) VALUES (?, ?, ?)",
        (asset_id, "Summer Vacation 2023/IMG_001.jpg", int(time.time())),
    )
    conn.commit()
    conn.close()

    _run_rescan_worker(tmp_path)

    check_conn = open_library_db(tmp_path)
    album_names = {
        r[0]
        for r in check_conn.execute(
            "SELECT al.name FROM albums al"
            " JOIN album_assets aa ON aa.album_id = al.id"
            " WHERE aa.asset_id = ?",
            (asset_id,),
        ).fetchall()
    }
    check_conn.close()
    assert "Photos from 2023" in album_names
    assert "Summer Vacation 2023" in album_names


# ---------------------------------------------------------------------------
# Indexing resume: phash filled in for partially-indexed assets
# ---------------------------------------------------------------------------


def test_run_index_fills_missing_phash_for_existing_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When an asset row already exists but has no phash (aborted indexing),
    a subsequent run_index must compute and store the phash."""
    import sqlite3  # noqa: E402

    from PIL import Image  # noqa: E402

    from takeout_rater.db.connection import library_db_path, open_library_db  # noqa: E402
    from takeout_rater.db.queries import upsert_asset  # noqa: E402
    from takeout_rater.db.schema import migrate  # noqa: E402
    from takeout_rater.indexing.run import run_index  # noqa: E402

    # Disable CLIP model loading so the warmup thread exits immediately.
    monkeypatch.setattr(
        "takeout_rater.scoring.scorers.clip_backbone.is_available",
        lambda: False,
    )

    photos_root = tmp_path / "photos"
    db_root = tmp_path / "state"

    # Build a minimal photos tree with one image
    photos_dir = photos_root / "Photos from 2023"
    photos_dir.mkdir(parents=True)
    img_path = photos_dir / "test.jpg"
    Image.new("RGB", (64, 64), color=(200, 100, 50)).save(img_path, "JPEG")

    # Pre-populate the DB with the asset row (simulating a completed upsert
    # from a previous indexing run that was aborted before phash was saved).
    lib_db = library_db_path(db_root)
    lib_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(lib_db))
    conn.row_factory = sqlite3.Row
    migrate(conn)
    upsert_asset(
        conn,
        {
            "relpath": "Photos from 2023/test.jpg",
            "filename": "test.jpg",
            "ext": ".jpg",
            "mime": "image/jpeg",
            "size_bytes": img_path.stat().st_size,
        },
    )
    conn.commit()
    conn.close()

    # Confirm no phash exists yet
    conn2 = open_library_db(db_root)
    assert conn2.execute("SELECT COUNT(*) FROM phash").fetchone()[0] == 0

    # Re-index — the existing asset row must NOT prevent phash from being computed
    run_index(photos_root, conn2, db_root=db_root)
    conn2.close()

    check_conn = open_library_db(db_root)
    phash_count = check_conn.execute("SELECT COUNT(*) FROM phash").fetchone()[0]
    check_conn.close()
    assert phash_count == 1, "phash must be computed even for assets already in the DB"
