"""Slow tests for background jobs API endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from takeout_rater.ui.app import create_app


@pytest.fixture()
def client_no_db() -> TestClient:
    app = create_app(None, None)
    return TestClient(app, follow_redirects=False)


@pytest.fixture()
def client_with_db(tmp_path: Path) -> TestClient:
    from takeout_rater.db.connection import open_library_db

    conn = open_library_db(tmp_path)
    app = create_app(tmp_path, conn, db_root=tmp_path)
    return TestClient(app, follow_redirects=False)


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


def test_start_index_job_returns_started(client_with_db: TestClient) -> None:
    resp = client_with_db.post("/api/jobs/index/start")
    assert resp.status_code == 200
    assert resp.json()["status"] == "started"


def test_start_index_job_creates_job_progress(client_with_db: TestClient) -> None:
    client_with_db.post("/api/jobs/index/start")
    jobs = client_with_db.app.state.jobs  # type: ignore[union-attr]
    assert "index" in jobs
    assert jobs["index"].job_type == "index"


def _run_rescan_worker(tmp_path: Path) -> None:
    import threading
    import time

    import takeout_rater.api.jobs as jobs_mod
    from takeout_rater.db.connection import open_library_db

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


def test_rescan_worker_links_asset_to_all_albums_via_aliases(tmp_path: Path) -> None:
    import time

    from takeout_rater.db.connection import open_library_db
    from takeout_rater.db.queries import upsert_asset

    conn = open_library_db(tmp_path)
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
