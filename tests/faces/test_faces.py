"""Tests for face detection DB queries, clustering, and API endpoints."""

from __future__ import annotations

import sqlite3
import struct
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from takeout_rater.db.queries import (
    bulk_insert_face_embeddings,
    clear_face_detection_data,
    count_face_embeddings,
    count_faces_for_asset,
    delete_face_cluster_run,
    delete_face_detection_run,
    finish_face_detection_run,
    get_face_cluster_assets,
    get_face_cluster_label,
    get_latest_face_detection_run_id,
    insert_face_detection_run,
    list_asset_ids_without_face_detection,
    list_face_cluster_runs,
    list_face_clusters_for_run,
    list_face_detection_runs,
    rename_face_cluster,
    upsert_asset,
)
from takeout_rater.db.schema import migrate
from takeout_rater.faces.detector import EMBEDDING_DIM, FaceDetector
from takeout_rater.ui.app import create_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _make_embedding(seed: float = 0.5) -> bytes:
    """Create a fake 512-d embedding packed as binary blob."""
    import numpy as np

    rng = np.random.RandomState(int(seed * 1000))
    vec = rng.randn(EMBEDDING_DIM).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return struct.pack(f"{EMBEDDING_DIM}f", *vec)


# ---------------------------------------------------------------------------
# DB query tests
# ---------------------------------------------------------------------------


class TestFaceDetectionRunQueries:
    def test_insert_and_list(self) -> None:
        conn = _make_db()
        run_id = insert_face_detection_run(conn, "buffalo_l", '{"det_thresh":0.5}')
        assert run_id > 0

        runs = list_face_detection_runs(conn)
        assert len(runs) == 1
        assert runs[0]["run_id"] == run_id
        assert runs[0]["model_id"] == "buffalo_l"
        assert runs[0]["finished_at"] is None

    def test_finish_run(self) -> None:
        conn = _make_db()
        run_id = insert_face_detection_run(conn, "buffalo_sc", None)
        finish_face_detection_run(conn, run_id)

        runs = list_face_detection_runs(conn)
        assert runs[0]["finished_at"] is not None


class TestFaceEmbeddingQueries:
    def test_bulk_insert_and_count(self) -> None:
        conn = _make_db()
        aid = _add_asset(conn, "Photos/face1.jpg")
        run_id = insert_face_detection_run(conn, "buffalo_l", None)

        blob = _make_embedding(1.0)
        bulk_insert_face_embeddings(
            conn,
            [
                (aid, run_id, 0, 10.0, 20.0, 100.0, 120.0, 0.95, blob),
            ],
        )

        assert count_face_embeddings(conn) == 1
        assert count_faces_for_asset(conn, aid) == 1

    def test_list_asset_ids_without_face_detection(self) -> None:
        conn = _make_db()
        a1 = _add_asset(conn, "Photos/a.jpg")
        a2 = _add_asset(conn, "Photos/b.jpg")
        run_id = insert_face_detection_run(conn, "buffalo_l", None)

        blob = _make_embedding(1.0)
        bulk_insert_face_embeddings(
            conn,
            [
                (a1, run_id, 0, 0, 0, 10, 10, 0.9, blob),
            ],
        )

        # a2 should still need detection
        missing = list_asset_ids_without_face_detection(conn)
        assert a2 in missing
        assert a1 not in missing

    def test_list_asset_ids_without_face_detection_for_run(self) -> None:
        conn = _make_db()
        a1 = _add_asset(conn, "Photos/a.jpg")
        a2 = _add_asset(conn, "Photos/b.jpg")
        run1 = insert_face_detection_run(conn, "buffalo_l", None)
        run2 = insert_face_detection_run(conn, "buffalo_sc", None)

        blob = _make_embedding(1.0)
        bulk_insert_face_embeddings(
            conn,
            [
                (a1, run1, 0, 0, 0, 10, 10, 0.9, blob),
            ],
        )

        # Both assets missing for run2
        missing = list_asset_ids_without_face_detection(conn, run_id=run2)
        assert a1 in missing
        assert a2 in missing

        # Only a2 missing for run1
        missing_r1 = list_asset_ids_without_face_detection(conn, run_id=run1)
        assert a1 not in missing_r1
        assert a2 in missing_r1

    def test_count_face_embeddings_can_filter_by_run(self) -> None:
        conn = _make_db()
        aid = _add_asset(conn, "Photos/a.jpg")
        run1 = insert_face_detection_run(conn, "buffalo_l", None)
        run2 = insert_face_detection_run(conn, "buffalo_l", None)
        blob = _make_embedding(1.0)
        bulk_insert_face_embeddings(
            conn,
            [
                (aid, run1, 0, 0, 0, 10, 10, 0.9, blob),
                (aid, run2, 0, 0, 0, 10, 10, 0.9, blob),
            ],
        )

        assert count_face_embeddings(conn) == 2
        assert count_face_embeddings(conn, run_id=run2) == 1

    def test_clear_face_detection_data_removes_stale_runs_and_clusters(self) -> None:
        conn = _make_db()
        cluster_run_id, _, _ = TestFaceClusterQueries()._setup_clusters(conn)

        assert count_face_embeddings(conn) == 2
        assert list_face_cluster_runs(conn)[0]["run_id"] == cluster_run_id

        clear_face_detection_data(conn)

        assert count_face_embeddings(conn) == 0
        assert list_face_detection_runs(conn) == []
        assert list_face_cluster_runs(conn) == []

    def test_get_latest_face_detection_run_id_prefers_finished_runs(self) -> None:
        conn = _make_db()
        old_finished = insert_face_detection_run(conn, "buffalo_l", None)
        finish_face_detection_run(conn, old_finished)
        newer_unfinished = insert_face_detection_run(conn, "buffalo_l", None)

        assert newer_unfinished > old_finished
        assert get_latest_face_detection_run_id(conn) == old_finished

    def test_delete_face_detection_run_removes_embeddings_and_clusters(self) -> None:
        conn = _make_db()
        cluster_run_id, _, det_run = TestFaceClusterQueries()._setup_clusters(conn)

        assert delete_face_detection_run(conn, det_run) is True

        assert conn.execute("SELECT COUNT(*) FROM face_detection_runs").fetchone()[0] == 0
        assert count_face_embeddings(conn) == 0
        assert list_face_cluster_runs(conn) == []
        assert delete_face_cluster_run(conn, cluster_run_id) is False

    def test_delete_face_detection_run_missing_returns_false(self) -> None:
        conn = _make_db()

        assert delete_face_detection_run(conn, 9999) is False


class TestFaceClusterQueries:
    def _setup_clusters(self, conn: sqlite3.Connection) -> tuple[int, int, int]:
        """Insert two assets with face embeddings and a cluster run with one cluster."""
        a1 = _add_asset(conn, "Photos/person_a1.jpg")
        a2 = _add_asset(conn, "Photos/person_a2.jpg")
        det_run = insert_face_detection_run(conn, "buffalo_l", None)

        blob1 = _make_embedding(1.0)
        blob2 = _make_embedding(1.1)
        bulk_insert_face_embeddings(
            conn,
            [
                (a1, det_run, 0, 0, 0, 100, 100, 0.95, blob1),
                (a2, det_run, 0, 0, 0, 100, 100, 0.90, blob2),
            ],
        )

        # Get face IDs
        face_ids = [
            r[0] for r in conn.execute("SELECT id FROM face_embeddings ORDER BY id").fetchall()
        ]

        # Insert cluster run + cluster
        now = int(time.time())
        cr_row = conn.execute(
            "INSERT INTO face_cluster_runs (method, params_json, detection_run_id, created_at)"
            " VALUES ('dbscan', NULL, ?, ?) RETURNING id",
            (det_run, now),
        ).fetchone()
        cluster_run_id = cr_row[0]
        conn.commit()

        fc_row = conn.execute(
            "INSERT INTO face_clusters (run_id, label, created_at)"
            " VALUES (?, 'Alice', ?) RETURNING id",
            (cluster_run_id, now),
        ).fetchone()
        cluster_id = fc_row[0]
        conn.commit()

        # Add members
        conn.execute(
            "INSERT INTO face_cluster_members (cluster_id, face_id, distance, is_representative)"
            " VALUES (?, ?, 0.0, 1)",
            (cluster_id, face_ids[0]),
        )
        conn.execute(
            "INSERT INTO face_cluster_members (cluster_id, face_id, distance, is_representative)"
            " VALUES (?, ?, 0.1, 0)",
            (cluster_id, face_ids[1]),
        )
        conn.commit()

        return cluster_run_id, cluster_id, det_run

    def test_list_face_cluster_runs(self) -> None:
        conn = _make_db()
        cr_id, _, _ = self._setup_clusters(conn)
        runs = list_face_cluster_runs(conn)
        assert len(runs) == 1
        assert runs[0]["run_id"] == cr_id
        assert runs[0]["n_clusters"] == 1

    def test_list_face_clusters_for_run(self) -> None:
        conn = _make_db()
        cr_id, fc_id, _ = self._setup_clusters(conn)
        clusters = list_face_clusters_for_run(conn, cr_id)
        assert len(clusters) == 1
        assert clusters[0]["cluster_id"] == fc_id
        assert clusters[0]["label"] == "Alice"
        assert clusters[0]["n_faces"] == 2

    def test_get_face_cluster_assets(self) -> None:
        conn = _make_db()
        _, fc_id, _ = self._setup_clusters(conn)
        assets = get_face_cluster_assets(conn, fc_id)
        assert len(assets) == 2
        # Representative should be first
        assert assets[0]["is_representative"] == 1

    def test_rename_face_cluster(self) -> None:
        conn = _make_db()
        _, fc_id, _ = self._setup_clusters(conn)
        ok = rename_face_cluster(conn, fc_id, "Bob")
        assert ok is True
        assert get_face_cluster_label(conn, fc_id) == "Bob"

    def test_rename_nonexistent(self) -> None:
        conn = _make_db()
        ok = rename_face_cluster(conn, 9999, "Nobody")
        assert ok is False

    def test_delete_face_cluster_run(self) -> None:
        conn = _make_db()
        cr_id, _, _ = self._setup_clusters(conn)
        ok = delete_face_cluster_run(conn, cr_id)
        assert ok is True
        assert list_face_cluster_runs(conn) == []

    def test_delete_nonexistent_run(self) -> None:
        conn = _make_db()
        ok = delete_face_cluster_run(conn, 9999)
        assert ok is False


# ---------------------------------------------------------------------------
# Face clustering algorithm tests
# ---------------------------------------------------------------------------


class TestFaceClusteringAlgorithm:
    def test_cluster_faces_groups_similar(self) -> None:
        """Two similar embeddings should end up in the same cluster."""
        import numpy as np

        from takeout_rater.faces.clustering import cluster_faces

        conn = _make_db()
        a1 = _add_asset(conn, "Photos/p1.jpg")
        a2 = _add_asset(conn, "Photos/p2.jpg")
        a3 = _add_asset(conn, "Photos/p3.jpg")
        det_run = insert_face_detection_run(conn, "buffalo_l", None)

        # Create two embeddings that are very similar and one that is different
        rng = np.random.RandomState(42)
        base_vec = rng.randn(EMBEDDING_DIM).astype(np.float32)
        base_vec = base_vec / np.linalg.norm(base_vec)

        similar_vec = base_vec + rng.randn(EMBEDDING_DIM).astype(np.float32) * 0.01
        similar_vec = similar_vec / np.linalg.norm(similar_vec)

        different_vec = rng.randn(EMBEDDING_DIM).astype(np.float32)
        different_vec = different_vec / np.linalg.norm(different_vec)

        blob1 = struct.pack(f"{EMBEDDING_DIM}f", *base_vec)
        blob2 = struct.pack(f"{EMBEDDING_DIM}f", *similar_vec)
        blob3 = struct.pack(f"{EMBEDDING_DIM}f", *different_vec)

        bulk_insert_face_embeddings(
            conn,
            [
                (a1, det_run, 0, 0, 0, 10, 10, 0.95, blob1),
                (a2, det_run, 0, 0, 0, 10, 10, 0.93, blob2),
                (a3, det_run, 0, 0, 0, 10, 10, 0.90, blob3),
            ],
        )

        n = cluster_faces(
            conn,
            detection_run_id=det_run,
            method="dbscan",
            eps=0.3,
            min_samples=2,
        )
        assert n >= 1  # At least one cluster

        runs = list_face_cluster_runs(conn)
        assert len(runs) == 1

    def test_cluster_faces_no_embeddings(self) -> None:
        from takeout_rater.faces.clustering import cluster_faces

        conn = _make_db()
        n = cluster_faces(conn, method="dbscan", eps=0.5, min_samples=2)
        assert n == 0

    def test_cluster_faces_hdbscan_groups_similar(self) -> None:
        import numpy as np

        from takeout_rater.faces.clustering import cluster_faces

        conn = _make_db()
        det_run = insert_face_detection_run(conn, "buffalo_l", None)

        rng = np.random.RandomState(42)
        bases = []
        for _ in range(2):
            base = rng.randn(EMBEDDING_DIM).astype(np.float32)
            bases.append(base / np.linalg.norm(base))

        for base_idx, base in enumerate(bases):
            for item_idx in range(5):
                aid = _add_asset(conn, f"Photos/person_{base_idx}_{item_idx}.jpg")
                vec = base + rng.randn(EMBEDDING_DIM).astype(np.float32) * 0.005
                vec = vec / np.linalg.norm(vec)
                blob = struct.pack(f"{EMBEDDING_DIM}f", *vec)
                bulk_insert_face_embeddings(
                    conn,
                    [(aid, det_run, 0, 0, 0, 10, 10, 0.9, blob)],
                )

        n = cluster_faces(
            conn,
            detection_run_id=det_run,
            method="hdbscan",
            min_cluster_size=2,
            min_samples=1,
        )

        assert n >= 2
        runs = list_face_cluster_runs(conn)
        assert runs[0]["method"] == "hdbscan"

    def test_cluster_faces_with_progress(self) -> None:
        import numpy as np

        from takeout_rater.faces.clustering import cluster_faces

        conn = _make_db()
        a1 = _add_asset(conn, "Photos/x1.jpg")
        a2 = _add_asset(conn, "Photos/x2.jpg")
        det_run = insert_face_detection_run(conn, "buffalo_l", None)

        rng = np.random.RandomState(99)
        for _i, aid in enumerate([a1, a2]):
            vec = rng.randn(EMBEDDING_DIM).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            blob = struct.pack(f"{EMBEDDING_DIM}f", *vec)
            bulk_insert_face_embeddings(
                conn,
                [
                    (aid, det_run, 0, 0, 0, 10, 10, 0.9, blob),
                ],
            )

        progress_calls = []
        cluster_faces(
            conn,
            detection_run_id=det_run,
            method="dbscan",
            eps=0.5,
            min_samples=2,
            on_progress=lambda p, t: progress_calls.append((p, t)),
        )
        assert len(progress_calls) >= 1


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def face_client(tmp_path: Path) -> TestClient:
    conn = _make_db()
    app = create_app(tmp_path, conn, db_root=tmp_path)
    return TestClient(app, follow_redirects=True)


@pytest.fixture()
def face_client_with_data(tmp_path: Path) -> TestClient:
    conn = _make_db()
    a1 = _add_asset(conn, "Photos/face_a.jpg")
    a2 = _add_asset(conn, "Photos/face_b.jpg")
    det_run = insert_face_detection_run(conn, "buffalo_l", None)

    blob1 = _make_embedding(1.0)
    blob2 = _make_embedding(1.5)
    bulk_insert_face_embeddings(
        conn,
        [
            (a1, det_run, 0, 0, 0, 100, 100, 0.95, blob1),
            (a2, det_run, 0, 0, 0, 100, 100, 0.90, blob2),
        ],
    )

    face_ids = [r[0] for r in conn.execute("SELECT id FROM face_embeddings ORDER BY id").fetchall()]

    now = int(time.time())
    cr_row = conn.execute(
        "INSERT INTO face_cluster_runs (method, params_json, detection_run_id, created_at)"
        " VALUES ('dbscan', NULL, ?, ?) RETURNING id",
        (det_run, now),
    ).fetchone()
    conn.commit()
    cr_id = cr_row[0]

    fc_row = conn.execute(
        "INSERT INTO face_clusters (run_id, label, created_at)"
        " VALUES (?, 'TestPerson', ?) RETURNING id",
        (cr_id, now),
    ).fetchone()
    conn.commit()
    fc_id = fc_row[0]

    conn.execute(
        "INSERT INTO face_cluster_members (cluster_id, face_id, distance, is_representative)"
        " VALUES (?, ?, 0.0, 1)",
        (fc_id, face_ids[0]),
    )
    conn.execute(
        "INSERT INTO face_cluster_members (cluster_id, face_id, distance, is_representative)"
        " VALUES (?, ?, 0.1, 0)",
        (fc_id, face_ids[1]),
    )
    conn.commit()

    app = create_app(tmp_path, conn, db_root=tmp_path)
    return TestClient(app, follow_redirects=True)


class TestFacesAPI:
    def test_detection_runs_empty(self, face_client: TestClient) -> None:
        resp = face_client.get("/api/faces/detection-runs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_cluster_runs_empty(self, face_client: TestClient) -> None:
        resp = face_client.get("/api/faces/cluster-runs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_detection_runs_with_data(self, face_client_with_data: TestClient) -> None:
        resp = face_client_with_data.get("/api/faces/detection-runs")
        assert resp.status_code == 200
        runs = resp.json()
        assert len(runs) == 1
        assert runs[0]["model_id"] == "buffalo_l"

    def test_cluster_runs_with_data(self, face_client_with_data: TestClient) -> None:
        resp = face_client_with_data.get("/api/faces/cluster-runs")
        assert resp.status_code == 200
        runs = resp.json()
        assert len(runs) == 1
        assert runs[0]["n_clusters"] == 1

    def test_clusters_for_run(self, face_client_with_data: TestClient) -> None:
        runs = face_client_with_data.get("/api/faces/cluster-runs").json()
        run_id = runs[0]["run_id"]
        resp = face_client_with_data.get(f"/api/faces/clusters/{run_id}")
        assert resp.status_code == 200
        clusters = resp.json()
        assert len(clusters) == 1
        assert clusters[0]["label"] == "TestPerson"
        assert clusters[0]["n_faces"] == 2

    def test_cluster_detail(self, face_client_with_data: TestClient) -> None:
        runs = face_client_with_data.get("/api/faces/cluster-runs").json()
        run_id = runs[0]["run_id"]
        clusters = face_client_with_data.get(f"/api/faces/clusters/{run_id}").json()
        cluster_id = clusters[0]["cluster_id"]
        resp = face_client_with_data.get(f"/api/faces/cluster/{cluster_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == "TestPerson"
        assert len(data["assets"]) == 2

    def test_rename_cluster(self, face_client_with_data: TestClient) -> None:
        runs = face_client_with_data.get("/api/faces/cluster-runs").json()
        run_id = runs[0]["run_id"]
        clusters = face_client_with_data.get(f"/api/faces/clusters/{run_id}").json()
        cluster_id = clusters[0]["cluster_id"]

        resp = face_client_with_data.post(
            f"/api/faces/cluster/{cluster_id}/rename",
            json={"label": "Jane Doe"},
        )
        assert resp.status_code == 200

        detail = face_client_with_data.get(f"/api/faces/cluster/{cluster_id}").json()
        assert detail["label"] == "Jane Doe"

    def test_rename_nonexistent(self, face_client: TestClient) -> None:
        resp = face_client.post(
            "/api/faces/cluster/99999/rename",
            json={"label": "Nobody"},
        )
        assert resp.status_code == 404

    def test_delete_cluster_run(self, face_client_with_data: TestClient) -> None:
        runs = face_client_with_data.get("/api/faces/cluster-runs").json()
        run_id = runs[0]["run_id"]
        resp = face_client_with_data.delete(f"/api/faces/cluster-run/{run_id}")
        assert resp.status_code == 200
        assert face_client_with_data.get("/api/faces/cluster-runs").json() == []

    def test_delete_detection_run(self, face_client_with_data: TestClient) -> None:
        runs = face_client_with_data.get("/api/faces/detection-runs").json()
        run_id = runs[0]["run_id"]

        resp = face_client_with_data.delete(f"/api/faces/detection-run/{run_id}")

        assert resp.status_code == 200
        assert face_client_with_data.get("/api/faces/detection-runs").json() == []
        assert face_client_with_data.get("/api/faces/cluster-runs").json() == []

    def test_delete_nonexistent_run(self, face_client: TestClient) -> None:
        resp = face_client.delete("/api/faces/cluster-run/99999")
        assert resp.status_code == 404

    def test_delete_nonexistent_detection_run(self, face_client: TestClient) -> None:
        resp = face_client.delete("/api/faces/detection-run/99999")
        assert resp.status_code == 404

    def test_face_count_for_asset(self, face_client_with_data: TestClient) -> None:
        resp = face_client_with_data.get("/api/faces/asset/1/count")
        assert resp.status_code == 200
        data = resp.json()
        assert data["face_count"] >= 0

    def test_similar_photos_no_clip(self, face_client_with_data: TestClient) -> None:
        """Similar photos returns empty when no CLIP embeddings exist."""
        runs = face_client_with_data.get("/api/faces/cluster-runs").json()
        run_id = runs[0]["run_id"]
        clusters = face_client_with_data.get(f"/api/faces/clusters/{run_id}").json()
        cluster_id = clusters[0]["cluster_id"]
        resp = face_client_with_data.get(f"/api/faces/cluster/{cluster_id}/similar")
        assert resp.status_code == 200
        assert resp.json() == []


# ---------------------------------------------------------------------------
# UI route tests
# ---------------------------------------------------------------------------


class TestFacesUIRoutes:
    def test_faces_page_renders(self, face_client: TestClient) -> None:
        resp = face_client.get("/faces")
        assert resp.status_code == 200
        assert "Faces" in resp.text

    def test_faces_page_contains_license_notice(self, face_client: TestClient) -> None:
        resp = face_client.get("/faces")
        assert "CC BY-NC 4.0" in resp.text

    def test_faces_nav_link(self, face_client: TestClient) -> None:
        resp = face_client.get("/assets")
        assert "Faces" in resp.text
        assert "/faces" in resp.text

    def test_faces_page_has_face_detection_card(self, face_client: TestClient) -> None:
        resp = face_client.get("/faces")
        assert "Detect Faces" in resp.text
        assert "Run Detection" in resp.text
        assert "Face Detection Runs" in resp.text
        assert 'data-face-accelerator="tensorrt"' in resp.text
        assert 'data-face-accelerator="gpu"' in resp.text
        assert 'class="seg-tab active" id="face-accelerator-tensorrt"' in resp.text
        assert "TensorRT" in resp.text
        assert "CUDA" in resp.text

    def test_faces_page_has_face_clustering_card(self, face_client: TestClient) -> None:
        resp = face_client.get("/faces")
        assert "Cluster Faces" in resp.text
        assert "Run Face Clustering" in resp.text
        assert 'data-face-cluster-method="hdbscan"' in resp.text
        assert 'data-face-cluster-method="dbscan"' in resp.text
        assert 'class="seg-tab active" id="face-cluster-method-hdbscan"' in resp.text

    def test_faces_page_links_to_face_clustering_runs(
        self, face_client_with_data: TestClient
    ) -> None:
        resp = face_client_with_data.get("/faces")
        assert resp.status_code == 200
        assert "Face clustering #" in resp.text
        assert "/faces/clusterings/" in resp.text

    def test_faces_page_lists_face_detection_runs(
        self, face_client_with_data: TestClient
    ) -> None:
        resp = face_client_with_data.get("/faces")

        assert resp.status_code == 200
        assert "Face detection #" in resp.text
        assert "deleteFaceDetectionRun" in resp.text
        assert "/api/faces/detection-run/" in resp.text

    def test_face_clustering_detail_links_to_face_clusters(
        self, face_client_with_data: TestClient
    ) -> None:
        runs = face_client_with_data.get("/api/faces/cluster-runs").json()
        run_id = runs[0]["run_id"]

        resp = face_client_with_data.get(f"/faces/clusterings/{run_id}")

        assert resp.status_code == 200
        assert "Face clustering #" in resp.text
        assert "/faces/clusters/" in resp.text

    def test_face_cluster_detail_links_to_asset_detail(
        self, face_client_with_data: TestClient
    ) -> None:
        runs = face_client_with_data.get("/api/faces/cluster-runs").json()
        run_id = runs[0]["run_id"]
        clusters = face_client_with_data.get(f"/api/faces/clusters/{run_id}").json()
        cluster_id = clusters[0]["cluster_id"]

        resp = face_client_with_data.get(f"/faces/clusters/{cluster_id}")

        assert resp.status_code == 200
        assert "TestPerson" in resp.text
        assert "/assets/1" in resp.text


class TestFaceDetectorAccelerator:
    def test_rejects_unknown_accelerator(self) -> None:
        with pytest.raises(ValueError, match="Unsupported accelerator"):
            FaceDetector(accelerator="cpu")

    def test_gpu_uses_cuda_before_cpu(self) -> None:
        detector = FaceDetector(accelerator="gpu")
        assert detector._providers() == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_tensorrt_uses_tensorrt_before_cuda_and_cpu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
        detector = FaceDetector(accelerator="tensorrt", trt_cache_dir=Path("trt-cache"))

        providers = detector._providers()

        assert providers[0][0] == "TensorrtExecutionProvider"
        assert providers[1:] == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_detect_batched_batches_recognition(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        from types import SimpleNamespace

        import numpy as np

        class _FakeDetModel:
            def __init__(self) -> None:
                self.calls = 0

            def detect(self, img_array, max_num=0, metric="default"):  # type: ignore[no-untyped-def]
                self.calls += 1
                bboxes = np.array(
                    [
                        [1.0, 2.0, 11.0, 12.0, 0.9],
                        [3.0, 4.0, 13.0, 14.0, 0.8],
                    ],
                    dtype=np.float32,
                )
                kpss = np.zeros((2, 5, 2), dtype=np.float32)
                return bboxes, kpss

        class _FakeRecModel:
            input_size = (112, 112)

            def __init__(self) -> None:
                self.batch_sizes: list[int] = []

            def get_feat(self, imgs):  # type: ignore[no-untyped-def]
                self.batch_sizes.append(len(imgs))
                embeddings = np.zeros((len(imgs), EMBEDDING_DIM), dtype=np.float32)
                embeddings[:, 0] = 2.0
                return embeddings

        fake_cv2 = SimpleNamespace(
            BORDER_CONSTANT=0,
            warpAffine=lambda img, transform, size, borderMode=0: np.zeros(
                (size[1], size[0], 3),
                dtype=np.uint8,
            ),
        )
        fake_face_align = SimpleNamespace(
            estimate_norm=lambda kps, image_size=112: np.array(
                [[1, 0, 0], [0, 1, 0]],
                dtype=np.float32,
            )
        )
        monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
        monkeypatch.setitem(
            sys.modules,
            "insightface.utils",
            SimpleNamespace(face_align=fake_face_align),
        )

        det_model = _FakeDetModel()
        rec_model = _FakeRecModel()
        detector = FaceDetector(recognition_batching=True)
        detector._app = type("_FakeApp", (), {"det_model": det_model})()
        detector._rec_model = rec_model

        faces = detector.detect_batched(np.zeros((32, 32, 3), dtype=np.uint8))

        assert det_model.calls == 1
        assert rec_model.batch_sizes == [2]
        assert len(faces) == 2
        assert faces[0].embedding[0] == pytest.approx(1.0)
        assert faces[1].face_index == 1

    def test_tensorrt_disables_recognition_batching_by_default(self) -> None:
        detector = FaceDetector(accelerator="tensorrt")

        assert detector._recognition_batching is False


# ---------------------------------------------------------------------------
# Schema migration test
# ---------------------------------------------------------------------------


class TestFaceSchema:
    def test_fresh_db_has_face_tables(self) -> None:
        conn = _make_db()
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]
        assert "face_detection_runs" in tables
        assert "face_embeddings" in tables
        assert "face_cluster_runs" in tables
        assert "face_clusters" in tables
        assert "face_cluster_members" in tables

    def test_schema_version_is_14(self) -> None:
        conn = _make_db()
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == 14

    def test_face_tables_in_baseline(self) -> None:
        """Face detection tables should be created in the baseline schema."""
        conn = _make_db()

        # Verify tables exist after migration
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]
        assert "face_embeddings" in tables
