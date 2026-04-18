"""FastAPI router for face detection and person-group management.

Endpoints
---------
GET  /api/faces/detection-runs      – list face detection runs
GET  /api/faces/cluster-runs        – list face clustering runs
GET  /api/faces/clusters/<run_id>   – list person groups for a clustering run
GET  /api/faces/cluster/<id>        – get assets in a person group
POST /api/faces/cluster/<id>/rename – rename a person group
GET  /api/faces/cluster/<id>/similar – CLIP-based similar-photo suggestions
GET  /api/faces/asset/<id>/count    – face count for an asset
DELETE /api/faces/cluster-run/<id>  – delete a face clustering run
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter()


def _require_db(request: Request) -> None:
    if request.app.state.db_conn is None and getattr(request.app.state, "db_path", None) is None:
        raise HTTPException(status_code=503, detail="Library not configured.")


def _get_conn(request: Request):  # noqa: ANN202
    """Return a DB connection, preferring the thread-safe db_path."""
    db_path = getattr(request.app.state, "db_path", None)
    if db_path is not None:
        from takeout_rater.db.connection import open_db  # noqa: PLC0415

        return open_db(db_path)
    return request.app.state.db_conn


# ---------------------------------------------------------------------------
# GET /api/faces/detection-runs
# ---------------------------------------------------------------------------


@router.get("/api/faces/detection-runs")
def list_detection_runs(request: Request) -> JSONResponse:
    """Return all face detection runs."""
    _require_db(request)
    from takeout_rater.db.queries import list_face_detection_runs  # noqa: PLC0415

    conn = _get_conn(request)
    runs = list_face_detection_runs(conn)
    return JSONResponse(runs)


# ---------------------------------------------------------------------------
# GET /api/faces/cluster-runs
# ---------------------------------------------------------------------------


@router.get("/api/faces/cluster-runs")
def list_cluster_runs(request: Request) -> JSONResponse:
    """Return all face clustering runs."""
    _require_db(request)
    from takeout_rater.db.queries import list_face_cluster_runs  # noqa: PLC0415

    conn = _get_conn(request)
    runs = list_face_cluster_runs(conn)
    return JSONResponse(runs)


# ---------------------------------------------------------------------------
# GET /api/faces/clusters/{run_id}
# ---------------------------------------------------------------------------


@router.get("/api/faces/clusters/{run_id}")
def list_clusters(run_id: int, request: Request) -> JSONResponse:
    """Return all person groups for a face clustering run."""
    _require_db(request)
    from takeout_rater.db.queries import list_face_clusters_for_run  # noqa: PLC0415

    conn = _get_conn(request)
    clusters = list_face_clusters_for_run(conn, run_id)
    return JSONResponse(clusters)


# ---------------------------------------------------------------------------
# GET /api/faces/cluster/{cluster_id}
# ---------------------------------------------------------------------------


@router.get("/api/faces/cluster/{cluster_id}")
def get_cluster_detail(cluster_id: int, request: Request) -> JSONResponse:
    """Return all assets in a face cluster (person group)."""
    _require_db(request)
    from takeout_rater.db.queries import (  # noqa: PLC0415
        get_face_cluster_assets,
        get_face_cluster_label,
    )

    conn = _get_conn(request)
    label = get_face_cluster_label(conn, cluster_id)
    assets = get_face_cluster_assets(conn, cluster_id)
    return JSONResponse({"cluster_id": cluster_id, "label": label, "assets": assets})


# ---------------------------------------------------------------------------
# POST /api/faces/cluster/{cluster_id}/rename
# ---------------------------------------------------------------------------


class _RenameBody(BaseModel):
    label: str


@router.post("/api/faces/cluster/{cluster_id}/rename")
def rename_cluster(cluster_id: int, body: _RenameBody, request: Request) -> JSONResponse:
    """Set the user-assigned label (person name) for a face cluster."""
    _require_db(request)
    from takeout_rater.db.queries import rename_face_cluster  # noqa: PLC0415

    conn = _get_conn(request)
    ok = rename_face_cluster(conn, cluster_id, body.label.strip())
    if not ok:
        raise HTTPException(status_code=404, detail="Face cluster not found.")
    return JSONResponse({"status": "ok"})


# ---------------------------------------------------------------------------
# GET /api/faces/cluster/{cluster_id}/similar
# ---------------------------------------------------------------------------


@router.get("/api/faces/cluster/{cluster_id}/similar")
def similar_photos(
    cluster_id: int,
    request: Request,
    threshold: float = 0.80,
    limit: int = 50,
) -> JSONResponse:
    """Find CLIP-similar photos that may contain the same person with a hidden face.

    These are *suggestions* — the face is not detected in these photos but
    the overall scene (clothing, background, body) is visually similar.

    Query params
    ------------
    threshold : float
        Minimum cosine similarity.  Default ``0.80``.
    limit : int
        Maximum number of suggestions.  Default ``50``.
    """
    _require_db(request)
    from takeout_rater.faces.similarity import find_similar_photos  # noqa: PLC0415

    conn = _get_conn(request)
    results = find_similar_photos(conn, cluster_id, threshold=threshold, limit=limit)
    return JSONResponse(results)


# ---------------------------------------------------------------------------
# GET /api/faces/asset/{asset_id}/count
# ---------------------------------------------------------------------------


@router.get("/api/faces/asset/{asset_id}/count")
def face_count_for_asset(asset_id: int, request: Request) -> JSONResponse:
    """Return the number of detected faces for a specific asset."""
    _require_db(request)
    from takeout_rater.db.queries import count_faces_for_asset  # noqa: PLC0415

    conn = _get_conn(request)
    count = count_faces_for_asset(conn, asset_id)
    return JSONResponse({"asset_id": asset_id, "face_count": count})


# ---------------------------------------------------------------------------
# DELETE /api/faces/cluster-run/{run_id}
# ---------------------------------------------------------------------------


@router.delete("/api/faces/cluster-run/{run_id}")
def delete_cluster_run(run_id: int, request: Request) -> JSONResponse:
    """Delete a face clustering run and all its clusters."""
    _require_db(request)
    from takeout_rater.db.queries import delete_face_cluster_run  # noqa: PLC0415

    conn = _get_conn(request)
    ok = delete_face_cluster_run(conn, run_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Face cluster run not found.")
    return JSONResponse({"status": "deleted"})
