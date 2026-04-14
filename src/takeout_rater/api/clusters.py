"""FastAPI router for cluster listing and detail views."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from takeout_rater.db.queries import (
    delete_all_clusters,
    delete_clustering_run,
    get_cluster_info,
    get_cluster_member_hashes,
    get_cluster_members,
    get_clustering_run,
    list_clustering_runs,
    list_clusters_for_run,
)

router = APIRouter()


def _get_conn(request: Request) -> Generator[sqlite3.Connection, None, None]:
    db_path = request.app.state.db_path
    if db_path is not None:
        from takeout_rater.db.connection import open_db  # noqa: PLC0415

        conn = open_db(db_path)
        try:
            yield conn
        finally:
            conn.close()
        return

    # Fallback for in-memory databases (used in tests).
    conn = request.app.state.db_conn
    if conn is None:
        raise HTTPException(status_code=503, detail="Library not configured — visit /setup")
    yield conn


def _parse_params(params_json: str | None) -> dict:
    """Parse a params_json string into a dict, returning {} on failure."""
    if not params_json:
        return {}
    try:
        return json.loads(params_json)
    except (ValueError, TypeError):
        return {}


@router.delete("/api/clusters/clear")
def clear_clusters(
    request: Request,
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse:
    """Delete all clustering runs and their clusters from the database.

    Returns a JSON object with the number of runs deleted.
    """
    n = delete_all_clusters(conn)
    return JSONResponse({"deleted": n})


@router.delete("/api/clusterings/{run_id}")
def delete_run(
    run_id: int,
    request: Request,
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse:
    """Delete a single clustering run and all its clusters.

    Returns 404 if the run does not exist.
    """
    deleted = delete_clustering_run(conn, run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Clustering run {run_id} not found")
    return JSONResponse({"deleted": run_id})


@router.get("/clusterings", response_class=HTMLResponse)
def browse_clusters(
    request: Request,
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> HTMLResponse:
    """Render the clustering runs list page."""
    runs = list_clustering_runs(conn)
    # Enrich each run with parsed params for easy template access.
    for run in runs:
        run["params"] = _parse_params(run["params_json"])

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "clusterings.html",
        {
            "request": request,
            "runs": runs,
        },
    )


@router.get("/clusterings/{run_id}", response_class=HTMLResponse)
def clustering_detail(
    run_id: int,
    request: Request,
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> HTMLResponse:
    """Render the detail page for a single clustering run (all its clusters)."""
    run = get_clustering_run(conn, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Clustering run {run_id} not found")

    clusters = list_clusters_for_run(conn, run_id)
    run["params"] = _parse_params(run["params_json"])

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "clustering_detail.html",
        {
            "request": request,
            "run": run,
            "clusters": clusters,
        },
    )


@router.get("/clusters/{cluster_id}", response_class=HTMLResponse)
def cluster_detail(
    cluster_id: int,
    request: Request,
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> HTMLResponse:
    """Render the detail page for a single cluster."""
    members = get_cluster_members(conn, cluster_id)
    if not members:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")

    cluster_info = get_cluster_info(conn, cluster_id)
    phash_by_id = get_cluster_member_hashes(conn, cluster_id)

    cluster_method = cluster_info["method"] if cluster_info else None
    cluster_params = _parse_params(cluster_info["params_json"] if cluster_info else None)

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "cluster_detail.html",
        {
            "request": request,
            "cluster_id": cluster_id,
            "run_id": cluster_info["run_id"] if cluster_info else None,
            "members": members,
            "cluster_diameter": cluster_info["diameter"] if cluster_info else None,
            "cluster_method": cluster_method,
            "cluster_params": cluster_params,
            "phash_by_id": phash_by_id,
        },
    )
