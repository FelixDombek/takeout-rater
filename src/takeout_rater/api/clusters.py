"""FastAPI router for cluster listing and detail views."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from takeout_rater.db.queries import (
    count_clusters,
    get_cluster_members,
    list_clusters_with_representatives,
)

router = APIRouter()

_PAGE_SIZE = 50


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


@router.get("/clusters", response_class=HTMLResponse)
def browse_clusters(
    request: Request,
    page: int = 1,
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> HTMLResponse:
    """Render the cluster browse page.

    Query parameters:
    - ``page``: 1-based page number (default 1).
    """
    offset = max(0, (page - 1) * _PAGE_SIZE)
    clusters = list_clusters_with_representatives(conn, limit=_PAGE_SIZE, offset=offset)
    total = count_clusters(conn)
    total_pages = max(1, (total + _PAGE_SIZE - 1) // _PAGE_SIZE)

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "clusters.html",
        {
            "request": request,
            "clusters": clusters,
            "page": page,
            "total_pages": total_pages,
            "total": total,
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

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "cluster_detail.html",
        {
            "request": request,
            "cluster_id": cluster_id,
            "members": members,
        },
    )
