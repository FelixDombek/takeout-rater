"""FastAPI router for the CLIP feature page.

Provides API endpoints for managing user-defined CLIP vocabulary tags and
for generating the 3-D embedding map used by the interactive visualization.

User tags are stored in the ``clip_user_tags`` table and are included
alongside the predefined vocabulary when computing CLIP word matches in
the asset detail view.

Endpoints
---------
GET  /api/clip/tags              – list all user-defined tags
POST /api/clip/tags              – add a new user-defined tag
DELETE /api/clip/tags/{term}     – remove a user-defined tag
GET  /api/clip/embedding-map     – compute/return 3-D UMAP projection + clusters
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from takeout_rater.db.queries import (
    delete_clip_user_tag,
    insert_clip_user_tag,
    list_clip_user_tags,
    load_clip_embeddings_with_relpaths,
)

router = APIRouter()


def _get_conn(request: Request) -> Generator[sqlite3.Connection, None, None]:
    db_path = request.app.state.db_path
    if db_path is None:
        raise HTTPException(status_code=503, detail="Library not configured.")
    from takeout_rater.db.connection import open_db

    conn = open_db(db_path)
    try:
        yield conn
    finally:
        conn.close()


class _AddTagRequest(BaseModel):
    term: str


@router.get("/api/clip/tags")
def list_tags(
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse:
    """Return all user-defined CLIP tagging terms.

    Returns JSON ``{"tags": ["term1", "term2", ...]}`` in creation order.
    """
    tags = list_clip_user_tags(conn)
    return JSONResponse({"tags": tags})


@router.post("/api/clip/tags", status_code=201)
def add_tag(
    body: _AddTagRequest,
    request: Request,
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse:
    """Add a new user-defined CLIP tagging term.

    Request body: ``{"term": "my custom tag"}``

    Returns ``409`` if the term already exists, ``400`` if the term is empty.
    The user-tag embedding cache is invalidated so the new term is picked up
    on the next ``/api/assets/{id}/clip-words`` call.
    """
    term = body.term.strip()
    if not term:
        raise HTTPException(status_code=400, detail="Tag term must not be empty.")

    inserted = insert_clip_user_tag(conn, term)
    if not inserted:
        raise HTTPException(status_code=409, detail=f"Tag '{term}' already exists.")

    # Invalidate the in-memory user-tag embedding cache
    request.app.state.clip_user_tags_matrix = None

    tags = list_clip_user_tags(conn)
    return JSONResponse({"tags": tags}, status_code=201)


@router.delete("/api/clip/tags/{term}")
def remove_tag(
    term: str,
    request: Request,
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse:
    """Remove a user-defined CLIP tagging term.

    Returns ``404`` if the term does not exist.
    The user-tag embedding cache is invalidated.
    """
    deleted = delete_clip_user_tag(conn, term)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Tag '{term}' not found.")

    # Invalidate the in-memory user-tag embedding cache
    request.app.state.clip_user_tags_matrix = None

    tags = list_clip_user_tags(conn)
    return JSONResponse({"tags": tags})


@router.get("/api/clip/embedding-map/progress")
def get_embedding_map_progress(request: Request) -> JSONResponse:
    """Return the progress of the current or most-recent embedding-map build.

    Intended to be polled by the frontend while ``/api/clip/embedding-map`` is
    computing.  Safe to call at any time; returns a no-op response when no
    build has been triggered yet.

    Returns JSON::

        {"fraction": 0.0..1.0, "message": "...", "active": true|false}

    ``active`` is ``true`` only while a build is in progress; it becomes
    ``false`` once the build finishes (or if no build has ever started).
    """
    progress = getattr(request.app.state, "clip_map_progress", None)
    if progress is None:
        return JSONResponse({"fraction": 0.0, "message": "", "active": False})
    return JSONResponse(progress)


@router.get("/api/clip/embedding-map")
def get_embedding_map(
    request: Request,
    refresh: bool = False,
    clustering_method: Literal["kmeans", "hdbscan"] = "kmeans",
    max_clusters: int = Query(24, ge=1, le=200),
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse:
    """Return a 3-D UMAP projection of all stored CLIP embeddings.

    The pipeline is:
    1. **StandardScaler** – per-dimension normalisation.
    2. **PCA** – reduce 768 → 50 components to remove noise.
    3. **UMAP** – project 50 → 3 dimensions (``metric="cosine"``).
    4. **Clustering** – cluster the 3-D points with K-Means or HDBSCAN.
    5. **Representative** – for each cluster, the asset closest to its
       centroid is selected as the preview thumbnail.

    The result is cached in ``app.state.clip_embedding_map`` and reused on
    subsequent calls unless ``refresh=true`` is passed.

    Query parameters
    ----------------
    refresh : bool
        Pass ``true`` to force recomputation (e.g. after new embeddings have
        been added).
    clustering_method : "kmeans" | "hdbscan"
        Algorithm used to group the 3-D points for colouring.
    max_clusters : int
        Maximum number of non-noise clusters to show.

    Returns
    -------
    JSON:

    .. code-block:: json

        {
            "points": [
                {"asset_id": 1, "x": 0.1, "y": -0.2, "z": 0.5,
                 "cluster_id": 0, "relpath": "Photos/img.jpg"}
            ],
            "clusters": [
                {"cluster_id": 0, "representative_asset_id": 1, "size": 42}
            ],
            "total": 500,
            "params": {
                "clustering_method": "kmeans",
                "max_clusters": 25
            }
        }
    """
    params = {
        "clustering_method": clustering_method,
        "max_clusters": max_clusters,
    }

    # Return cached result when available and refresh not requested.
    cached = getattr(request.app.state, "clip_embedding_map", None)
    if cached is not None and not refresh and cached.get("params") == params:
        return JSONResponse(cached)

    # Initialise progress state visible to the polling endpoint.
    app_state = request.app.state
    app_state.clip_map_progress = {
        "fraction": 0.0,
        "message": "Loading embeddings from database…",
        "active": True,
    }

    def _progress(fraction: float, message: str) -> None:
        app_state.clip_map_progress = {
            "fraction": fraction,
            "message": message,
            "active": True,
        }

    rows = load_clip_embeddings_with_relpaths(conn)
    if not rows:
        app_state.clip_map_progress = {"fraction": 1.0, "message": "", "active": False}
        return JSONResponse({"points": [], "clusters": [], "total": 0, "params": params})

    from takeout_rater.clustering.embedding_map import build_embedding_map

    result = build_embedding_map(
        rows,
        progress_callback=_progress,
        clustering_method=clustering_method,
        max_clusters=max_clusters,
    )
    app_state.clip_map_progress = {"fraction": 1.0, "message": "Done", "active": False}
    app_state.clip_embedding_map = result
    return JSONResponse(result)
