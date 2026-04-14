"""FastAPI router for the CLIP feature page.

Provides API endpoints for managing user-defined CLIP vocabulary tags.
User tags are stored in the ``clip_user_tags`` table and are included
alongside the predefined vocabulary when computing CLIP word matches in
the asset detail view.

Endpoints
---------
GET  /api/clip/tags           – list all user-defined tags
POST /api/clip/tags           – add a new user-defined tag
DELETE /api/clip/tags/{term}  – remove a user-defined tag
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from takeout_rater.db.queries import (
    delete_clip_user_tag,
    insert_clip_user_tag,
    list_clip_user_tags,
)

router = APIRouter()


def _get_conn(request: Request) -> Generator[sqlite3.Connection, None, None]:
    db_path = request.app.state.db_path
    if db_path is None:
        raise HTTPException(status_code=503, detail="Library not configured.")
    from takeout_rater.db.connection import open_library_db  # noqa: PLC0415

    conn = open_library_db(request.app.state.library_root)
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
