"""FastAPI router for album listing and detail views."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from takeout_rater.db.queries import (
    count_album_assets,
    get_album,
    get_album_assets,
    list_albums,
)

router = APIRouter()

_PAGE_SIZE = 50


def _get_conn(request: Request) -> Generator[sqlite3.Connection, None, None]:
    db_path = request.app.state.db_path
    if db_path is not None:
        from takeout_rater.db.connection import open_db

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


@router.get("/albums", response_class=HTMLResponse)
def browse_albums(
    request: Request,
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> HTMLResponse:
    """Render the top-level album list page."""
    albums = list_albums(conn)
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "albums.html",
        {
            "request": request,
            "albums": albums,
        },
    )


@router.get("/albums/{album_id}", response_class=HTMLResponse)
def album_detail(
    album_id: int,
    request: Request,
    page: int = 1,
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> HTMLResponse:
    """Render the detail page for a single album, with paginated photo grid and lightbox."""
    album = get_album(conn, album_id)
    if album is None:
        raise HTTPException(status_code=404, detail=f"Album {album_id} not found")

    page = max(1, page)
    total = count_album_assets(conn, album_id)
    total_pages = max(1, (total + _PAGE_SIZE - 1) // _PAGE_SIZE)
    page = min(page, total_pages)
    offset = (page - 1) * _PAGE_SIZE

    assets = get_album_assets(conn, album_id, limit=_PAGE_SIZE, offset=offset)

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "album_detail.html",
        {
            "request": request,
            "album": album,
            "assets": assets,
            "page": page,
            "total_pages": total_pages,
            "total": total,
        },
    )
