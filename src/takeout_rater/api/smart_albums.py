"""FastAPI router for smart album suggestions (pages + JSON API).

Pages
-----
GET  /smart-albums                      – suggestions and kept albums
GET  /smart-albums/{id}                 – one album with a photo grid

JSON
----
GET    /api/smart-albums                – list
POST   /api/smart-albums/{id}/rename    – rename (also keeps the album)
POST   /api/smart-albums/{id}/keep      – keep / un-keep
DELETE /api/smart-albums/{id}           – dismiss
GET    /api/assets/{id}/smart-albums    – albums containing one photo

The suggestion job itself is started via ``POST /api/jobs/smart_albums/start``
(see :mod:`takeout_rater.api.jobs`).
"""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from takeout_rater.api.albums import _get_conn
from takeout_rater.db import smart_album_queries as sq
from takeout_rater.smart_albums.suggest import DEFAULT_PARAMS

router = APIRouter()

_PAGE_SIZE = 100


class SmartAlbumsBody(BaseModel):
    """Settings accepted by ``POST /api/jobs/smart_albums/start``."""

    n_albums: int = Field(DEFAULT_PARAMS.n_albums, ge=2, le=300)
    min_photos: int = Field(DEFAULT_PARAMS.min_photos, ge=1, le=10000)
    terms_per_name: int = Field(DEFAULT_PARAMS.terms_per_name, ge=1, le=3)


@router.get("/smart-albums", response_class=HTMLResponse)
def smart_albums_page(
    request: Request,
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> HTMLResponse:
    """Render kept smart albums and the current suggestions."""
    albums = sq.list_smart_albums(conn)
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "smart_albums.html",
        {
            "request": request,
            "kept": [a for a in albums if a["kept"]],
            "suggested": [a for a in albums if not a["kept"]],
            "stats": sq.smart_album_stats(conn),
            "defaults": DEFAULT_PARAMS,
        },
    )


@router.get("/smart-albums/{album_id}", response_class=HTMLResponse)
def smart_album_detail(
    album_id: int,
    request: Request,
    page: int = 1,
    sort: str = "typical",
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> HTMLResponse:
    """Render one smart album: header, matched terms and a paginated photo grid."""
    album = sq.get_smart_album(conn, album_id)
    if album is None:
        raise HTTPException(status_code=404, detail=f"Smart album {album_id} not found")
    sort = "date" if sort == "date" else "typical"
    total = album["n_assets"]
    total_pages = max(1, (total + _PAGE_SIZE - 1) // _PAGE_SIZE)
    page = min(max(1, page), total_pages)
    assets = sq.get_smart_album_assets(
        conn, album_id, sort=sort, limit=_PAGE_SIZE, offset=(page - 1) * _PAGE_SIZE
    )
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "smart_album_detail.html",
        {
            "request": request,
            "album": album,
            "assets": assets,
            "page": page,
            "total_pages": total_pages,
            "total": total,
            "sort": sort,
        },
    )


# ── JSON API ──────────────────────────────────────────────────────────────────


@router.get("/api/smart-albums")
def api_list_smart_albums(
    kept: bool | None = None,
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse:
    """List smart albums (kept first, then by size)."""
    return JSONResponse(
        {"items": sq.list_smart_albums(conn, kept=kept), "stats": sq.smart_album_stats(conn)}
    )


@router.get("/api/assets/{asset_id}/smart-albums")
def api_asset_smart_albums(
    asset_id: int,
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse:
    """Smart albums that contain one photo."""
    return JSONResponse({"items": sq.get_smart_albums_for_asset(conn, asset_id)})


class _RenameBody(BaseModel):
    name: str | None = Field(None, max_length=200)


class _KeepBody(BaseModel):
    kept: bool = True


def _album_or_404(conn: sqlite3.Connection, album_id: int) -> dict:
    album = sq.get_smart_album(conn, album_id)
    if album is None:
        raise HTTPException(status_code=404, detail=f"Smart album {album_id} not found")
    return album


@router.post("/api/smart-albums/{album_id}/rename")
def api_rename_smart_album(
    album_id: int,
    body: _RenameBody,
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse:
    """Rename an album (which also keeps it); an empty name restores the suggested one."""
    if not sq.rename_smart_album(conn, album_id, body.name):
        raise HTTPException(status_code=404, detail=f"Smart album {album_id} not found")
    album = _album_or_404(conn, album_id)
    return JSONResponse({"id": album_id, "name": album["name"], "kept": album["kept"]})


@router.post("/api/smart-albums/{album_id}/keep")
def api_keep_smart_album(
    album_id: int,
    body: _KeepBody,
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse:
    """Keep an album across re-runs, or turn it back into a suggestion."""
    if not sq.set_smart_album_kept(conn, album_id, body.kept):
        raise HTTPException(status_code=404, detail=f"Smart album {album_id} not found")
    return JSONResponse({"id": album_id, "kept": body.kept})


@router.delete("/api/smart-albums/{album_id}")
def api_delete_smart_album(
    album_id: int,
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse:
    """Dismiss a suggestion or delete a kept album (photos are not touched)."""
    if not sq.delete_smart_album(conn, album_id):
        raise HTTPException(status_code=404, detail=f"Smart album {album_id} not found")
    return JSONResponse({"id": album_id, "deleted": True})
