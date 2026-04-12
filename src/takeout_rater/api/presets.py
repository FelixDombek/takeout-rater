"""FastAPI router for view-preset CRUD endpoints."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from takeout_rater.db.queries import (
    delete_view_preset,
    get_view_preset,
    list_view_presets,
    upsert_view_preset,
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


class PresetBody(BaseModel):
    """Request body for creating or updating a view preset."""

    name: str
    sort_by: str | None = None
    favorited: int | None = None
    min_score: float | None = None
    max_score: float | None = None


@router.get("/api/presets")
def api_list_presets(
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse:
    """Return all saved view presets as a JSON array."""
    presets = list_view_presets(conn)
    return JSONResponse(
        [
            {
                "id": p.id,
                "name": p.name,
                "sort_by": p.sort_by,
                "favorited": p.favorited,
                "min_score": p.min_score,
                "max_score": p.max_score,
            }
            for p in presets
        ]
    )


@router.post("/api/presets", status_code=201)
def api_upsert_preset(
    body: PresetBody,
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse:
    """Create or update a view preset by name.

    If a preset with the given *name* already exists it is updated in place.
    """
    if not body.name.strip():
        raise HTTPException(status_code=422, detail="Preset name must not be blank")
    preset_id = upsert_view_preset(
        conn,
        name=body.name.strip(),
        sort_by=body.sort_by,
        favorited=body.favorited,
        min_score=body.min_score,
        max_score=body.max_score,
    )
    preset = get_view_preset(conn, preset_id)
    assert preset is not None  # just inserted
    return JSONResponse(
        {
            "id": preset.id,
            "name": preset.name,
            "sort_by": preset.sort_by,
            "favorited": preset.favorited,
            "min_score": preset.min_score,
            "max_score": preset.max_score,
        },
        status_code=201,
    )


@router.delete("/api/presets/{preset_id}", status_code=204)
def api_delete_preset(
    preset_id: int,
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse:
    """Delete a view preset by ID.  Returns 404 if the preset does not exist."""
    deleted = delete_view_preset(conn, preset_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Preset {preset_id} not found")
    return JSONResponse(None, status_code=204)
