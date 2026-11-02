"""FastAPI router for detected trips and events (pages + JSON API)."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from takeout_rater.db import event_queries as eq
from takeout_rater.events.detector import DetectionParams, format_date_range

router = APIRouter()

_PAGE_SIZE = 100


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


def _has_map(request: Request) -> bool:
    """Whether the map page (``/map``) is available in this build."""
    return any(getattr(r, "path", None) == "/map" for r in request.app.routes)


def _decorate(ev: dict, has_map: bool) -> dict:
    """Add display-only fields used by templates and the JSON API.

    ``map_url`` is ``None`` when the map page is not available.
    """
    ev = dict(ev)
    ev["date_label"] = format_date_range(ev["start_ts"], ev["end_ts"], ev["center_lon"])
    ev["year"] = datetime.fromtimestamp(ev["start_ts"], tz=UTC).year
    ev["map_url"] = f"/map?from={ev['start_ts']}&to={ev['end_ts']}" if has_map else None
    return ev


@router.get("/trips", response_class=HTMLResponse)
def trips_page(
    request: Request,
    show: str = "all",
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> HTMLResponse:
    """Render the trips & events overview, grouped by year (newest first)."""
    kind = {"trips": "trip", "events": "event"}.get(show)
    has_map = _has_map(request)
    items = [_decorate(e, has_map) for e in eq.list_events(conn, kind=kind)]
    years: dict[int, list[dict]] = {}
    for ev in items:
        years.setdefault(ev["year"], []).append(ev)
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "trips.html",
        {
            "request": request,
            "years": years,
            "show": show if kind else "all",
            "stats": eq.event_stats(conn),
            "defaults": DetectionParams(),
        },
    )


@router.get("/trips/{event_id}", response_class=HTMLResponse)
def trip_detail(
    event_id: int,
    request: Request,
    page: int = 1,
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> HTMLResponse:
    """Render one trip/event: header, nested events and a paginated photo grid."""
    ev = eq.get_event(conn, event_id)
    if ev is None:
        raise HTTPException(status_code=404, detail=f"Trip/event {event_id} not found")
    has_map = _has_map(request)
    ev = _decorate(ev, has_map)
    parent = (
        _decorate(p, has_map)
        if ev["parent_id"] and (p := eq.get_event(conn, ev["parent_id"]))
        else None
    )
    children = [_decorate(c, has_map) for c in eq.list_events(conn, parent_id=event_id)]

    total = ev["n_assets"]
    total_pages = max(1, (total + _PAGE_SIZE - 1) // _PAGE_SIZE)
    page = min(max(1, page), total_pages)
    assets = eq.get_event_assets(conn, event_id, limit=_PAGE_SIZE, offset=(page - 1) * _PAGE_SIZE)
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "trip_detail.html",
        {
            "request": request,
            "ev": ev,
            "parent": parent,
            "children": children,
            "assets": assets,
            "page": page,
            "total_pages": total_pages,
            "total": total,
        },
    )


# ── JSON API ──────────────────────────────────────────────────────────────────


@router.get("/api/trips")
def api_list_trips(
    request: Request,
    kind: str | None = None,
    nested: bool = False,
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse:
    """List trips/events as JSON (newest first).

    Query parameters:
        kind: ``trip`` or ``event`` to filter.
        nested: Include events that sit inside a trip (default: top level only).
    """
    if kind not in (None, "trip", "event"):
        raise HTTPException(status_code=400, detail="kind must be 'trip' or 'event'")
    has_map = _has_map(request)
    items = [
        _decorate(e, has_map) for e in eq.list_events(conn, kind=kind, top_level_only=not nested)
    ]
    return JSONResponse({"items": items, "stats": eq.event_stats(conn)})


@router.get("/api/assets/{asset_id}/trips")
def api_asset_trips(
    asset_id: int,
    request: Request,
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse:
    """Trips/events that contain one photo."""
    has_map = _has_map(request)
    items = [_decorate(e, has_map) for e in eq.get_events_for_asset(conn, asset_id)]
    return JSONResponse({"items": items})


class _DetectBody(BaseModel):
    trip_min_distance_km: float = Field(DetectionParams.trip_min_distance_km, gt=0, le=20000)
    trip_min_photos: int = Field(DetectionParams.trip_min_photos, ge=1)
    trip_max_gap_days: int = Field(DetectionParams.trip_max_gap_days, ge=0, le=30)
    event_gap_hours: float = Field(DetectionParams.event_gap_hours, gt=0, le=240)
    event_min_photos: int = Field(DetectionParams.event_min_photos, ge=2)


@router.post("/api/trips/detect")
def api_detect_trips(
    body: _DetectBody | None = None,
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse:
    """(Re-)detect all trips and events.  Runs synchronously (a few seconds for 200k photos)."""
    from takeout_rater.events.service import run_detection

    body = body or _DetectBody()
    params = DetectionParams(**body.model_dump())
    summary = run_detection(conn, params)
    return JSONResponse(summary)


class _RenameBody(BaseModel):
    name: str | None = Field(None, max_length=200)


@router.post("/api/trips/{event_id}/rename")
def api_rename_trip(
    event_id: int,
    body: _RenameBody,
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse:
    """Rename a trip/event; an empty name restores the automatic name."""
    if not eq.rename_event(conn, event_id, body.name):
        raise HTTPException(status_code=404, detail=f"Trip/event {event_id} not found")
    ev = eq.get_event(conn, event_id)
    return JSONResponse({"id": event_id, "name": ev["name"] if ev else None})
