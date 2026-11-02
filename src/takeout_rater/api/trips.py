"""FastAPI router for detected trips and events (pages + JSON API)."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from takeout_rater.api.albums import _get_conn
from takeout_rater.db import event_queries as eq
from takeout_rater.events.detector import DEFAULT_PARAMS, format_date_range, local_year

router = APIRouter()

_PAGE_SIZE = 100


def _decorate(ev: dict) -> dict:
    """Add display-only fields used by templates and the JSON API."""
    ev = dict(ev)
    ev["date_label"] = format_date_range(ev["start_ts"], ev["end_ts"], ev["center_lon"])
    ev["year"] = local_year(ev["start_ts"], ev["center_lon"])
    ev["map_url"] = f"/map?from={ev['start_ts']}&to={ev['end_ts']}"
    return ev


@router.get("/trips", response_class=HTMLResponse)
def trips_page(
    request: Request,
    show: str = "all",
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> HTMLResponse:
    """Render the trips & events overview, grouped by year (newest first)."""
    kind = {"trips": "trip", "events": "event"}.get(show)
    items = [_decorate(e) for e in eq.list_events(conn, kind=kind)]
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
            "defaults": DEFAULT_PARAMS,
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
    ev = _decorate(ev)
    parent = (
        _decorate(p) if ev["parent_id"] and (p := eq.get_event(conn, ev["parent_id"])) else None
    )
    children = [_decorate(c) for c in eq.list_events(conn, parent_id=event_id)]

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
    items = [_decorate(e) for e in eq.list_events(conn, kind=kind, top_level_only=not nested)]
    return JSONResponse({"items": items, "stats": eq.event_stats(conn)})


@router.get("/api/assets/{asset_id}/trips")
def api_asset_trips(
    asset_id: int,
    request: Request,
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse:
    """Trips/events that contain one photo."""
    items = [_decorate(e) for e in eq.get_events_for_asset(conn, asset_id)]
    return JSONResponse({"items": items})


class TripDetectBody(BaseModel):
    """Detection thresholds accepted by ``POST /api/jobs/trips/start``."""

    trip_min_distance_km: float = Field(DEFAULT_PARAMS.trip_min_distance_km, gt=0, le=20000)
    trip_min_photos: int = Field(DEFAULT_PARAMS.trip_min_photos, ge=1)
    trip_max_gap_days: int = Field(DEFAULT_PARAMS.trip_max_gap_days, ge=0, le=30)
    trip_max_days: int = Field(DEFAULT_PARAMS.trip_max_days, ge=1, le=3650)
    event_gap_hours: float = Field(DEFAULT_PARAMS.event_gap_hours, gt=0, le=240)
    event_min_photos: int = Field(DEFAULT_PARAMS.event_min_photos, ge=2)


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
