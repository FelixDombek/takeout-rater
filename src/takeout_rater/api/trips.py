"""FastAPI router for detected trips and events (pages + JSON API)."""

from __future__ import annotations

import logging
import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from takeout_rater.api.albums import _get_conn
from takeout_rater.db import event_queries as eq
from takeout_rater.events import chapters as ch
from takeout_rater.events.detector import DEFAULT_PARAMS, format_date_range, local_year

_log = logging.getLogger(__name__)

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


def _sensitivity(value: str) -> str:
    return value if value in ch.SENSITIVITY else "normal"


def _build_story(
    conn: sqlite3.Connection, ev: dict, sensitivity: str
) -> tuple[ch.StoryResult, list[dict]]:
    """Split one trip/event into chapters; returns the result and its photos in time order."""
    assets = eq.get_event_assets(conn, ev["id"], limit=-1, offset=0)
    photos = [(a["id"], a["taken_at"]) for a in assets if a["taken_at"] is not None]
    story = ch.split_chapters(
        photos,
        eq.load_event_embeddings(conn, ev["id"]),
        ch.params_for(sensitivity),
        min_photos=ch.default_min_photos(len(photos), sensitivity),
        aesthetic=eq.load_event_aesthetic_scores(conn, ev["id"]),
    )
    by_id = {a["id"]: a for a in assets}
    return story, [by_id[aid] for aid, _ in sorted(photos, key=lambda p: (p[1], p[0]))]


def _chapter_dicts(ev: dict, story: ch.StoryResult) -> list[dict]:
    multi_day = ev["n_days"] > 1
    out = []
    for i, c in enumerate(story.chapters):
        out.append(
            {
                "index": i + 1,
                "start_ts": c.start_ts,
                "end_ts": c.end_ts,
                "n_assets": len(c.asset_ids),
                "asset_ids": c.asset_ids,
                "cover_asset_id": c.cover_asset_id,
                "break_score": c.break_score,
                "time_label": ch.time_label(
                    c.start_ts, c.end_ts, ev["start_ts"], ev["center_lon"], show_day=multi_day
                ),
            }
        )
    return out


def _break_curve(story: ch.StoryResult, width: int = 1000, height: int = 48) -> dict | None:
    """SVG geometry of the break-score curve with chapter cut markers (``None`` if < 3 photos)."""
    scores = story.scores
    if len(scores) < 2:
        return None
    top = max(max(scores), story.threshold) or 1.0
    step = width / (len(scores) - 1)

    def y(v: float) -> float:
        return round(height - 2 - (height - 4) * v / top, 1)

    points = " ".join(f"{round(i * step, 1)},{y(v)}" for i, v in enumerate(scores))
    cuts = [
        {"x": round((c.start - 1) * step, 1), "index": i + 1}
        for i, c in enumerate(story.chapters)
        if c.start > 0
    ]
    return {
        "width": width,
        "height": height,
        "points": points,
        "threshold_y": y(story.threshold),
        "cuts": cuts,
    }


def _chapter_labels(request: Request, story: ch.StoryResult) -> list[str] | None:
    """Zero-shot CLIP titles per chapter, or ``None`` when CLIP is unavailable."""
    if not any(c.centroid is not None for c in story.chapters):
        return None
    try:
        from takeout_rater.api.assets import _get_clip_vocab_matrix

        vocab = _get_clip_vocab_matrix(request)
    except Exception:  # noqa: BLE001 - labels are a nice-to-have
        _log.warning("Could not load the CLIP vocabulary for chapter titles", exc_info=True)
        return None
    if vocab is None:
        return None
    matrix, terms = vocab
    if matrix.shape[1] != story.chapters[0].centroid.shape[0]:
        return None
    labels = ch.label_chapters(story.chapters, matrix, terms, allowed=ch.label_terms())
    return [ch.chapter_title(t) for t in labels]


@router.get("/trips/{event_id}", response_class=HTMLResponse)
def trip_detail(
    event_id: int,
    request: Request,
    page: int = 1,
    view: str = "grid",
    sensitivity: str = "normal",
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> HTMLResponse:
    """Render one trip/event: header, nested events and either a paginated photo
    grid (``view=grid``) or the photos split into story chapters (``view=story``)."""
    ev = eq.get_event(conn, event_id)
    if ev is None:
        raise HTTPException(status_code=404, detail=f"Trip/event {event_id} not found")
    ev = _decorate(ev)
    parent = (
        _decorate(p) if ev["parent_id"] and (p := eq.get_event(conn, ev["parent_id"])) else None
    )
    children = [_decorate(c) for c in eq.list_events(conn, parent_id=event_id)]

    total = ev["n_assets"]
    templates = request.app.state.templates
    ctx: dict = {
        "request": request,
        "ev": ev,
        "parent": parent,
        "children": children,
        "total": total,
        "view": "story" if view == "story" else "grid",
        "sensitivity": _sensitivity(sensitivity),
    }
    if ctx["view"] == "story":
        story, ordered = _build_story(conn, ev, ctx["sensitivity"])
        chapters = _chapter_dicts(ev, story)
        pos = 0
        for c in chapters:
            c["assets"] = ordered[pos : pos + c["n_assets"]]
            pos += c["n_assets"]
        ctx.update(
            chapters=chapters,
            curve=_break_curve(story),
            n_with_embedding=story.n_with_embedding,
            n_story_photos=story.n_photos,
        )
    else:
        total_pages = max(1, (total + _PAGE_SIZE - 1) // _PAGE_SIZE)
        page = min(max(1, page), total_pages)
        ctx.update(
            assets=eq.get_event_assets(
                conn, event_id, limit=_PAGE_SIZE, offset=(page - 1) * _PAGE_SIZE
            ),
            page=page,
            total_pages=total_pages,
        )
    return templates.TemplateResponse("trip_detail.html", ctx)


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


@router.get("/api/trips/{event_id}/chapters")
def api_trip_chapters(
    event_id: int,
    request: Request,
    sensitivity: str = "normal",
    labels: bool = False,
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse:
    """Story chapters of one trip/event.

    Query parameters:
        sensitivity: ``fewer``, ``normal`` (default) or ``more`` chapters.
        labels: Also name each chapter with zero-shot CLIP vocabulary terms
            (loads the CLIP text model on first use, so the story page asks
            for this after it has rendered).
    """
    ev = eq.get_event(conn, event_id)
    if ev is None:
        raise HTTPException(status_code=404, detail=f"Trip/event {event_id} not found")
    sensitivity = _sensitivity(sensitivity)
    story, _ = _build_story(conn, ev, sensitivity)
    chapters = _chapter_dicts(ev, story)
    titles = _chapter_labels(request, story) if labels else None
    for i, c in enumerate(chapters):
        c["title"] = titles[i] if titles else None
    return JSONResponse(
        {
            "event_id": event_id,
            "sensitivity": sensitivity,
            "n_photos": story.n_photos,
            "n_with_embedding": story.n_with_embedding,
            "threshold": story.threshold,
            "break_scores": story.scores,
            "chapters": chapters,
        }
    )


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
