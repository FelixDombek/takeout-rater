"""FastAPI router for the map view (photos arranged by geolocation and time)."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from takeout_rater.api.assets import _get_conn
from takeout_rater.db.queries import list_geotagged_points

router = APIRouter()


@router.get("/map", response_class=HTMLResponse)
def map_page(request: Request) -> HTMLResponse:
    """Render the map view page."""
    if request.app.state.db_conn is None:
        return RedirectResponse(url="/setup")  # type: ignore[return-value]
    templates = request.app.state.templates
    return templates.TemplateResponse("map.html", {"request": request})


@router.get("/api/map/points")
def map_points(
    request: Request,
    favorited: str | None = None,
    dedupe: str = "1",
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse:
    """Return every geotagged asset as compact columnar arrays.

    Query parameters:
    - ``favorited``: ``"1"`` to restrict to favorited assets.
    - ``dedupe``: ``"0"`` to include byte-identical duplicates; defaults to ``"1"``.

    Response (arrays are index-aligned and sorted by ``taken_at``, NULLs first)::

        {
          "count": N,
          "total_assets": M,        # all non-trashed assets, for coverage stats
          "ids": [...], "lat": [...], "lon": [...],
          "t": [...],               # Unix seconds or null
          "fav": [...]              # 0/1
        }

    Columnar arrays keep the payload small (~25 bytes per photo), so a 200k
    photo library transfers in a few MB and is clustered client-side.
    """
    fav_filter: bool | None = True if favorited == "1" else None
    rows = list_geotagged_points(conn, favorited=fav_filter, deduped=dedupe != "0")
    total = conn.execute("SELECT COUNT(*) FROM assets WHERE COALESCE(trashed, 0) = 0").fetchone()[0]
    return JSONResponse(
        {
            "count": len(rows),
            "total_assets": total,
            "ids": [r[0] for r in rows],
            "lat": [round(r[1], 5) for r in rows],
            "lon": [round(r[2], 5) for r in rows],
            "t": [r[3] for r in rows],
            "fav": [r[4] for r in rows],
        }
    )
