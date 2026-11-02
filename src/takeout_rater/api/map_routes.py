"""FastAPI router for the map view (photos arranged by geolocation and time)."""

from __future__ import annotations

import gzip
import json
import sqlite3

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from takeout_rater.api.assets import _get_conn
from takeout_rater.db.queries import count_map_candidates, list_geotagged_points

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
) -> Response:
    """Return every geotagged asset as compact columnar arrays.

    Query parameters:
    - ``favorited``: ``"1"`` to restrict to favorited assets.
    - ``dedupe``: ``"0"`` to include byte-identical duplicates; defaults to ``"1"``.

    Response (arrays are index-aligned and sorted by ``taken_at``, NULLs first)::

        {
          "count": N,
          "total_assets": M,        # assets passing the same filters, with or
                                    # without a location (for coverage stats)
          "ids": [...], "lat": [...], "lon": [...],
          "t": [...],               # Unix seconds or null
          "fav": [...]              # 0/1
        }

    Columnar arrays keep the payload compact (~40-60 bytes per photo as JSON),
    and the response is gzip-compressed when the client accepts it, which
    shrinks a 200k-photo library from ~10 MB to a few MB.  Clustering happens
    client-side.
    """
    fav_filter: bool | None = True if favorited == "1" else None
    rows = list_geotagged_points(conn, favorited=fav_filter, deduped=dedupe != "0")
    total = count_map_candidates(conn, favorited=fav_filter, deduped=dedupe != "0")
    payload = {
        "count": len(rows),
        "total_assets": total,
        "ids": [r[0] for r in rows],
        "lat": [round(r[1], 5) for r in rows],
        "lon": [round(r[2], 5) for r in rows],
        "t": [r[3] for r in rows],
        "fav": [r[4] for r in rows],
    }
    body = json.dumps(payload, separators=(",", ":")).encode()
    headers = {"Vary": "Accept-Encoding"}
    if len(body) > 1024 and "gzip" in request.headers.get("accept-encoding", ""):
        body = gzip.compress(body, compresslevel=5)
        headers["Content-Encoding"] = "gzip"
    return Response(content=body, media_type="application/json", headers=headers)
