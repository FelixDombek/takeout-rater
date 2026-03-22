"""FastAPI router for asset listing, detail, and thumbnail serving."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from takeout_rater.db.queries import (
    AssetRow,
    count_assets,
    count_assets_with_score,
    get_asset_by_id,
    get_asset_scores,
    list_assets,
    list_assets_by_score,
)
from takeout_rater.indexing.thumbnailer import thumb_path_for_id
from takeout_rater.scorers.registry import list_specs

router = APIRouter()

_PAGE_SIZE = 50


def _get_conn(request: Request) -> sqlite3.Connection:
    """Dependency: retrieve the DB connection from the app state."""
    return request.app.state.db_conn


def _get_takeout_root(request: Request) -> Path:
    """Dependency: retrieve the takeout root path from the app state."""
    return request.app.state.takeout_root


def _get_thumbs_dir(request: Request) -> Path:
    """Dependency: retrieve the thumbs directory path from the app state."""
    return request.app.state.thumbs_dir


def _parse_sort_by(sort_by: str | None) -> tuple[str, str] | None:
    """Parse a ``sort_by`` query parameter of the form ``scorer_id:metric_key``.

    Returns a ``(scorer_id, metric_key)`` tuple or ``None`` if *sort_by* is
    absent or malformed.
    """
    if not sort_by:
        return None
    parts = sort_by.split(":", 1)
    if len(parts) != 2:
        return None
    scorer_id, metric_key = parts
    if not scorer_id or not metric_key:
        return None
    return scorer_id, metric_key


@router.get("/assets", response_class=HTMLResponse)
def browse_assets(
    request: Request,
    page: int = 1,
    favorited: str | None = None,
    sort_by: str | None = None,
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> HTMLResponse:
    """Render the browse page with paginated asset thumbnails.

    Query parameters:
    - ``page``: 1-based page number (default 1).
    - ``favorited``: ``"1"`` to show only favorited assets.
    - ``sort_by``: ``"scorer_id:metric_key"`` to sort by a score metric
      (e.g. ``"blur:sharpness"``).  Only scored assets are shown.
    """
    offset = max(0, (page - 1) * _PAGE_SIZE)
    fav_filter: bool | None = True if favorited == "1" else None

    sort_parsed = _parse_sort_by(sort_by)

    # Score map: asset_id → score value (populated when sorting by score)
    score_map: dict[int, float] = {}

    if sort_parsed is not None:
        scorer_id, metric_key = sort_parsed
        asset_score_pairs = list_assets_by_score(
            conn,
            scorer_id,
            metric_key,
            limit=_PAGE_SIZE,
            offset=offset,
            favorited=fav_filter,
        )
        assets = [a for a, _ in asset_score_pairs]
        score_map = {a.id: s for a, s in asset_score_pairs}
        total = count_assets_with_score(conn, scorer_id, metric_key, favorited=fav_filter)
    else:
        assets = list_assets(conn, limit=_PAGE_SIZE, offset=offset, favorited=fav_filter)
        total = count_assets(conn, favorited=fav_filter)

    total_pages = max(1, (total + _PAGE_SIZE - 1) // _PAGE_SIZE)

    # Build sort options from registered scorer specs
    sort_options = [
        (f"{spec.scorer_id}:{m.key}", f"{spec.display_name} – {m.display_name}")
        for spec in list_specs()
        for m in spec.metrics
        if spec.scorer_id != "dummy"
    ]

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "browse.html",
        {
            "request": request,
            "assets": assets,
            "page": page,
            "total_pages": total_pages,
            "total": total,
            "favorited": favorited,
            "sort_by": sort_by,
            "sort_options": sort_options,
            "score_map": score_map,
        },
    )


@router.get("/assets/{asset_id}", response_class=HTMLResponse)
def asset_detail(
    asset_id: int,
    request: Request,
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> HTMLResponse:
    """Render the detail page for a single asset."""
    asset: AssetRow | None = get_asset_by_id(conn, asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail=f"Asset {asset_id} not found")

    scores = get_asset_scores(conn, asset_id)

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "detail.html",
        {"request": request, "asset": asset, "scores": scores},
    )


@router.get("/thumbs/{asset_id}")
def serve_thumbnail(
    asset_id: int,
    request: Request,
    thumbs_dir: Path = Depends(_get_thumbs_dir),  # noqa: B008
) -> FileResponse:
    """Serve the JPEG thumbnail for an asset.

    Returns 404 if the thumbnail has not been generated yet.
    """
    thumb = thumb_path_for_id(thumbs_dir, asset_id)
    if not thumb.exists():
        raise HTTPException(status_code=404, detail=f"Thumbnail for asset {asset_id} not found")
    return FileResponse(str(thumb), media_type="image/jpeg")
