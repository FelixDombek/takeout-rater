"""FastAPI router for asset listing, detail, and thumbnail serving."""

from __future__ import annotations

import datetime
import json
import sqlite3
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from takeout_rater.db.queries import (
    AssetRow,
    count_assets,
    count_assets_deduped,
    count_assets_newer_than,
    count_assets_with_score,
    get_asset_by_id,
    get_asset_scores,
    get_duplicate_assets,
    get_taken_at_range,
    list_assets,
    list_assets_by_score,
    list_assets_deduped,
    list_view_presets,
)
from takeout_rater.indexing.thumbnailer import thumb_path_for_id
from takeout_rater.scorers.registry import list_specs

router = APIRouter()

_PAGE_SIZE = 50


def _get_conn(request: Request) -> sqlite3.Connection:
    """Dependency: retrieve the DB connection from the app state.

    Redirects to the setup page if the library has not been configured yet.
    """
    conn = request.app.state.db_conn
    if conn is None:
        raise HTTPException(status_code=503, detail="Library not configured — visit /setup")
    return conn


def _get_takeout_root(request: Request) -> Path:
    """Dependency: retrieve the takeout root path from the app state."""
    return request.app.state.takeout_root


def _get_thumbs_dir(request: Request) -> Path:
    """Dependency: retrieve the thumbs directory path from the app state."""
    thumbs_dir = request.app.state.thumbs_dir
    if thumbs_dir is None:
        raise HTTPException(status_code=503, detail="Library not configured — visit /setup")
    return thumbs_dir


def _read_sidecar_json(takeout_root: Path | None, asset: AssetRow) -> str | None:
    """Return the pretty-printed raw sidecar JSON for *asset*, or ``None``.

    Reads ``takeout_root / asset.sidecar_relpath`` when both values are
    available and the file exists.  Returns ``None`` on any I/O or parse error
    so that the UI can gracefully degrade.
    """
    if takeout_root is None or asset.sidecar_relpath is None:
        return None
    sidecar_path = takeout_root / asset.sidecar_relpath
    try:
        raw = json.loads(sidecar_path.read_text(encoding="utf-8"))
        return json.dumps(raw, indent=2, ensure_ascii=False)
    except (OSError, json.JSONDecodeError):
        return None


def _parse_score(raw: str | None) -> float | None:
    """Parse a score query parameter, treating blank/missing values as ``None``.

    Silently ignores non-numeric values rather than raising a validation error,
    so that submitting an empty ``min_score=`` or ``max_score=`` form field
    never causes a 422 response.
    """
    if not raw or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        return None


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
    min_score: str | None = None,
    max_score: str | None = None,
    dedupe: str = "1",
    partial: str = "0",
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> HTMLResponse:
    """Render the browse page with paginated asset thumbnails.

    Query parameters:
    - ``page``: 1-based page number (default 1).
    - ``favorited``: ``"1"`` to show only favorited assets.
    - ``sort_by``: ``"scorer_id:metric_key"`` to sort by a score metric
      (e.g. ``"blur:sharpness"``).  Only scored assets are shown.
    - ``min_score``: Inclusive lower bound on the score value (requires
      ``sort_by``).  Blank or non-numeric values are silently ignored.
    - ``max_score``: Inclusive upper bound on the score value (requires
      ``sort_by``).  Blank or non-numeric values are silently ignored.
    - ``dedupe``: ``"1"`` (default) to hide exact duplicate files (same SHA-256
      content hash); ``"0"`` to show all physical copies.
    - ``partial``: ``"1"`` to return only a card fragment (for infinite scroll
      fetch requests); ``"0"`` (default) to return the full page.
    """
    offset = max(0, (page - 1) * _PAGE_SIZE)
    fav_filter: bool | None = True if favorited == "1" else None
    dedup_enabled = dedupe != "0"

    sort_parsed = _parse_sort_by(sort_by)
    # Normalize sort_by: use the canonical form when valid, None when malformed.
    # This prevents untrusted query text from being propagated into template links.
    canonical_sort_by: str | None = None

    # Score map: asset_id → score value (populated when sorting by score)
    score_map: dict[int, float] = {}

    # Score range is only meaningful when sorting by score; parse safely to float.
    eff_min = _parse_score(min_score) if sort_parsed else None
    eff_max = _parse_score(max_score) if sort_parsed else None

    if sort_parsed is not None:
        scorer_id, metric_key = sort_parsed
        canonical_sort_by = f"{scorer_id}:{metric_key}"
        asset_score_pairs = list_assets_by_score(
            conn,
            scorer_id,
            metric_key,
            limit=_PAGE_SIZE,
            offset=offset,
            favorited=fav_filter,
            min_score=eff_min,
            max_score=eff_max,
        )
        assets = [a for a, _ in asset_score_pairs]
        score_map = {a.id: s for a, s in asset_score_pairs}
        total = count_assets_with_score(
            conn,
            scorer_id,
            metric_key,
            favorited=fav_filter,
            min_score=eff_min,
            max_score=eff_max,
        )
    elif dedup_enabled:
        assets = list_assets_deduped(conn, limit=_PAGE_SIZE, offset=offset, favorited=fav_filter)
        total = count_assets_deduped(conn, favorited=fav_filter)
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

    # Load saved presets for the toolbar
    presets = list_view_presets(conn)

    templates = request.app.state.templates
    if partial == "1":
        return templates.TemplateResponse(
            "browse_partial.html",
            {
                "request": request,
                "assets": assets,
                "page": page,
                "total_pages": total_pages,
                "score_map": score_map,
            },
        )
    return templates.TemplateResponse(
        "browse.html",
        {
            "request": request,
            "assets": assets,
            "page": page,
            "total_pages": total_pages,
            "total": total,
            "favorited": favorited,
            "sort_by": canonical_sort_by,
            "sort_options": sort_options,
            "score_map": score_map,
            "min_score": eff_min,
            "max_score": eff_max,
            "presets": presets,
            # dedupe is only actively applied when not sorting by score metric
            "dedupe": dedup_enabled and sort_parsed is None,
        },
    )


@router.get("/assets/{asset_id}", response_class=HTMLResponse)
def asset_detail(
    asset_id: int,
    request: Request,
    partial: str = "0",
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
    takeout_root: Path | None = Depends(_get_takeout_root),  # noqa: B008
) -> HTMLResponse:
    """Render the detail page for a single asset.

    Query parameters:
    - ``partial``: ``"1"`` to return only the detail fragment (for the
      lightbox panel); ``"0"`` (default) to return the full page.
    """
    asset: AssetRow | None = get_asset_by_id(conn, asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail=f"Asset {asset_id} not found")

    scores = get_asset_scores(conn, asset_id)

    # Collect all physical copies that share the same SHA-256 hash so the
    # detail view can list every path/album where this image appears.
    duplicates: list[AssetRow] = []
    if asset.sha256 is not None:
        all_copies = get_duplicate_assets(conn, asset.sha256)
        duplicates = [a for a in all_copies if a.id != asset_id]

    # Load raw sidecar JSON for the main asset and each duplicate.
    sidecar_json = _read_sidecar_json(takeout_root, asset)
    duplicate_sidecars: list[tuple[AssetRow, str | None]] = [
        (dup, _read_sidecar_json(takeout_root, dup)) for dup in duplicates
    ]

    templates = request.app.state.templates
    ctx = {
        "request": request,
        "asset": asset,
        "scores": scores,
        "duplicates": duplicates,
        "sidecar_json": sidecar_json,
        "duplicate_sidecars": duplicate_sidecars,
    }
    if partial == "1":
        return templates.TemplateResponse("detail_partial.html", ctx)
    return templates.TemplateResponse("detail.html", ctx)


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


@router.get("/api/timeline")
def get_timeline(
    request: Request,
    favorited: str | None = None,
    dedupe: str = "1",
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse:
    """Return the year range of photos available in the library.

    Query parameters mirror those of ``/assets`` so the timeline always reflects
    the currently active view filter.

    - ``favorited``: ``"1"`` to restrict to favorited assets.
    - ``dedupe``: ``"0"`` to include duplicate files; defaults to ``"1"``.

    Returns a JSON object with:
    - ``has_data``: ``false`` when no photos have ``taken_at`` metadata.
    - ``min_year``: integer – oldest year with photos.
    - ``max_year``: integer – newest year with photos.
    """
    min_ts, max_ts = get_taken_at_range(conn)
    if min_ts is None or max_ts is None:
        return JSONResponse({"has_data": False})

    min_year = datetime.datetime.fromtimestamp(min_ts, tz=datetime.UTC).year
    max_year = datetime.datetime.fromtimestamp(max_ts, tz=datetime.UTC).year

    return JSONResponse({"has_data": True, "min_year": min_year, "max_year": max_year})


@router.get("/api/timeline/seek")
def timeline_seek(
    request: Request,
    timestamp: int,
    favorited: str | None = None,
    dedupe: str = "1",
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse:
    """Return the 1-based page number for the given Unix timestamp.

    Assets are sorted by ``taken_at DESC`` (newest first), so this counts the
    number of assets *newer* than *timestamp* and divides by the page size.

    Query parameters:
    - ``timestamp``: Unix timestamp (seconds) to seek to.
    - ``favorited``: ``"1"`` to restrict to favorited assets.
    - ``dedupe``: ``"0"`` to include duplicate files; defaults to ``"1"``.

    Returns ``{"page": N}`` where *N* is the 1-based page that contains photos
    taken around the requested time.
    """
    fav_filter: bool | None = True if favorited == "1" else None
    dedup_enabled = dedupe != "0"

    count = count_assets_newer_than(
        conn,
        timestamp,
        favorited=fav_filter,
        deduped=dedup_enabled,
    )
    page = max(1, count // _PAGE_SIZE + 1)
    return JSONResponse({"page": page})
