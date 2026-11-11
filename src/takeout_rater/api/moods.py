"""FastAPI router for the mood timeline (issue #110).

Charts how strongly the photos of each period match a set of mood words
("happy", "peaceful" …), using text-to-image similarity on the stored
search embeddings (SigLIP 2 when available, CLIP ViT-L/14 otherwise).
Nothing is written to the database; results are computed on the fly from the
in-memory search matrix and cached per mood set.

Endpoints
---------
GET /moods                 – the mood timeline page
GET /api/moods/timeline    – per-period mood shares and lifts
GET /api/moods/photos      – the photos that lean most towards one mood
"""

from __future__ import annotations

import sqlite3
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from takeout_rater.api.search import _get_conn, _get_search_index, _resolve_model
from takeout_rater.db.queries import (
    CLIP_L14_MODEL_KEY,
    dated_assets_fingerprint,
    load_dated_asset_ids,
)
from takeout_rater.embeddings import search_backbones
from takeout_rater.moods import timeline as mt

router = APIRouter()

_CACHE_SIZE = 4
_cache_lock = threading.Lock()
_text_cache: dict[tuple[str, str], np.ndarray] = {}
_text_lock = threading.Lock()


@dataclass
class _MoodData:
    """Per-photo mood distribution for one (backbone, mood set) pair."""

    asset_ids: np.ndarray  # (n,) int64, ordered by id
    taken_at: np.ndarray  # (n,) int64
    probs: np.ndarray  # (n, k) float32
    total_embedded: int
    index: object  # the search index this was computed from (identity check)
    fingerprint: tuple  # dated_assets_fingerprint() at computation time


def _parse_moods(raw: str) -> list[str]:
    moods = mt.normalize_moods(raw.split(",")) if raw.strip() else list(mt.DEFAULT_MOODS)
    if len(moods) < 2:
        raise HTTPException(status_code=400, detail="Give at least two moods to compare.")
    return moods


def _encode_cached(model_key: str, text: str) -> np.ndarray:
    key = (model_key, text)
    vec = _text_cache.get(key)
    if vec is None:
        with _text_lock:
            vec = _text_cache.get(key)
            if vec is None:
                vec = search_backbones.encode_text(model_key, text)
                _text_cache[key] = vec
    return vec


def _mood_data(
    request: Request, conn: sqlite3.Connection, moods: list[str], model: str
) -> tuple[str, _MoodData | None]:
    """Return the resolved backbone key and the mood distribution of every dated photo."""
    model_key = _resolve_model(request, conn, model)
    if model_key != CLIP_L14_MODEL_KEY and not search_backbones.get_backbone(model_key).available():
        raise HTTPException(status_code=503, detail=f"{model_key} is not installed.")
    index = _get_search_index(request, conn, model_key)
    if index is None:
        return model_key, None

    state = request.app.state
    cache: OrderedDict | None = getattr(state, "mood_cache", None)
    if cache is None:
        with _cache_lock:
            cache = getattr(state, "mood_cache", None)
            if cache is None:
                cache = state.mood_cache = OrderedDict()
    cache_key = (model_key, tuple(moods))
    # Dates and trash flags can change (rescan, trash) without the search index
    # being rebuilt, e.g. for SigLIP 2, so check them separately.
    fingerprint = dated_assets_fingerprint(conn)
    with _cache_lock:
        data = cache.get(cache_key)
        # A rebuilt search index is a new object, so stale entries never match.
        if data is not None and data.index is index and data.fingerprint == fingerprint:
            cache.move_to_end(cache_key)
            return model_key, data

    text = mt.mood_text_matrix(moods, lambda p: _encode_cached(model_key, p))
    if text.shape[1] != index.embeddings.shape[1]:
        raise HTTPException(
            status_code=500,
            detail=f"Embedding dimension mismatch for {model_key}; recompute embeddings.",
        )
    dated_ids, dated_ts = load_dated_asset_ids(conn)
    _, emb_pos, dated_pos = np.intersect1d(
        index.asset_ids, dated_ids, assume_unique=True, return_indices=True
    )
    probs = mt.mood_probabilities(index.embeddings[emb_pos], text)
    data = _MoodData(
        asset_ids=index.asset_ids[emb_pos],
        taken_at=dated_ts[dated_pos],
        probs=probs,
        total_embedded=len(index.asset_ids),
        index=index,
        fingerprint=fingerprint,
    )
    with _cache_lock:
        cache[cache_key] = data
        while len(cache) > _CACHE_SIZE:
            cache.popitem(last=False)
    return model_key, data


@router.get("/moods", response_class=HTMLResponse)
def moods_page(request: Request) -> HTMLResponse:
    """Render the mood timeline page."""
    if request.app.state.db_conn is None:
        from fastapi.responses import RedirectResponse

        return RedirectResponse(url="/setup")  # type: ignore[return-value]
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "moods.html",
        {
            "request": request,
            "default_moods": list(mt.DEFAULT_MOODS),
            "max_moods": mt.MAX_MOODS,
            "prompt_templates": list(mt.PROMPT_TEMPLATES),
        },
    )


@router.get("/api/moods/timeline")
def mood_timeline(
    request: Request,
    moods: str = "",
    granularity: str = "month",
    model: str = "auto",
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse:
    """Return how much each mood dominates each time period.

    Query parameters:
    - ``moods``: comma-separated mood words (default: :data:`DEFAULT_MOODS`).
    - ``granularity``: ``week``, ``month`` (default), ``quarter`` or ``year``.
    - ``model``: search backbone key or ``auto``.

    Every photo's similarity to the moods is turned into a probability
    distribution (zero-shot classification); ``share`` is the average of
    those per period and ``lift`` is share divided by the library-wide share.
    """
    if granularity not in mt.GRANULARITIES:
        raise HTTPException(status_code=400, detail="Unknown granularity.")
    mood_list = _parse_moods(moods)
    model_key, data = _mood_data(request, conn, mood_list, model)
    if data is None:
        return JSONResponse(
            {
                "moods": mood_list,
                "granularity": granularity,
                "model": model_key,
                "buckets": [],
                "overall": {"count": 0, "share": [0.0] * len(mood_list)},
                "total_embedded": 0,
                "error": "No embeddings computed yet.",
            }
        )
    result = mt.mood_timeline(data.probs, data.taken_at, mood_list, granularity)
    result["model"] = model_key
    result["total_embedded"] = data.total_embedded
    return JSONResponse(result)


@router.get("/api/moods/photos")
def mood_photos(
    request: Request,
    mood: str,
    moods: str = "",
    start: int | None = None,
    end: int | None = None,
    limit: int = Query(12, ge=1, le=100),
    model: str = "auto",
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse:
    """Return the photos leaning most towards *mood* (within ``[start, end)`` if given).

    ``moods`` must be the same set the timeline was computed with, because a
    photo's mood probability depends on the moods it is compared against.
    """
    mood_list = _parse_moods(moods)
    folded = [m.casefold() for m in mood_list]
    wanted = " ".join(mood.split()).casefold()
    if wanted not in folded:
        raise HTTPException(status_code=400, detail=f"'{mood}' is not one of the moods.")
    model_key, data = _mood_data(request, conn, mood_list, model)
    photos = (
        []
        if data is None
        else mt.top_photos(
            data.probs,
            data.asset_ids,
            data.taken_at,
            folded.index(wanted),
            start=start,
            end=end,
            limit=limit,
        )
    )
    return JSONResponse(
        {"mood": mood_list[folded.index(wanted)], "model": model_key, "photos": photos}
    )
