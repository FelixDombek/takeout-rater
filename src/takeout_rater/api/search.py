"""FastAPI router for semantic (text-to-image) search.

Ranks images by cosine similarity between a text query and stored image
embeddings.  Several backbones are supported (see
:mod:`takeout_rater.embeddings.search_backbones`): SigLIP 2 when its
embeddings have been computed, and CLIP ViT-L/14 otherwise.

Each backbone's embeddings are loaded once into a numpy matrix (via
``np.frombuffer``, not per-row unpacking) and cached in
``app.state.search_indexes``, keyed by model.  The cache is dropped by
:func:`invalidate_search_indexes` when embeddings change.

Text-in-photo search (``mode=text``) uses the SQLite FTS5 index over text
recognised by the OCR job instead.

Endpoints
---------
GET /api/search            – search by text query, returns ranked asset IDs + scores
GET /api/search/backbones  – available backbones and how many images each has embedded
GET /api/ocr/status        – OCR coverage counts for the search page
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Generator
from dataclasses import dataclass

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from markupsafe import Markup, escape

from takeout_rater.db.queries import (
    CLIP_L14_MODEL_KEY,
    OCR_SNIPPET_END,
    OCR_SNIPPET_START,
    count_image_embeddings,
    get_ocr_stats,
    load_embedding_matrix,
    search_ocr_text,
)
from takeout_rater.embeddings import search_backbones

router = APIRouter()

_DEFAULT_LIMIT = 50
_index_lock = threading.Lock()  # guards app.state.search_indexes creation and _model_locks
_model_locks: dict[str, threading.Lock] = {}


@dataclass
class _SearchIndex:
    """In-memory search index for one backbone."""

    model_key: str
    asset_ids: np.ndarray  # shape (N,), int64
    embeddings: np.ndarray  # shape (N, dim), float32, L2-normalised rows


def _get_conn(request: Request) -> Generator[sqlite3.Connection, None, None]:
    """Dependency: open a per-request DB connection and close it when done."""
    db_path = request.app.state.db_path
    if db_path is not None:
        from takeout_rater.db.connection import open_db

        conn = open_db(db_path)
        try:
            yield conn
        finally:
            conn.close()
        return

    conn = request.app.state.db_conn
    if conn is None:
        raise HTTPException(status_code=503, detail="Library not configured — visit /setup")
    yield conn


def _index_lock_for(model_key: str) -> threading.Lock:
    with _index_lock:
        lock = _model_locks.get(model_key)
        if lock is None:
            lock = _model_locks[model_key] = threading.Lock()
        return lock


def _indexes(app_state: object) -> dict[str, _SearchIndex]:
    """Return ``app_state.search_indexes``, creating it once under the lock."""
    indexes: dict[str, _SearchIndex] | None = getattr(app_state, "search_indexes", None)
    if indexes is None:
        with _index_lock:
            indexes = getattr(app_state, "search_indexes", None)
            if indexes is None:
                indexes = {}
                app_state.search_indexes = indexes  # type: ignore[attr-defined]
    return indexes


def invalidate_search_indexes(app_state: object, model_key: str | None = None) -> None:
    """Drop cached search matrices (all of them, or only *model_key*'s)."""
    indexes = _indexes(app_state)
    keys = list(indexes) if model_key is None else [model_key]
    for key in keys:
        with _index_lock_for(key):
            indexes.pop(key, None)


def _build_search_index(conn: sqlite3.Connection, model_key: str) -> _SearchIndex | None:
    """Load all embeddings for *model_key*; ``None`` if none are stored."""
    asset_ids, matrix = load_embedding_matrix(conn, model_key)
    if len(asset_ids) == 0:
        return None
    return _SearchIndex(model_key=model_key, asset_ids=asset_ids, embeddings=matrix)


def _get_search_index(
    request: Request, conn: sqlite3.Connection, model_key: str
) -> _SearchIndex | None:
    """Return the cached index for *model_key*, building it if necessary.

    For backbones stored in ``image_embeddings`` the cached matrix is checked
    against the (indexed, cheap) row count, so an index cached while an embed
    job was still writing, or after it was cancelled, is rebuilt instead of
    serving a partial result forever.  CLIP L/14 relies on explicit
    invalidation because counting ``clip_embeddings`` scans the table.
    """
    indexes = _indexes(request.app.state)
    expected = None
    if model_key != CLIP_L14_MODEL_KEY:
        expected = count_image_embeddings(conn, model_key)
    index = indexes.get(model_key)
    if index is not None and (expected is None or len(index.asset_ids) == expected):
        return index
    with _index_lock_for(model_key):
        index = indexes.get(model_key)
        if index is None or (expected is not None and len(index.asset_ids) != expected):
            index = _build_search_index(conn, model_key)
            if index is not None:
                indexes[model_key] = index
            else:
                indexes.pop(model_key, None)
    return index


def _availability() -> dict[str, bool]:
    return {key: b.available() for key, b in search_backbones.BACKBONES.items()}


def _resolve_model(request: Request, conn: sqlite3.Connection, model: str) -> str:
    """Resolve the ``model`` query parameter to a concrete backbone key."""
    if model in search_backbones.BACKBONES:
        return model
    counts: dict[str, int] = {}
    available: dict[str, bool] = {CLIP_L14_MODEL_KEY: True}
    for key in search_backbones.AUTO_ORDER:
        if key == CLIP_L14_MODEL_KEY:
            continue
        counts[key] = count_image_embeddings(conn, key)
        # Only import-check backbones that could actually be chosen.
        available[key] = counts[key] > 0 and search_backbones.get_backbone(key).available()
    if any(n > 0 and available[k] for k, n in counts.items()):
        # Only needed to judge coverage; reuse the cached CLIP matrix when present.
        cached = _indexes(request.app.state).get(CLIP_L14_MODEL_KEY)
        counts[CLIP_L14_MODEL_KEY] = (
            len(cached.asset_ids)
            if cached is not None
            else count_image_embeddings(conn, CLIP_L14_MODEL_KEY)
        )
    return search_backbones.resolve_backbone_key("auto", counts, available)


@router.get("/api/search/backbones")
def list_search_backbones(
    request: Request,  # noqa: ARG001
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse:
    """List search backbones with how many images each has embedded.

    Returns ``{"backbones": [...], "total_assets": N, "auto": key}`` where
    each entry has ``key``, ``label``, ``description``, ``dim``, ``count``
    and ``available`` (whether its Python dependencies are installed).
    """
    total_assets = conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
    entries = []
    counts: dict[str, int] = {}
    available = _availability()
    for key in search_backbones.AUTO_ORDER:
        b = search_backbones.get_backbone(key)
        counts[key] = count_image_embeddings(conn, key)
        entries.append(
            {
                "key": b.key,
                "label": b.label,
                "description": b.description,
                "dim": b.dim,
                "count": counts[key],
                "available": available[key],
            }
        )
    return JSONResponse(
        {
            "backbones": entries,
            "total_assets": total_assets,
            "auto": search_backbones.resolve_backbone_key("auto", counts, available),
            "auto_min_coverage": search_backbones.AUTO_MIN_COVERAGE,
        }
    )


@router.get("/api/search", response_model=None)
def search_assets(
    request: Request,
    q: str = "",
    limit: int = _DEFAULT_LIMIT,
    page: int = 1,
    partial: str = "0",
    model: str = "auto",
    mode: str = "visual",
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse | HTMLResponse:
    """Search images by text description using embedding cosine similarity.

    Query parameters:
    - ``q``: Text query describing the desired images.
    - ``mode``: ``"visual"`` (default, embedding search; ``"clip"`` is an
      accepted alias) or ``"text"`` (full-text search over words recognised
      in the photos by OCR; ``model`` is ignored).
    - ``limit``: Maximum number of results per page (default 50).
    - ``page``: 1-based page number for pagination.
    - ``partial``: ``"1"`` to return an HTML partial of result cards.
    - ``model``: backbone key (see ``/api/search/backbones``) or ``"auto"``
      (default: SigLIP 2 when it has embeddings, else CLIP ViT-L/14).

    Returns a JSON response with ``results`` (list of ``{asset_id, score}``),
    ``total`` (number of embedded assets) and ``model`` (the backbone used),
    or an HTML partial when ``partial=1``.
    """
    if mode not in ("visual", "clip", "text"):
        raise HTTPException(status_code=400, detail="mode must be 'visual' or 'text'.")
    if mode == "text":
        return _search_ocr(request, conn, q, limit=limit, page=page, partial=partial == "1")

    templates = request.app.state.templates
    if not q.strip():
        if partial == "1":
            return templates.TemplateResponse(
                "search_results_partial.html",
                {"request": request, "results": [], "query": "", "total_embedded": 0},
            )
        return JSONResponse({"results": [], "total": 0, "query": ""})

    model_key = _resolve_model(request, conn, model)
    if model_key != CLIP_L14_MODEL_KEY and not search_backbones.get_backbone(model_key).available():
        raise HTTPException(
            status_code=503,
            detail=f"{model_key} is not installed (SigLIP 2 needs open-clip-torch>=2.31).",
        )
    index = _get_search_index(request, conn, model_key)
    if index is None:
        if partial == "1":
            return templates.TemplateResponse(
                "search_results_partial.html",
                {
                    "request": request,
                    "results": [],
                    "query": q,
                    "total_embedded": 0,
                    "no_embeddings": True,
                    "model": model_key,
                },
            )
        return JSONResponse(
            {
                "results": [],
                "total": 0,
                "query": q,
                "model": model_key,
                "error": "No embeddings computed yet.",
            }
        )

    query_vec = search_backbones.encode_text(model_key, q)
    if query_vec.shape[0] != index.embeddings.shape[1]:
        raise HTTPException(
            status_code=500,
            detail=f"Embedding dimension mismatch for {model_key}; recompute embeddings.",
        )

    # Cosine similarity (rows and query are L2-normalised)
    scores = index.embeddings @ query_vec.astype(np.float32)  # shape (N,)

    limit = max(1, min(limit, 200))
    offset = max(0, (page - 1) * limit)
    n = scores.shape[0]
    end = min(n, offset + limit)
    if offset >= n:
        page_indices = np.empty(0, dtype=np.int64)
    else:
        # Partial sort: only the top `end` scores need ordering.
        top = np.argpartition(-scores, end - 1)[:end] if end < n else np.arange(n)
        top = top[np.argsort(-scores[top], kind="stable")]
        page_indices = top[offset:end]

    results = [
        {"asset_id": int(index.asset_ids[i]), "score": round(float(scores[i]), 4)}
        for i in page_indices
    ]

    if partial == "1":
        from takeout_rater.db.queries import get_asset_by_id

        assets_with_scores = []
        for r in results:
            asset = get_asset_by_id(conn, r["asset_id"])
            if asset is not None:
                assets_with_scores.append({"asset": asset, "score": r["score"]})

        return templates.TemplateResponse(
            "search_results_partial.html",
            {
                "request": request,
                "results": assets_with_scores,
                "query": q,
                "total_embedded": n,
                "page": page,
                "total_results": n,
                "model": model_key,
            },
        )

    return JSONResponse(
        {
            "results": results,
            "total": n,
            "query": q,
            "page": page,
            "model": model_key,
        }
    )


def _snippet_html(snippet: str) -> Markup:
    """HTML-escape an FTS5 snippet and turn its match sentinels into ``<mark>``."""
    escaped = str(escape(snippet.replace("\n", " · ")))
    return Markup(escaped.replace(OCR_SNIPPET_START, "<mark>").replace(OCR_SNIPPET_END, "</mark>"))


def _search_ocr(
    request: Request,
    conn: sqlite3.Connection,
    q: str,
    *,
    limit: int,
    page: int,
    partial: bool,
) -> JSONResponse | HTMLResponse:
    """Full-text search over OCR text (``mode=text``)."""
    stats = get_ocr_stats(conn)
    limit = max(1, min(limit, 200))
    page = max(1, page)
    results, total = ([], 0)
    if q.strip() and stats["processed"] > 0:
        results, total = search_ocr_text(conn, q, limit=limit, offset=(page - 1) * limit)

    if partial:
        from takeout_rater.db.queries import get_asset_by_id

        items = []
        for r in results:
            asset = get_asset_by_id(conn, r["asset_id"])
            if asset is not None:
                items.append({"asset": asset, "snippet_html": _snippet_html(r["snippet"])})
        templates = request.app.state.templates
        return templates.TemplateResponse(
            "search_results_partial.html",
            {
                "request": request,
                "mode": "text",
                "results": items,
                "query": q,
                "total_embedded": stats["processed"],
                "total_results": total,
                "no_ocr": stats["processed"] == 0,
                "page": page,
            },
        )

    return JSONResponse(
        {
            "mode": "text",
            "results": [
                {
                    "asset_id": r["asset_id"],
                    "rank": round(r["rank"], 4),
                    "snippet": r["snippet"]
                    .replace(OCR_SNIPPET_START, "")
                    .replace(OCR_SNIPPET_END, ""),
                }
                for r in results
            ],
            "total": total,
            "processed": stats["processed"],
            "query": q,
            "page": page,
        }
    )


@router.get("/api/ocr/status")
def ocr_status(
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse:
    """Return OCR coverage: ``total_assets``, ``processed``, ``with_text``,
    plus whether RapidOCR is installed and CUDA is available."""
    from takeout_rater.ocr.engine import cuda_available, rapidocr_available

    stats = get_ocr_stats(conn)
    stats_out: dict[str, object] = dict(stats)
    stats_out["rapidocr_installed"] = rapidocr_available()
    stats_out["cuda_available"] = cuda_available()
    return JSONResponse(stats_out)
