"""FastAPI router for CLIP semantic search.

Provides an endpoint to search images by natural-language description using
CLIP text–image cosine similarity.  The search index (a numpy matrix of all
stored CLIP embeddings) is built lazily on the first query and cached in
``app.state.clip_index``.  It is invalidated when the embed job completes.

Endpoints
---------
GET /api/search   – search by text query, returns ranked asset IDs + scores
"""

from __future__ import annotations

import sqlite3
import struct
from collections.abc import Generator
from dataclasses import dataclass

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from takeout_rater.db.queries import count_clip_embeddings, load_all_clip_embeddings

router = APIRouter()

_DEFAULT_LIMIT = 50


@dataclass
class _ClipIndex:
    """In-memory search index built from stored CLIP embeddings."""

    asset_ids: np.ndarray  # shape (N,), int64
    embeddings: np.ndarray  # shape (N, 768), float32


def _get_conn(request: Request) -> Generator[sqlite3.Connection, None, None]:
    """Dependency: open a per-request DB connection and close it when done."""
    db_path = request.app.state.db_path
    if db_path is not None:
        from takeout_rater.db.connection import open_db  # noqa: PLC0415

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


def _build_clip_index(conn: sqlite3.Connection) -> _ClipIndex | None:
    """Build an in-memory numpy matrix from all stored CLIP embeddings.

    Returns ``None`` if no embeddings are stored.
    """
    rows = load_all_clip_embeddings(conn)
    if not rows:
        return None

    n = len(rows)
    # Each embedding is 768 float32 values = 3072 bytes
    dim = 768
    asset_ids = np.empty(n, dtype=np.int64)
    embeddings = np.empty((n, dim), dtype=np.float32)

    for i, (aid, blob) in enumerate(rows):
        asset_ids[i] = aid
        embeddings[i] = np.array(struct.unpack(f"{dim}f", blob), dtype=np.float32)

    return _ClipIndex(asset_ids=asset_ids, embeddings=embeddings)


def _get_clip_index(request: Request, conn: sqlite3.Connection) -> _ClipIndex | None:
    """Return the cached clip index, rebuilding it if necessary."""
    index: _ClipIndex | None = getattr(request.app.state, "clip_index", None)
    if index is not None:
        return index

    index = _build_clip_index(conn)
    if index is not None:
        request.app.state.clip_index = index
    return index


@router.get("/api/search", response_model=None)
def search_assets(
    request: Request,
    q: str = "",
    limit: int = _DEFAULT_LIMIT,
    page: int = 1,
    partial: str = "0",
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse | HTMLResponse:
    """Search images by text description using CLIP cosine similarity.

    Query parameters:
    - ``q``: Text query describing the desired images.
    - ``limit``: Maximum number of results per page (default 50).
    - ``page``: 1-based page number for pagination.
    - ``partial``: ``"1"`` to return an HTML partial for HTMX.

    Returns a JSON response with ``results`` (list of ``{asset_id, score}``)
    and ``total`` (number of embedded assets), or an HTML partial when
    ``partial=1``.
    """
    if not q.strip():
        if partial == "1":
            templates = request.app.state.templates
            return templates.TemplateResponse(
                "search_results_partial.html",
                {"request": request, "results": [], "query": "", "total_embedded": 0},
            )
        return JSONResponse({"results": [], "total": 0, "query": ""})

    # Check if we have any embeddings
    embed_count = count_clip_embeddings(conn)
    if embed_count == 0:
        if partial == "1":
            templates = request.app.state.templates
            return templates.TemplateResponse(
                "search_results_partial.html",
                {
                    "request": request,
                    "results": [],
                    "query": q,
                    "total_embedded": 0,
                    "no_embeddings": True,
                },
            )
        return JSONResponse(
            {"results": [], "total": 0, "query": q, "error": "No embeddings computed yet."}
        )

    # Build/retrieve the search index
    index = _get_clip_index(request, conn)
    if index is None:
        return JSONResponse({"results": [], "total": 0, "query": q})

    # Encode the text query using CLIP
    import torch  # noqa: PLC0415

    from takeout_rater.scorers.adapters.clip_backbone import get_clip_model  # noqa: PLC0415

    _model, _preprocess, tokenizer, device = get_clip_model()

    tokens = tokenizer([q]).to(device)
    with torch.no_grad():
        text_features = _model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        query_vec = text_features.cpu().float().numpy().flatten()  # shape (768,)

    # Cosine similarity (embeddings are already L2-normalized)
    scores = index.embeddings @ query_vec  # shape (N,)

    # Rank by score descending
    ranked_indices = np.argsort(-scores)

    # Paginate
    limit = max(1, min(limit, 200))
    offset = max(0, (page - 1) * limit)
    page_indices = ranked_indices[offset : offset + limit]

    results = [
        {"asset_id": int(index.asset_ids[i]), "score": round(float(scores[i]), 4)}
        for i in page_indices
    ]

    if partial == "1":
        # Look up asset rows for the results to render thumbnails
        from takeout_rater.db.queries import get_asset_by_id  # noqa: PLC0415

        assets_with_scores = []
        for r in results:
            asset = get_asset_by_id(conn, r["asset_id"])
            if asset is not None:
                assets_with_scores.append({"asset": asset, "score": r["score"]})

        templates = request.app.state.templates
        return templates.TemplateResponse(
            "search_results_partial.html",
            {
                "request": request,
                "results": assets_with_scores,
                "query": q,
                "total_embedded": len(index.asset_ids),
                "page": page,
                "total_results": len(ranked_indices),
            },
        )

    return JSONResponse(
        {
            "results": results,
            "total": len(index.asset_ids),
            "query": q,
            "page": page,
        }
    )
