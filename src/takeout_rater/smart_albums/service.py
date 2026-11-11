"""Run smart album suggestion against a library database and store the result."""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from takeout_rater.db import smart_album_queries as sq
from takeout_rater.smart_albums.suggest import DEFAULT_PARAMS, SuggestParams, suggest_albums

_log = logging.getLogger(__name__)

# Prompt template for the text side; "a photo of …" matches CLIP's training
# captions better than a bare word and gives cleaner names.
_PROMPT = "a photo of {}"
_COVER_POOL = 20  # cover = best aesthetic score among the N most typical photos
_KEPT_OVERLAP = 0.5  # Jaccard above which a suggestion duplicates a kept album

TextEncoder = Callable[[list[str]], np.ndarray]

_text_cache: dict[tuple[str, ...], np.ndarray] = {}
_text_cache_lock = threading.Lock()


class SuggestionCancelled(Exception):
    """Raised when a run is cancelled before anything was written."""


def encode_clip_texts(terms: list[str]) -> np.ndarray:
    """Encode *terms* with the CLIP ViT-L/14 text encoder (cached per term list).

    Returns:
        ``(len(terms), 768)`` float32 matrix of unit vectors.
    """
    key = tuple(terms)
    with _text_cache_lock:
        if key in _text_cache:
            return _text_cache[key]

    import torch

    from takeout_rater.scoring.scorers.clip_backbone import get_clip_model

    model, _preprocess, tokenizer, device = get_clip_model()
    vecs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(terms), 64):
            batch = [_PROMPT.format(t) for t in terms[start : start + 64]]
            feats = model.encode_text(tokenizer(batch).to(device))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            vecs.append(feats.cpu().float().numpy())
    matrix = np.vstack(vecs).astype(np.float32)
    with _text_cache_lock:
        _text_cache[key] = matrix
    return matrix


def vocabulary(conn: sqlite3.Connection) -> tuple[list[str], set[str]]:
    """Built-in clip_vocab terms plus the user's own CLIP tags (deduplicated).

    Returns:
        ``(terms, lead_terms)``: all terms, and those that may lead an album
        name (subjects from the vocabulary plus every user tag).
    """
    from takeout_rater.db.queries import list_clip_user_tags
    from takeout_rater.scoring.scorers.clip_vocab import (
        CLIP_VOCAB_SUBJECT_TERMS,
        CLIP_VOCAB_TERMS,
    )

    terms = list(CLIP_VOCAB_TERMS)
    lead = set(CLIP_VOCAB_SUBJECT_TERMS)
    seen = set(terms)
    for tag in list_clip_user_tags(conn):
        lead.add(tag)
        if tag not in seen:
            seen.add(tag)
            terms.append(tag)
    return terms, lead


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def run_suggestions(
    conn: sqlite3.Connection,
    params: SuggestParams = DEFAULT_PARAMS,
    *,
    encode_texts: TextEncoder | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Cluster the library's CLIP embeddings and replace the stored suggestions.

    Albums the user kept are left alone, and a new suggestion that mostly
    repeats a kept album is dropped.

    Args:
        encode_texts: Text encoder (defaults to :func:`encode_clip_texts`);
            injectable for tests.
        on_progress: Called with ``(message, step, n_steps)`` between stages.
        should_cancel: Polled between stages; when it returns ``True`` the run
            stops with :class:`SuggestionCancelled` before writing anything.

    Returns:
        A summary dict (counts and timing).
    """
    from takeout_rater.db.event_queries import load_aesthetic_scores

    n_steps = 4

    def _step(message: str, step: int) -> None:
        if should_cancel is not None and should_cancel():
            raise SuggestionCancelled()
        if on_progress is not None:
            on_progress(message, step, n_steps)

    t0 = time.monotonic()
    _step("Loading CLIP embeddings…", 0)
    asset_ids, mat = sq.load_embeddings(conn)
    if not asset_ids:
        raise ValueError(
            "No CLIP embeddings yet. Compute them on the Jobs page (Embed) first, "
            "then suggest albums."
        )

    _step("Encoding vocabulary…", 1)
    terms, lead_terms = vocabulary(conn)
    vocab = (encode_texts or encode_clip_texts)(terms)

    _step(f"Clustering {len(asset_ids):,} photos…", 2)
    suggestions = suggest_albums(asset_ids, mat, vocab, terms, params, lead_terms=lead_terms)

    _step("Saving albums…", 3)
    kept = sq.load_kept_members(conn)
    scores = load_aesthetic_scores(conn)
    albums: list[dict[str, Any]] = []
    n_duplicates = 0
    for s in suggestions:
        members = set(s.asset_ids)
        if any(_jaccard(members, k) >= _KEPT_OVERLAP for k in kept):
            n_duplicates += 1
            continue
        pool = s.asset_ids[:_COVER_POOL]
        cover = max(pool, key=lambda a: scores.get(a, float("-inf"))) if scores else pool[0]
        albums.append(
            {
                "name": s.name,
                "terms": s.terms,
                "asset_ids": s.asset_ids,
                "similarities": s.similarities,
                "cohesion": s.cohesion,
                "cover_asset_id": cover,
            }
        )
    if should_cancel is not None and should_cancel():
        raise SuggestionCancelled()
    sq.replace_suggestions(conn, albums)

    return {
        "n_photos": len(asset_ids),
        "n_albums": len(albums),
        "n_skipped_duplicates": n_duplicates,
        "n_terms": len(terms),
        "elapsed_s": round(time.monotonic() - t0, 1),
    }
