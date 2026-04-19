"""Generic CLIP-based and pHash-based similarity search for photos.

Public functions:

``find_similar_by_asset``
    Given a single asset ID, searches all assets for photos that are
    semantically or perceptually similar using either CLIP embedding
    cosine/euclidean/angular distance, or pHash Hamming distance.

``find_similar_by_embedding``
    Like :func:`find_similar_by_asset` with ``method="clip"``, but accepts
    a raw embedding blob directly instead of loading one from the database.
    Used for *search by image* where the user uploads an arbitrary reference
    image that is not stored in the library.

``find_similar_by_phash_hex``
    Like :func:`find_similar_by_asset` with ``method="phash"``, but accepts
    a pHash hex string directly instead of loading one from the database.

Usage::

    from takeout_rater.similarity import (
        find_similar_by_asset,
        find_similar_by_embedding,
        find_similar_by_phash_hex,
    )

    similar = find_similar_by_asset(conn, asset_id=42, method="clip", metric="cosine", threshold=0.85)
    similar_phash = find_similar_by_asset(conn, asset_id=42, method="phash", threshold=20)
    # Search by image – reference data from an uploaded file:
    similar_rev = find_similar_by_embedding(conn, embedding_blob, metric="cosine")
    similar_rev_ph = find_similar_by_phash_hex(conn, "abcdef...", threshold=20)
"""

from __future__ import annotations

import math
import sqlite3
import struct

_CLIP_DIM = 768


def _cos_sim_passes(cos_sim: float, metric: str, threshold: float) -> bool:
    """Return True if *cos_sim* meets *threshold* under *metric*.

    Mirrors the ``_are_similar`` helper in :mod:`takeout_rater.clustering.clip_builder`.

    - ``cosine``:    cos_sim >= threshold  (higher = more similar)
    - ``euclidean``: L2 distance √(2 − 2·cos_sim) ≤ threshold  (lower = more similar)
    - ``combined``:  angular distance arccos(cos_sim) ≤ threshold  (lower = more similar)
    """
    if metric == "cosine":
        return cos_sim >= threshold
    if metric == "euclidean":
        return math.sqrt(max(0.0, 2.0 - 2.0 * cos_sim)) <= threshold
    # combined / angular
    return math.acos(max(-1.0, min(1.0, cos_sim))) <= threshold


def _cos_sim_to_score(cos_sim: float, metric: str) -> float:
    """Convert cosine similarity to a *score* value stored in results.

    For ``cosine`` the score IS the similarity (range 0–1, higher = better).
    For ``euclidean`` and ``combined`` we return the raw distance so the
    caller can interpret it correctly (lower = better).
    """
    if metric == "cosine":
        return round(cos_sim, 4)
    if metric == "euclidean":
        return round(math.sqrt(max(0.0, 2.0 - 2.0 * cos_sim)), 4)
    # combined / angular
    return round(math.acos(max(-1.0, min(1.0, cos_sim))), 4)


def find_similar_by_asset(
    conn: sqlite3.Connection,
    asset_id: int,
    *,
    method: str = "clip",
    metric: str = "cosine",
    threshold: float | None = None,
) -> list[dict]:
    """Find photos similar to a given asset.

    Supports two similarity methods:

    ``method="clip"``
        Uses cosine similarity in CLIP embedding space.  Three distance
        *metric* values are supported (mirroring the clustering settings):

        - ``"cosine"``   – *threshold* is a min cosine similarity (0–1, default 0.85).
        - ``"euclidean"``– *threshold* is a max L2 distance (0–2, default 0.45).
        - ``"combined"`` – *threshold* is a max angular distance in radians (0–π, default 0.46).

    ``method="phash"``
        Uses Hamming distance over the stored 256-bit dhash value.  *threshold*
        is the maximum number of differing bits (0–256, default 20).  *metric*
        is ignored in this mode.

    Args:
        conn: Open library database connection.
        asset_id: The reference asset to search from.
        method: ``"clip"`` (default) or ``"phash"``.
        metric: For CLIP mode: ``"cosine"`` (default), ``"euclidean"``, or
            ``"combined"``.  Ignored for pHash mode.
        threshold: Similarity / distance threshold.  Defaults depend on method
            and metric; see above.

    Returns:
        List of dicts with ``asset_id``, ``score``, ``taken_at``, and
        ``filename`` keys.  The reference asset is excluded.  Empty list when
        the reference has no embedding / pHash.

        For CLIP-cosine the ``score`` is the cosine similarity (higher = better).
        For CLIP-euclidean / CLIP-combined the ``score`` is the distance (lower = better).
        For pHash the ``score`` is the Hamming distance (lower = better).

        Results are sorted best-first (similarity descending for cosine,
        distance ascending for all other metrics).
    """
    if method == "phash":
        return _find_similar_phash(conn, asset_id, threshold=threshold)
    return _find_similar_clip(conn, asset_id, metric=metric, threshold=threshold)


# ---------------------------------------------------------------------------
# Implementation helpers
# ---------------------------------------------------------------------------

_CLIP_METRIC_DEFAULTS: dict[str, float] = {
    "cosine": 0.85,
    "euclidean": 0.45,
    "combined": 0.46,
}


def _find_similar_clip(
    conn: sqlite3.Connection,
    asset_id: int,
    *,
    metric: str = "cosine",
    threshold: float | None = None,
) -> list[dict]:
    """CLIP-embedding based similarity search."""
    import numpy as np

    if metric not in _CLIP_METRIC_DEFAULTS:
        metric = "cosine"
    if threshold is None:
        threshold = _CLIP_METRIC_DEFAULTS[metric]

    expected = _CLIP_DIM * 4

    # Load the reference embedding
    ref_row = conn.execute(
        "SELECT embedding FROM clip_embeddings WHERE asset_id = ?",
        (asset_id,),
    ).fetchone()

    if ref_row is None or len(ref_row[0]) != expected:
        return []

    ref_vec = np.array(struct.unpack(f"{_CLIP_DIM}f", ref_row[0]), dtype=np.float32)
    ref_norm = float(np.linalg.norm(ref_vec))
    if ref_norm < 1e-9:
        return []
    ref_vec = ref_vec / ref_norm

    # Load all embeddings joined with asset metadata
    all_rows = conn.execute(
        "SELECT ce.asset_id, ce.embedding, a.taken_at, a.filename"
        " FROM clip_embeddings ce"
        " JOIN assets a ON a.id = ce.asset_id"
        " ORDER BY ce.asset_id"
    ).fetchall()

    results: list[dict] = []
    for aid, blob, taken_at, filename in all_rows:
        if aid == asset_id:
            continue
        if len(blob) != expected:
            continue
        vec = np.array(struct.unpack(f"{_CLIP_DIM}f", blob), dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            continue
        vec = vec / norm
        cos_sim = float(np.dot(vec, ref_vec))
        if _cos_sim_passes(cos_sim, metric, threshold):
            results.append(
                {
                    "asset_id": aid,
                    "score": _cos_sim_to_score(cos_sim, metric),
                    "taken_at": taken_at,
                    "filename": filename,
                }
            )

    # Sort best-first: for cosine higher is better; for distance metrics lower is better
    reverse_sort = metric == "cosine"
    results.sort(key=lambda r: r["score"], reverse=reverse_sort)
    return results


def _find_similar_phash(
    conn: sqlite3.Connection,
    asset_id: int,
    *,
    threshold: float | None = None,
) -> list[dict]:
    """pHash Hamming-distance based similarity search."""
    if threshold is None:
        threshold = 20.0
    max_bits = int(threshold)

    # Load reference pHash
    ref_row = conn.execute(
        "SELECT phash_hex FROM phash WHERE asset_id = ?",
        (asset_id,),
    ).fetchone()

    if ref_row is None:
        return []

    ref_hex = ref_row[0]
    results = _phash_search_by_hex(conn, ref_hex, max_bits=max_bits)
    # Exclude the reference asset itself
    return [r for r in results if r["asset_id"] != asset_id]


def _phash_search_by_hex(
    conn: sqlite3.Connection,
    ref_hex: str,
    *,
    max_bits: int = 20,
) -> list[dict]:
    """Inner pHash search given a pre-computed hex string (includes all assets)."""
    try:
        ref_int = int(ref_hex, 16)
    except (ValueError, TypeError):
        return []

    # Load all pHashes joined with asset metadata
    all_rows = conn.execute(
        "SELECT p.asset_id, p.phash_hex, a.taken_at, a.filename"
        " FROM phash p"
        " JOIN assets a ON a.id = p.asset_id"
        " ORDER BY p.asset_id"
    ).fetchall()

    results: list[dict] = []
    for aid, phash_hex, taken_at, filename in all_rows:
        try:
            h_int = int(phash_hex, 16)
        except (ValueError, TypeError):
            continue
        dist = bin(ref_int ^ h_int).count("1")
        if dist <= max_bits:
            results.append(
                {
                    "asset_id": aid,
                    "score": dist,
                    "taken_at": taken_at,
                    "filename": filename,
                }
            )

    # Sort best-first: lower Hamming distance = more similar
    results.sort(key=lambda r: r["score"])
    return results


# ---------------------------------------------------------------------------
# Public helpers for "search by image" (reference data supplied directly)
# ---------------------------------------------------------------------------


def find_similar_by_embedding(
    conn: sqlite3.Connection,
    ref_blob: bytes,
    *,
    metric: str = "cosine",
    threshold: float | None = None,
) -> list[dict]:
    """Find photos similar to a given CLIP embedding blob.

    Like :func:`find_similar_by_asset` with ``method="clip"``, but accepts
    the reference embedding directly instead of loading it from the database.
    Used for *search by image* where the user supplies an arbitrary reference
    image that is **not** stored in the library.

    Args:
        conn: Open library database connection.
        ref_blob: Raw CLIP embedding as packed float32 bytes (``_CLIP_DIM × 4``
            bytes, i.e. 3072 bytes for the default 768-dimensional model).
        metric: ``"cosine"`` (default), ``"euclidean"``, or ``"combined"``.
        threshold: Similarity / distance threshold.  Defaults to the same values
            as :func:`find_similar_by_asset`.

    Returns:
        Same format as :func:`find_similar_by_asset`.  Empty list when
        *ref_blob* cannot be decoded or is all-zero.
    """
    import numpy as np

    if metric not in _CLIP_METRIC_DEFAULTS:
        metric = "cosine"
    if threshold is None:
        threshold = _CLIP_METRIC_DEFAULTS[metric]

    expected = _CLIP_DIM * 4
    if len(ref_blob) != expected:
        return []

    ref_vec = np.array(struct.unpack(f"{_CLIP_DIM}f", ref_blob), dtype=np.float32)
    ref_norm = float(np.linalg.norm(ref_vec))
    if ref_norm < 1e-9:
        return []
    ref_vec = ref_vec / ref_norm

    # Load all embeddings joined with asset metadata
    all_rows = conn.execute(
        "SELECT ce.asset_id, ce.embedding, a.taken_at, a.filename"
        " FROM clip_embeddings ce"
        " JOIN assets a ON a.id = ce.asset_id"
        " ORDER BY ce.asset_id"
    ).fetchall()

    results: list[dict] = []
    for aid, blob, taken_at, filename in all_rows:
        if len(blob) != expected:
            continue
        vec = np.array(struct.unpack(f"{_CLIP_DIM}f", blob), dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            continue
        vec = vec / norm
        cos_sim = float(np.dot(vec, ref_vec))
        if _cos_sim_passes(cos_sim, metric, threshold):
            results.append(
                {
                    "asset_id": aid,
                    "score": _cos_sim_to_score(cos_sim, metric),
                    "taken_at": taken_at,
                    "filename": filename,
                }
            )

    reverse_sort = metric == "cosine"
    results.sort(key=lambda r: r["score"], reverse=reverse_sort)
    return results


def find_similar_by_phash_hex(
    conn: sqlite3.Connection,
    ref_hex: str,
    *,
    threshold: float | None = None,
) -> list[dict]:
    """Find photos similar to a given pHash hex string.

    Like :func:`find_similar_by_asset` with ``method="phash"``, but accepts
    the reference pHash directly instead of loading it from the database.
    Used for *search by image* where the user supplies an arbitrary reference
    image that is **not** stored in the library.

    Args:
        conn: Open library database connection.
        ref_hex: 64-character hexadecimal dhash string.
        threshold: Maximum Hamming distance (default 20).

    Returns:
        Same format as :func:`find_similar_by_asset`.  Empty list when
        *ref_hex* cannot be parsed.
    """
    if threshold is None:
        threshold = 20.0
    max_bits = int(threshold)
    return _phash_search_by_hex(conn, ref_hex, max_bits=max_bits)
