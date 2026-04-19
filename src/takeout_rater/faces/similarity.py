"""Face-cluster-based similarity search for photos.

Public function:

``find_similar_photos``
    Given a face cluster (person group), computes a mean CLIP embedding
    from the assets where the person's face *is* detected and then
    searches the full ``clip_embeddings`` table for visually similar
    photos that may contain the same person with a hidden face.

For generic asset/image similarity (not face-specific), see
:mod:`takeout_rater.similarity`.

Usage::

    from takeout_rater.faces.similarity import find_similar_photos

    suggestions = find_similar_photos(conn, cluster_id=5, threshold=0.80, limit=50)
"""

from __future__ import annotations

import sqlite3
import struct

_CLIP_DIM = 768


def find_similar_photos(
    conn: sqlite3.Connection,
    cluster_id: int,
    *,
    threshold: float = 0.80,
    limit: int = 50,
) -> list[dict]:
    """Find photos visually similar to a face cluster's assets via CLIP.

    Computes the mean CLIP embedding of assets in the given face cluster,
    then ranks all other assets by cosine similarity.

    Args:
        conn: Open library database connection.
        cluster_id: The face cluster (person group) to search from.
        threshold: Minimum cosine similarity to include in results.
        limit: Maximum number of suggestions to return.

    Returns:
        List of dicts with ``asset_id`` and ``similarity`` keys, sorted
        by similarity descending.  Only assets *not* already in the
        cluster are returned.
    """
    import numpy as np

    # Get asset IDs in the face cluster
    cluster_asset_rows = conn.execute(
        "SELECT DISTINCT fe.asset_id"
        " FROM face_cluster_members fcm"
        " JOIN face_embeddings fe ON fe.id = fcm.face_id"
        " WHERE fcm.cluster_id = ?",
        (cluster_id,),
    ).fetchall()

    if not cluster_asset_rows:
        return []

    cluster_asset_ids = {row[0] for row in cluster_asset_rows}

    # Load CLIP embeddings for these assets
    placeholders = ",".join("?" for _ in cluster_asset_ids)
    ref_rows = conn.execute(
        f"SELECT asset_id, embedding FROM clip_embeddings"  # noqa: S608
        f" WHERE asset_id IN ({placeholders})",
        list(cluster_asset_ids),
    ).fetchall()

    if not ref_rows:
        return []

    expected = _CLIP_DIM * 4
    ref_vecs = []
    for _aid, blob in ref_rows:
        if len(blob) != expected:
            continue
        vec = np.array(struct.unpack(f"{_CLIP_DIM}f", blob), dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-9:
            ref_vecs.append(vec / norm)

    if not ref_vecs:
        return []

    # Compute mean reference vector
    ref_matrix = np.stack(ref_vecs)
    mean_vec = ref_matrix.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_vec))
    if mean_norm < 1e-9:
        return []
    mean_vec = mean_vec / mean_norm

    # Load all CLIP embeddings
    all_rows = conn.execute(
        "SELECT asset_id, embedding FROM clip_embeddings ORDER BY asset_id"
    ).fetchall()

    results: list[dict] = []
    for aid, blob in all_rows:
        if aid in cluster_asset_ids:
            continue
        if len(blob) != expected:
            continue
        vec = np.array(struct.unpack(f"{_CLIP_DIM}f", blob), dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            continue
        vec = vec / norm
        sim = float(np.dot(vec, mean_vec))
        if sim >= threshold:
            results.append({"asset_id": aid, "similarity": round(sim, 4)})

    results.sort(key=lambda r: r["similarity"], reverse=True)
    return results[:limit]
