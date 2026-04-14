"""CLIP-embedding-based semantic cluster builder.

Algorithm
---------
1. Load all stored CLIP embeddings from the DB (normalised float-32 unit
   vectors, 768-dimensional ViT-L/14).
2. Build a NumPy matrix of shape ``(N, 768)``.
3. Compute pairwise cosine similarities in row batches using matrix
   multiplication (``batch @ all.T``).  Apply the chosen threshold to
   identify *neighbour* pairs.
4. Union-Find pass over all neighbour pairs → connected components.
5. Optional **complete-linkage post-processing**: every pair of assets within
   a sub-cluster must be within the threshold distance of each other.  Omit
   this step when *single_linkage=True*.
6. Persist clusters to the DB.

Three distance metrics are supported
--------------------------------------
``cosine``
    Two images are considered *similar* when
    ``cosine_similarity(a, b) >= threshold``.  For stored unit vectors this
    equals ``dot(a, b)``.  Default threshold: **0.90** (90 % cosine
    similarity).

``euclidean``
    Two images are considered *similar* when
    ``‖a − b‖₂ ≤ threshold``.  For unit vectors the range is [0, 2].
    Default threshold: **0.45**.

``combined``
    Two images are considered *similar* when the angular distance is ≤
    *threshold* radians.  This equals ``arccos(clamp(dot(a, b), −1, 1))``
    for unit vectors.  Default threshold: **0.46** rad (≈ 26°).

All three metrics are mathematically equivalent for unit-norm embeddings
(i.e. the same pair of images will be considered similar at the default
thresholds above), but they expose different scales to the user.

Usage::

    from takeout_rater.clustering.clip_builder import build_clip_clusters

    n = build_clip_clusters(conn, metric="cosine", threshold=0.90)
"""

from __future__ import annotations

import json
import math
import sqlite3
import struct
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from takeout_rater.clustering.builder import _UnionFind
from takeout_rater.db.queries import (
    bulk_insert_cluster_members,
    insert_cluster,
    insert_clustering_run,
    load_all_clip_embeddings,
)

_METHOD = "clip_embedding"
_DIM = 768
_BATCH = 512  # rows processed at once in the pairwise similarity pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_embeddings(
    rows: list[tuple[int, bytes]],
) -> tuple[list[int], object]:  # returns (valid_ids, ndarray)
    """Decode raw embedding blobs into a normalised float32 numpy matrix.

    Rows whose blob length does not match ``_DIM * 4`` bytes or whose L2
    norm is zero are silently skipped.

    Returns:
        ``(valid_asset_ids, matrix)`` where *matrix* has shape ``(n, _DIM)``.
    """
    import numpy as np  # noqa: PLC0415

    valid_ids: list[int] = []
    vecs: list[object] = []
    expected = _DIM * 4

    for asset_id, blob in rows:
        if len(blob) != expected:
            continue
        vec = np.array(struct.unpack(f"{_DIM}f", blob), dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            continue
        # Ensure unit length (embeddings are stored normalised, but recheck)
        vec = vec / norm
        vecs.append(vec)
        valid_ids.append(asset_id)

    if not vecs:
        return [], None

    return valid_ids, np.stack(vecs)


def _cos_sim_to_metric(cos_sim: float, metric: str) -> float:
    """Convert a cosine similarity value to the requested metric distance."""
    if metric == "cosine":
        return cos_sim  # similarity (higher = closer)
    if metric == "euclidean":
        return math.sqrt(max(0.0, 2.0 - 2.0 * cos_sim))
    # combined = angular distance in radians
    return math.acos(max(-1.0, min(1.0, cos_sim)))


def _are_similar(cos_sim: float, metric: str, threshold: float) -> bool:
    """Return True if two unit-norm embeddings with *cos_sim* meet *threshold*."""
    if metric == "cosine":
        return cos_sim >= threshold
    if metric == "euclidean":
        return math.sqrt(max(0.0, 2.0 - 2.0 * cos_sim)) <= threshold
    # combined (angular)
    return math.acos(max(-1.0, min(1.0, cos_sim))) <= threshold


def _split_by_complete_linkage_clip(
    component: list[int],
    emb_matrix: object,
    aid_to_idx: dict[int, int],
    metric: str,
    threshold: float,
) -> list[list[int]]:
    """Split a single-linkage component into complete-linkage sub-clusters.

    Every pair of members within a returned sub-cluster satisfies the
    distance threshold.  Uses the same greedy first-fit algorithm as the
    pHash builder.

    Args:
        component: Asset IDs in the component.
        emb_matrix: Full (N, 768) float32 embedding matrix.
        aid_to_idx: Mapping asset_id → row index in *emb_matrix*.
        metric: One of ``"cosine"``, ``"euclidean"``, ``"combined"``.
        threshold: The threshold appropriate for *metric*.

    Returns:
        List of sub-clusters; each is a list of asset IDs.
    """
    members = sorted(component)
    sub_clusters: list[list[int]] = []

    for member in members:
        placed = False
        ei = emb_matrix[aid_to_idx[member]]
        for sc in sub_clusters:
            # Vectorised check: compute cos-sim of member against all in sc
            sc_embs = emb_matrix[[aid_to_idx[m] for m in sc]]  # (k, 768)
            cos_sims = sc_embs @ ei  # (k,)
            if all(_are_similar(float(c), metric, threshold) for c in cos_sims):
                sc.append(member)
                placed = True
                break
        if not placed:
            sub_clusters.append([member])

    return sub_clusters


def _find_representative(
    members: list[int],
    emb_matrix: object,
    aid_to_idx: dict[int, int],
) -> int:
    """Return the most *central* asset in the cluster.

    The representative is the embedding closest to the normalised centroid of
    the cluster.  This gives a semantically more meaningful representative
    than the lowest-ID heuristic used for pHash clusters.

    Args:
        members: Asset IDs in the cluster.
        emb_matrix: Full (N, 768) float32 embedding matrix.
        aid_to_idx: Mapping asset_id → row index in *emb_matrix*.

    Returns:
        The asset_id of the most central member.
    """
    import numpy as np  # noqa: PLC0415

    if len(members) == 1:
        return members[0]

    idxs = [aid_to_idx[m] for m in members]
    embs = emb_matrix[idxs]  # (k, 768)
    centroid = embs.mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    if centroid_norm > 1e-9:
        centroid = centroid / centroid_norm
    sims = embs @ centroid  # (k,)
    best_local = int(np.argmax(sims))
    return members[best_local]


def _compute_diameter_clip(
    members: list[int],
    emb_matrix: object,
    aid_to_idx: dict[int, int],
    metric: str,
) -> float:
    """Return the intra-cluster diameter in the requested metric.

    The diameter is the maximum pairwise distance among all members.
    For ``cosine`` metric the diameter is the maximum *cosine distance*
    ``1 − min_cosine_similarity`` (so it stays non-negative and comparable
    to euclidean / combined values: 0 = all identical).

    Args:
        members: Asset IDs in the cluster.
        emb_matrix: Full (N, 768) float32 embedding matrix.
        aid_to_idx: Mapping asset_id → row index in *emb_matrix*.
        metric: One of ``"cosine"``, ``"euclidean"``, ``"combined"``.

    Returns:
        Maximum pairwise distance (float).
    """
    import numpy as np  # noqa: PLC0415

    if len(members) <= 1:
        return 0.0

    idxs = [aid_to_idx[m] for m in members]
    embs = emb_matrix[idxs]  # (k, 768)
    # Full pairwise cosine similarity matrix
    sims = embs @ embs.T  # (k, k)

    if metric == "cosine":
        # Diameter = max cosine distance = 1 - min cosine similarity
        # Exclude the diagonal (which is 1.0)
        np.fill_diagonal(sims, 1.0)
        return float(1.0 - np.min(sims))
    if metric == "euclidean":
        dists_sq = np.maximum(0.0, 2.0 - 2.0 * sims)
        np.fill_diagonal(dists_sq, 0.0)
        return float(np.sqrt(np.max(dists_sq)))
    # combined (angular)
    np.fill_diagonal(sims, 1.0)
    angles = np.arccos(np.clip(sims, -1.0, 1.0))
    np.fill_diagonal(angles, 0.0)
    return float(np.max(angles))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_clip_clusters(
    conn: sqlite3.Connection,
    *,
    metric: str = "cosine",
    threshold: float = 0.90,
    min_cluster_size: int = 2,
    single_linkage: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Build CLIP-embedding clusters and persist them to the DB.

    Each call creates a new clustering run.  Existing runs are left
    untouched (runs are independent and accumulate over time).

    Args:
        conn: Open library database connection.
        metric: Distance metric — one of:

            * ``"cosine"`` — threshold is the **minimum cosine similarity**
              in [0, 1].  Two images are similar when
              ``cosine_similarity >= threshold``.  Default: 0.90.
            * ``"euclidean"`` — threshold is the **maximum L2 distance**
              in [0, 2] (for unit vectors).  Default: 0.45.
            * ``"combined"`` — threshold is the **maximum angular distance**
              in radians in [0, π].  Default: 0.46 (≈ 26 °).

        threshold: Similarity / distance threshold for the chosen metric.
        min_cluster_size: Minimum number of members for a group to be
            stored as a cluster (default 2).
        single_linkage: When ``True``, skip complete-linkage
            post-processing.  Two images can be in the same cluster even if
            they are far apart, as long as a chain of step-by-step similar
            embeddings connects them.  When ``False`` (default), every pair
            in a cluster must satisfy the threshold — this is complete-linkage
            and prevents spurious "chaining" of dissimilar images.
        on_progress: Optional callback called periodically with
            ``(processed_so_far, total)`` integers.

    Returns:
        Number of clusters persisted to the DB, or 0 if no embeddings were
        found.

    Raises:
        ValueError: When *metric* is not one of the supported values.
        ImportError: When NumPy is not installed.
    """
    if metric not in ("cosine", "euclidean", "combined"):
        raise ValueError(
            f"Unsupported metric {metric!r}. Must be 'cosine', 'euclidean', or 'combined'."
        )

    import numpy as np  # noqa: PLC0415

    params: dict[str, float | str | bool] = {"metric": metric, "threshold": threshold}
    if single_linkage:
        params["single_linkage"] = True
    params_json = json.dumps(params, separators=(",", ":"), sort_keys=True)

    # Load all embeddings
    raw_rows = load_all_clip_embeddings(conn)
    if not raw_rows:
        return 0

    valid_ids, emb_matrix = _load_embeddings(raw_rows)
    if not valid_ids:
        return 0

    n = len(valid_ids)
    aid_to_idx: dict[int, int] = {aid: i for i, aid in enumerate(valid_ids)}

    uf = _UnionFind()
    for aid in valid_ids:
        uf.find(aid)

    # Pairwise similarity pass in row batches
    for i_start in range(0, n, _BATCH):
        i_end = min(i_start + _BATCH, n)
        batch = emb_matrix[i_start:i_end]  # (batch, 768)
        # sims[j, bi] = cosine_sim(emb_matrix[j], batch[bi])
        sims_block = emb_matrix @ batch.T  # (n, batch)

        for bi in range(i_end - i_start):
            global_i = i_start + bi
            aid_i = valid_ids[global_i]

            # Only examine j > global_i to avoid double-processing
            row = sims_block[global_i + 1 :, bi]  # (n - global_i - 1,)

            if metric == "cosine":
                near_mask = row >= threshold
            elif metric == "euclidean":
                near_mask = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * row)) <= threshold
            else:  # combined
                near_mask = np.arccos(np.clip(row, -1.0, 1.0)) <= threshold

            for offset in np.where(near_mask)[0]:
                j = global_i + 1 + int(offset)
                uf.union(aid_i, valid_ids[j])

        if on_progress:
            on_progress(i_end, n)

    # Collect components and apply linkage post-processing
    final_clusters: list[list[int]] = []
    for members in uf.components().values():
        if len(members) < 2:
            continue
        if single_linkage:
            if len(members) >= min_cluster_size:
                final_clusters.append(members)
        else:
            for sub in _split_by_complete_linkage_clip(
                members, emb_matrix, aid_to_idx, metric, threshold
            ):
                if len(sub) >= min_cluster_size:
                    final_clusters.append(sub)

    if not final_clusters:
        return 0

    run_id = insert_clustering_run(conn, _METHOD, params_json)

    n_persisted = 0
    for members in sorted(final_clusters, key=lambda m: min(m)):
        rep_id = _find_representative(members, emb_matrix, aid_to_idx)
        diameter = _compute_diameter_clip(members, emb_matrix, aid_to_idx, metric)

        cluster_id = insert_cluster(conn, _METHOD, params_json, diameter=diameter, run_id=run_id)

        rep_emb = emb_matrix[aid_to_idx[rep_id]]
        rows_to_insert: list[tuple[int, float | None, int]] = []
        for aid in members:
            cos_sim = float(np.dot(emb_matrix[aid_to_idx[aid]], rep_emb))
            dist = _cos_sim_to_metric(cos_sim, metric)
            # For cosine: dist is the cosine similarity itself (not a distance)
            # Store as-is; the UI will interpret based on the method/metric.
            rows_to_insert.append((aid, dist, 1 if aid == rep_id else 0))

        bulk_insert_cluster_members(conn, cluster_id, rows_to_insert)
        n_persisted += 1

    return n_persisted
