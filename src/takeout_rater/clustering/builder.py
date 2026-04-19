"""pHash-based near-duplicate cluster builder.

Algorithm
---------
1. Fetch all stored dhash values (algo = "dhash16") from the DB.
2. Sort ``(asset_id, hash_int)`` pairs by their integer hash value.
3. Slide a window of size *window* over the sorted list; for every pair
   within the window, compute Hamming distance.  If distance ≤ *threshold*,
   union the two assets.
4. After processing all pairs, collect the resulting connected components
   (Union-Find); each component becomes a candidate cluster.
5. Apply **complete-linkage post-processing** to each component: the component
   is split into sub-clusters such that every pair of members within a
   sub-cluster satisfies Hamming distance ≤ *threshold*.  This prevents
   single-linkage chaining (where A≈B and B≈C incorrectly merges A and C even
   when dist(A,C) > threshold).
6. Sub-clusters with ≥ *min_cluster_size* members are persisted.  The
   *representative* is the asset with the lowest ID; the *diameter* (maximum
   pairwise Hamming distance within the sub-cluster) is stored on the cluster
   row.

Usage::

    from takeout_rater.clustering.builder import build_clusters

    n_clusters = build_clusters(conn, threshold=20, window=200)
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable

from takeout_rater.db.queries import (
    bulk_insert_cluster_members,
    insert_cluster,
    insert_clustering_run,
    list_all_phashes,
    update_clustering_run_n_skipped,
)
from src.takeout_rater.clustering.phash import DHASH_ALGO

_METHOD = "dhash_hamming"


# ---------------------------------------------------------------------------
# Union-Find (path-compressed)
# ---------------------------------------------------------------------------


class _UnionFind:
    """Weighted union-find with path compression."""

    def __init__(self) -> None:
        self._parent: dict[int, int] = {}
        self._rank: dict[int, int] = {}

    def _make(self, x: int) -> None:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0

    def find(self, x: int) -> int:
        self._make(x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def components(self) -> dict[int, list[int]]:
        """Return mapping of root → list of members."""
        result: dict[int, list[int]] = {}
        for node in self._parent:
            root = self.find(node)
            result.setdefault(root, []).append(node)
        return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def hamming_distance_int(a: int, b: int) -> int:
    """Return the Hamming distance between two integers."""
    return bin(a ^ b).count("1")


def _split_by_complete_linkage(
    component: list[int],
    hash_map: dict[int, int],
    threshold: int,
) -> list[list[int]]:
    """Split a single-linkage component into complete-linkage sub-clusters.

    Within each returned sub-cluster every pair of members satisfies
    Hamming distance ≤ *threshold*.  Uses a greedy first-fit assignment
    (members sorted by integer hash value for determinism).

    Args:
        component: Asset IDs forming a single-linkage connected component.
        hash_map: Mapping of asset_id → integer hash value.
        threshold: Maximum allowed pairwise Hamming distance.

    Returns:
        List of sub-clusters; each sub-cluster is a list of asset IDs.
    """
    # Sort by hash value for determinism
    members = sorted(component, key=lambda aid: hash_map[aid])
    sub_clusters: list[list[int]] = []
    for member in members:
        placed = False
        h_member = hash_map[member]
        for sc in sub_clusters:
            if all(hamming_distance_int(h_member, hash_map[m]) <= threshold for m in sc):
                sc.append(member)
                placed = True
                break
        if not placed:
            sub_clusters.append([member])
    return sub_clusters


def _compute_diameter(members: list[int], hash_map: dict[int, int]) -> float:
    """Return the maximum pairwise Hamming distance within *members*."""
    max_dist = 0
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            d = hamming_distance_int(hash_map[members[i]], hash_map[members[j]])
            if d > max_dist:
                max_dist = d
    return float(max_dist)


def build_clusters(
    conn: sqlite3.Connection,
    *,
    threshold: int = 20,
    window: int = 200,
    min_cluster_size: int = 2,
    max_cluster_size: int | None = None,
    single_linkage: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
    on_post_progress: Callable[[int, int], None] | None = None,
    on_save_progress: Callable[[int, int], None] | None = None,
) -> tuple[int, int]:
    """Build pHash clusters and persist them to the DB.

    Existing clusters for the same *method* + *params* are deleted first so
    the operation is idempotent.

    Args:
        conn: Open library database connection.
        threshold: Maximum Hamming distance for two images to be considered
            near-duplicates (default 20, out of 256 bits for the 256-bit
            dhash16 algorithm).
        window: Sliding-window size over the sorted hash list (default 200).
            Larger values find more near-duplicates at higher CPU cost.
        min_cluster_size: Minimum number of members for a group to be
            stored as a cluster (default 2).  Singletons are ignored.
        max_cluster_size: Maximum number of members allowed in a cluster.
            Components larger than this are skipped entirely and counted in
            the returned *n_skipped* value.  ``None`` means no upper limit.
        single_linkage: When ``True``, skip the complete-linkage post-processing
            step.  Two images can end up in the same cluster even if they are
            far apart, as long as there is a chain of pairwise-similar images
            connecting them (A≈B and B≈C merges A, B, C even if dist(A,C) >
            threshold).  This allows gradual progressions of similar shots to
            end up in one cluster.  Defaults to ``False`` (complete-linkage).
        on_progress: Optional callback called periodically with
            ``(processed_so_far, total)`` integers during the sliding-window
            hash-comparison pass.
        on_post_progress: Optional callback called with
            ``(processed_components, total_components)`` after each
            multi-member connected component has been post-processed
            (complete-linkage splitting).
        on_save_progress: Optional callback called with
            ``(saved_so_far, total_to_save)`` after each cluster is written
            to the database.

    Returns:
        ``(n_persisted, n_skipped)`` — the number of clusters stored in the
        DB and the number of components that were skipped (due to
        *max_cluster_size* or a :class:`MemoryError` during processing).
    """
    params: dict[str, int | bool] = {"threshold": threshold, "window": window}
    if single_linkage:
        params["single_linkage"] = True
    params_json = json.dumps(params, separators=(",", ":"), sort_keys=True)

    # Fetch only hashes computed with the current algorithm
    rows = list_all_phashes(conn, algo=DHASH_ALGO)
    if not rows:
        return 0, 0

    # Convert to (asset_id, hash_int) and sort by hash value
    pairs = sorted([(aid, int(h, 16)) for aid, h in rows], key=lambda x: x[1])
    n = len(pairs)

    # Build a hash map for O(1) lookup during complete-linkage post-processing
    hash_map: dict[int, int] = {aid: hash_int for aid, hash_int in pairs}

    uf = _UnionFind()
    # Initialise all nodes so singletons appear in components()
    for aid, _ in pairs:
        uf.find(aid)

    # Sliding-window single-linkage pass
    for i in range(n):
        if on_progress and i % 1000 == 0:
            on_progress(i, n)
        aid_i, hash_i = pairs[i]
        for j in range(i + 1, min(i + window + 1, n)):
            aid_j, hash_j = pairs[j]
            dist = hamming_distance_int(hash_i, hash_j)
            if dist <= threshold:
                uf.union(aid_i, aid_j)

    if on_progress:
        on_progress(n, n)

    # Collect multi-member components; singletons are never clusters.
    multi_member_components = [members for members in uf.components().values() if len(members) >= 2]
    total_components = len(multi_member_components)

    # Apply complete-linkage post-processing to each component.
    final_clusters: list[list[int]] = []
    n_skipped = 0
    for comp_idx, members in enumerate(multi_member_components):
        if max_cluster_size is not None and len(members) > max_cluster_size:
            n_skipped += 1
            if on_post_progress:
                on_post_progress(comp_idx + 1, total_components)
            continue
        try:
            if single_linkage:
                # Use the raw single-linkage components without further splitting.
                if len(members) >= min_cluster_size:
                    final_clusters.append(members)
            else:
                for sub in _split_by_complete_linkage(members, hash_map, threshold):
                    if len(sub) >= min_cluster_size:
                        final_clusters.append(sub)
        except MemoryError:
            n_skipped += 1
        if on_post_progress:
            on_post_progress(comp_idx + 1, total_components)

    if not final_clusters and n_skipped == 0:
        return 0, 0

    # Create a new clustering run to group all clusters created in this call.
    run_id = insert_clustering_run(conn, _METHOD, params_json, n_skipped=n_skipped)

    # Persist clusters
    total_to_save = len(final_clusters)
    n_persisted = 0
    n_skipped_save = 0
    for members in sorted(final_clusters, key=lambda m: min(m)):
        try:
            representative = min(members)
            diameter = _compute_diameter(members, hash_map)
        except MemoryError:
            n_skipped_save += 1
            continue
        cluster_id = insert_cluster(conn, _METHOD, params_json, diameter=diameter, run_id=run_id)
        rows_to_insert: list[tuple[int, float | None, int]] = [
            (aid, None, 1 if aid == representative else 0) for aid in members
        ]
        bulk_insert_cluster_members(conn, cluster_id, rows_to_insert)
        n_persisted += 1
        if on_save_progress:
            on_save_progress(n_persisted, total_to_save)

    if n_skipped_save > 0:
        update_clustering_run_n_skipped(conn, run_id, n_skipped + n_skipped_save)

    return n_persisted, n_skipped + n_skipped_save
