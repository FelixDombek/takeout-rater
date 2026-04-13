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
    delete_clusters_by_method_params,
    insert_cluster,
    list_all_phashes,
)
from takeout_rater.scoring.phash import DHASH_ALGO

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
    single_linkage: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
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
        single_linkage: When ``True``, skip the complete-linkage post-processing
            step.  Two images can end up in the same cluster even if they are
            far apart, as long as there is a chain of pairwise-similar images
            connecting them (A≈B and B≈C merges A, B, C even if dist(A,C) >
            threshold).  This allows gradual progressions of similar shots to
            end up in one cluster.  Defaults to ``False`` (complete-linkage).
        on_progress: Optional callback called periodically with
            ``(processed_so_far, total)`` integers.

    Returns:
        Number of clusters persisted to the DB.
    """
    params: dict[str, int | bool] = {"threshold": threshold, "window": window}
    if single_linkage:
        params["single_linkage"] = True
    params_json = json.dumps(params, separators=(",", ":"), sort_keys=True)

    # Fetch only hashes computed with the current algorithm
    rows = list_all_phashes(conn, algo=DHASH_ALGO)
    if not rows:
        return 0

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

    # Collect components, then apply complete-linkage post-processing to each
    final_clusters: list[list[int]] = []
    for members in uf.components().values():
        if len(members) < 2:  # skip singletons early
            continue
        if single_linkage:
            # Use the raw single-linkage components without further splitting.
            if len(members) >= min_cluster_size:
                final_clusters.append(members)
        else:
            for sub in _split_by_complete_linkage(members, hash_map, threshold):
                if len(sub) >= min_cluster_size:
                    final_clusters.append(sub)

    if not final_clusters:
        return 0

    # Delete previous run with same method+params
    delete_clusters_by_method_params(conn, _METHOD, params_json)

    # Persist clusters
    n_persisted = 0
    for members in sorted(final_clusters, key=lambda m: min(m)):
        representative = min(members)
        diameter = _compute_diameter(members, hash_map)
        cluster_id = insert_cluster(conn, _METHOD, params_json, diameter=diameter)
        rows_to_insert: list[tuple[int, float | None, int]] = [
            (aid, None, 1 if aid == representative else 0) for aid in members
        ]
        bulk_insert_cluster_members(conn, cluster_id, rows_to_insert)
        n_persisted += 1

    return n_persisted
