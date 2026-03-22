"""pHash-based near-duplicate cluster builder.

Algorithm
---------
1. Fetch all stored dhash values from the DB.
2. Sort ``(asset_id, hash_int)`` pairs by their integer hash value.
3. Slide a window of size *window* over the sorted list; for every pair
   within the window, compute Hamming distance.  If distance ≤ *threshold*,
   union the two assets.
4. After processing all pairs, collect the resulting connected components
   (Union-Find); each component with ≥ *min_cluster_size* members becomes a
   cluster.
5. Within each cluster, the *representative* is the asset with the lowest ID.
6. The results are written to the ``clusters`` and ``cluster_members`` DB
   tables (after first deleting any existing run for the same method+params).

The window-over-sorted-hashes strategy is an efficient O(n × window)
approximation.  Near-duplicate photos almost always have similar dhash
integers (they share most bits), so they naturally end up adjacent in the
sorted order.  The algorithm is exact for duplicates differing only in
low-order bits and approximate (may miss some pairs) for duplicates that
differ in high-order bits.  For a library of 236k photos a window of 200
and threshold of 10 bits takes < 1 second.

Usage::

    from takeout_rater.clustering.builder import build_clusters

    n_clusters = build_clusters(conn, threshold=10, window=200)
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


def build_clusters(
    conn: sqlite3.Connection,
    *,
    threshold: int = 10,
    window: int = 200,
    min_cluster_size: int = 2,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Build pHash clusters and persist them to the DB.

    Existing clusters for the same *method* + *params* are deleted first so
    the operation is idempotent.

    Args:
        conn: Open library database connection.
        threshold: Maximum Hamming distance for two images to be considered
            near-duplicates (default 10, out of 64 bits).
        window: Sliding-window size over the sorted hash list (default 200).
            Larger values find more near-duplicates at higher CPU cost.
        min_cluster_size: Minimum number of members for a group to be
            stored as a cluster (default 2).  Singletons are ignored.
        on_progress: Optional callback called periodically with
            ``(processed_so_far, total)`` integers.

    Returns:
        Number of clusters persisted to the DB.
    """
    params = {"threshold": threshold, "window": window}
    params_json = json.dumps(params, separators=(",", ":"), sort_keys=True)

    # Fetch all stored hashes
    rows = list_all_phashes(conn)
    if not rows:
        return 0

    # Convert to (asset_id, hash_int) and sort by hash value
    pairs = sorted([(aid, int(h, 16)) for aid, h in rows], key=lambda x: x[1])
    n = len(pairs)

    uf = _UnionFind()
    # Initialise all nodes so singletons appear in components()
    for aid, _ in pairs:
        uf.find(aid)

    # Sliding-window comparison
    for i in range(n):
        if on_progress and i % 1000 == 0:
            on_progress(i, n)
        aid_i, hash_i = pairs[i]
        for j in range(i + 1, min(i + window + 1, n)):
            aid_j, hash_j = pairs[j]
            dist = hamming_distance_int(hash_i, hash_j)
            if dist <= threshold:
                uf.union(aid_i, aid_j)
            # Early-exit: if integer distance is very large, Hamming can't be ≤ threshold.
            # This is a heuristic — not an exact bound, but saves work in practice.
            elif (hash_j - hash_i) > (1 << (threshold + 4)):
                break

    if on_progress:
        on_progress(n, n)

    # Collect components that meet the minimum size
    components = {
        root: members
        for root, members in uf.components().items()
        if len(members) >= min_cluster_size
    }

    if not components:
        return 0

    # Delete previous run with same method+params
    delete_clusters_by_method_params(conn, _METHOD, params_json)

    # Persist clusters
    n_persisted = 0
    for members in sorted(components.values(), key=lambda m: min(m)):
        representative = min(members)  # lowest asset_id as representative
        cluster_id = insert_cluster(conn, _METHOD, params_json)
        rows_to_insert: list[tuple[int, float | None, int]] = [
            (aid, None, 1 if aid == representative else 0) for aid in members
        ]
        bulk_insert_cluster_members(conn, cluster_id, rows_to_insert)
        n_persisted += 1

    return n_persisted
