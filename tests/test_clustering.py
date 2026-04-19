"""Tests for the pHash-based cluster builder."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from takeout_rater.clustering.builder import (
    _split_by_complete_linkage,
    _UnionFind,
    build_clusters,
    hamming_distance_int,
)
from takeout_rater.db.queries import (
    count_clusters,
    get_cluster_info,
    get_cluster_members,
    list_clusters_with_representatives,
    upsert_asset,
    upsert_phash,
)
from takeout_rater.db.schema import migrate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_in_memory() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    return conn


def _add_asset(conn: sqlite3.Connection, relpath: str = "Photos/img.jpg") -> int:
    return upsert_asset(
        conn,
        {
            "relpath": relpath,
            "filename": Path(relpath).name,
            "ext": Path(relpath).suffix.lower(),
            "size_bytes": 512,
            "mime": "image/jpeg",
            "indexed_at": int(time.time()),
        },
    )


def _add_asset_with_phash(conn: sqlite3.Connection, relpath: str, phash_hex: str) -> int:
    asset_id = _add_asset(conn, relpath)
    upsert_phash(conn, asset_id, phash_hex)
    return asset_id


# ---------------------------------------------------------------------------
# hamming_distance_int
# ---------------------------------------------------------------------------


def test_hamming_distance_int_identical() -> None:
    assert hamming_distance_int(0, 0) == 0
    assert hamming_distance_int(0xDEADBEEF, 0xDEADBEEF) == 0


def test_hamming_distance_int_one_bit() -> None:
    assert hamming_distance_int(0, 1) == 1
    assert hamming_distance_int(0, 2) == 1


def test_hamming_distance_int_all_bits() -> None:
    assert hamming_distance_int(0, 0xFFFFFFFFFFFFFFFF) == 64


def test_hamming_distance_int_symmetric() -> None:
    a, b = 0xABCDEF0123456789, 0x123456789ABCDEF0
    assert hamming_distance_int(a, b) == hamming_distance_int(b, a)


# ---------------------------------------------------------------------------
# _UnionFind
# ---------------------------------------------------------------------------


def test_union_find_single_node() -> None:
    uf = _UnionFind()
    uf.find(1)
    components = uf.components()
    assert 1 in components[uf.find(1)]


def test_union_find_union_merges() -> None:
    uf = _UnionFind()
    uf.union(1, 2)
    uf.union(2, 3)
    assert uf.find(1) == uf.find(3)


def test_union_find_disjoint_sets() -> None:
    uf = _UnionFind()
    uf.union(1, 2)
    uf.union(3, 4)
    assert uf.find(1) != uf.find(3)
    comps = uf.components()
    sizes = sorted(len(v) for v in comps.values())
    assert sizes == [2, 2]


def test_union_find_path_compression() -> None:
    uf = _UnionFind()
    for i in range(10):
        uf.union(i, i + 1)
    root = uf.find(0)
    for i in range(11):
        assert uf.find(i) == root


# ---------------------------------------------------------------------------
# build_clusters
# ---------------------------------------------------------------------------


def test_build_clusters_no_phashes_returns_zero() -> None:
    conn = _open_in_memory()
    _add_asset(conn, "p/a.jpg")
    n_clusters, n_skipped = build_clusters(conn)
    assert n_clusters == 0
    assert n_skipped == 0
    assert count_clusters(conn) == 0


def test_build_clusters_no_duplicates_returns_zero() -> None:
    """Completely different hashes → no clusters."""
    conn = _open_in_memory()
    # Use maximally different hashes
    _add_asset_with_phash(conn, "p/a.jpg", "0000000000000000")
    _add_asset_with_phash(conn, "p/b.jpg", "ffffffffffffffff")
    n_clusters, n_skipped = build_clusters(conn, threshold=10)
    assert n_clusters == 0
    assert n_skipped == 0


def test_build_clusters_identical_hashes_cluster() -> None:
    """Identical hashes → one cluster."""
    conn = _open_in_memory()
    _add_asset_with_phash(conn, "p/a.jpg", "aabbccdd11223344")
    _add_asset_with_phash(conn, "p/b.jpg", "aabbccdd11223344")
    n_clusters, n_skipped = build_clusters(conn, threshold=10)
    assert n_clusters == 1
    assert n_skipped == 0
    assert count_clusters(conn) == 1


def test_build_clusters_within_threshold_cluster() -> None:
    """Hashes within threshold → cluster."""
    conn = _open_in_memory()
    # Hashes that differ in only 1 bit (last bit)
    _add_asset_with_phash(conn, "p/a.jpg", "0000000000000000")
    _add_asset_with_phash(conn, "p/b.jpg", "0000000000000001")
    n_clusters, _ = build_clusters(conn, threshold=5)
    assert n_clusters == 1


def test_build_clusters_just_above_threshold_no_cluster() -> None:
    """Hashes differing in more bits than threshold → not clustered."""
    conn = _open_in_memory()
    # These hashes differ in 11 bits (just above threshold=10)
    h1 = "0000000000000000"
    # Create hash with 11 bits set: 0x7FF = 0000 0000 0000 07FF → last 11 bits set
    h2 = f"{0x7FF:016x}"
    from takeout_rater.clustering.phash import hamming_distance

    dist = hamming_distance(h1, h2)
    assert dist == 11, f"Expected 11 bit difference, got {dist}"

    _add_asset_with_phash(conn, "p/a.jpg", h1)
    _add_asset_with_phash(conn, "p/b.jpg", h2)
    n_clusters, _ = build_clusters(conn, threshold=10)
    assert n_clusters == 0


def test_build_clusters_three_assets_two_clustered() -> None:
    """Three assets: two near-duplicates + one unrelated."""
    conn = _open_in_memory()
    _add_asset_with_phash(conn, "p/a.jpg", "0000000000000000")
    _add_asset_with_phash(conn, "p/b.jpg", "0000000000000001")  # 1 bit diff
    _add_asset_with_phash(conn, "p/c.jpg", "ffffffffffffffff")  # 64 bit diff
    n_clusters, _ = build_clusters(conn, threshold=5)
    assert n_clusters == 1
    assert count_clusters(conn) == 1


def test_build_clusters_representative_is_lowest_id() -> None:
    """The representative should have the lowest asset_id in the cluster."""
    conn = _open_in_memory()
    id_a = _add_asset_with_phash(conn, "p/a.jpg", "0000000000000000")
    id_b = _add_asset_with_phash(conn, "p/b.jpg", "0000000000000001")
    # Ensure a has lower ID
    assert id_a < id_b

    build_clusters(conn, threshold=5)
    members = get_cluster_members(conn, 1)
    reps = [a for a, _d, is_rep in members if is_rep]
    assert len(reps) == 1
    assert reps[0].id == id_a


def test_build_clusters_members_returned() -> None:
    conn = _open_in_memory()
    _add_asset_with_phash(conn, "p/a.jpg", "0000000000000000")
    _add_asset_with_phash(conn, "p/b.jpg", "0000000000000001")
    build_clusters(conn, threshold=5)

    members = get_cluster_members(conn, 1)
    assert len(members) == 2
    asset_ids = {a.id for a, _d, _r in members}
    assert len(asset_ids) == 2


def test_build_clusters_each_run_is_independent() -> None:
    """Running build_clusters twice creates two independent clustering runs."""
    conn = _open_in_memory()
    _add_asset_with_phash(conn, "p/a.jpg", "0000000000000000")
    _add_asset_with_phash(conn, "p/b.jpg", "0000000000000001")

    build_clusters(conn, threshold=5)
    build_clusters(conn, threshold=5)  # second run creates a new clustering_run

    # Each run produces one cluster, so two runs → two clusters total.
    assert count_clusters(conn) == 2


def test_build_clusters_progress_callback() -> None:
    conn = _open_in_memory()
    for i in range(5):
        _add_asset_with_phash(conn, f"p/{i}.jpg", f"{i:016x}")

    calls: list[tuple[int, int]] = []
    build_clusters(conn, threshold=5, on_progress=lambda d, t: calls.append((d, t)))
    assert len(calls) > 0
    # Last call should have done == total
    assert calls[-1][0] == calls[-1][1]


def test_build_clusters_post_progress_callback() -> None:
    """on_post_progress is called after each multi-member component is post-processed."""
    conn = _open_in_memory()
    _add_asset_with_phash(conn, "p/a.jpg", "0000000000000000")
    _add_asset_with_phash(conn, "p/b.jpg", "0000000000000001")

    calls: list[tuple[int, int]] = []
    build_clusters(conn, threshold=5, on_post_progress=lambda d, t: calls.append((d, t)))
    assert len(calls) > 0
    # Last call should signal completion
    assert calls[-1][0] == calls[-1][1]


def test_build_clusters_save_progress_callback() -> None:
    """on_save_progress is called after each cluster is written to the DB."""
    conn = _open_in_memory()
    _add_asset_with_phash(conn, "p/a.jpg", "0000000000000000")
    _add_asset_with_phash(conn, "p/b.jpg", "0000000000000001")

    calls: list[tuple[int, int]] = []
    build_clusters(conn, threshold=5, on_save_progress=lambda d, t: calls.append((d, t)))
    assert len(calls) > 0
    # Last call should signal completion
    assert calls[-1][0] == calls[-1][1]


def test_build_clusters_post_progress_not_called_when_no_clusters() -> None:
    """on_post_progress is not called when there are no multi-member components."""
    conn = _open_in_memory()
    _add_asset_with_phash(conn, "p/a.jpg", "0000000000000000")
    _add_asset_with_phash(conn, "p/b.jpg", "ffffffffffffffff")  # very different

    calls: list[tuple[int, int]] = []
    build_clusters(conn, threshold=5, on_post_progress=lambda d, t: calls.append((d, t)))
    assert calls == []


def test_build_clusters_min_size_filters_small_groups() -> None:
    """With min_cluster_size=3, pairs are not stored."""
    conn = _open_in_memory()
    _add_asset_with_phash(conn, "p/a.jpg", "0000000000000000")
    _add_asset_with_phash(conn, "p/b.jpg", "0000000000000001")
    result = build_clusters(conn, threshold=5, min_cluster_size=3)
    assert result == (0, 0)
    assert count_clusters(conn) == 0


def test_build_clusters_max_size_skips_large_component() -> None:
    """Components larger than max_cluster_size are skipped and counted."""
    conn = _open_in_memory()
    # Create 3 identical hashes → they form one component of size 3
    _add_asset_with_phash(conn, "p/a.jpg", "0000000000000000")
    _add_asset_with_phash(conn, "p/b.jpg", "0000000000000000")
    _add_asset_with_phash(conn, "p/c.jpg", "0000000000000000")
    # max_cluster_size=2 → component of size 3 is skipped
    n_clusters, n_skipped = build_clusters(conn, threshold=5, max_cluster_size=2)
    assert n_clusters == 0
    assert n_skipped == 1
    assert count_clusters(conn) == 0


def test_build_clusters_max_size_allows_smaller_components() -> None:
    """Components at or below max_cluster_size are processed normally."""
    conn = _open_in_memory()
    _add_asset_with_phash(conn, "p/a.jpg", "0000000000000000")
    _add_asset_with_phash(conn, "p/b.jpg", "0000000000000001")
    n_clusters, n_skipped = build_clusters(conn, threshold=5, max_cluster_size=2)
    assert n_clusters == 1
    assert n_skipped == 0


def test_build_clusters_n_skipped_stored_in_db() -> None:
    """n_skipped is persisted in the clustering_runs table."""
    from takeout_rater.db.queries import list_clustering_runs

    conn = _open_in_memory()
    # 3 identical hashes → component of size 3, skipped by max_cluster_size=2
    _add_asset_with_phash(conn, "p/a.jpg", "0000000000000000")
    _add_asset_with_phash(conn, "p/b.jpg", "0000000000000000")
    _add_asset_with_phash(conn, "p/c.jpg", "0000000000000000")
    build_clusters(conn, threshold=5, max_cluster_size=2)
    runs = list_clustering_runs(conn)
    assert len(runs) == 1
    assert runs[0]["n_skipped"] == 1


# ---------------------------------------------------------------------------
# list_clusters_with_representatives
# ---------------------------------------------------------------------------


def test_list_clusters_with_representatives_empty() -> None:
    conn = _open_in_memory()
    result = list_clusters_with_representatives(conn)
    assert result == []


def test_list_clusters_with_representatives_data() -> None:
    conn = _open_in_memory()
    _add_asset_with_phash(conn, "p/a.jpg", "0000000000000000")
    _add_asset_with_phash(conn, "p/b.jpg", "0000000000000001")
    build_clusters(conn, threshold=5)

    result = list_clusters_with_representatives(conn)
    assert len(result) == 1
    row = result[0]
    assert row["member_count"] == 2
    assert row["rep_filename"] in {"a.jpg", "b.jpg"}


def test_list_clusters_pagination() -> None:
    conn = _open_in_memory()
    # Create two separate clusters (different hash ranges, no overlap)
    _add_asset_with_phash(conn, "p/a.jpg", "0000000000000000")
    _add_asset_with_phash(conn, "p/b.jpg", "0000000000000001")
    _add_asset_with_phash(conn, "p/c.jpg", "7777777777777777")
    _add_asset_with_phash(conn, "p/d.jpg", "7777777777777778")
    build_clusters(conn, threshold=5)

    page1 = list_clusters_with_representatives(conn, limit=1, offset=0)
    page2 = list_clusters_with_representatives(conn, limit=1, offset=1)
    assert len(page1) == 1
    assert len(page2) == 1
    assert page1[0]["cluster_id"] != page2[0]["cluster_id"]


# ---------------------------------------------------------------------------
# _split_by_complete_linkage
# ---------------------------------------------------------------------------


def test_split_complete_linkage_pair_within_threshold_unchanged() -> None:
    """A 2-member component where both are within threshold stays as one sub-cluster."""
    hash_map = {1: 0, 2: 1}  # dist = 1
    result = _split_by_complete_linkage([1, 2], hash_map, threshold=5)
    assert len(result) == 1
    assert set(result[0]) == {1, 2}


def test_split_complete_linkage_pair_exceeds_threshold_splits() -> None:
    """A 2-member component where distance exceeds threshold splits into singletons."""
    hash_map = {1: 0, 2: 0x7FF}  # dist = 11
    result = _split_by_complete_linkage([1, 2], hash_map, threshold=5)
    assert len(result) == 2
    assert all(len(sc) == 1 for sc in result)


def test_split_complete_linkage_chain_splits_correctly() -> None:
    """A-B similar, B-C similar, A-C NOT similar → only A-B in same sub-cluster."""
    # A=0, B=0x3FF (bits 0-9), C=0xFFC00 (bits 10-19)
    # dist(A,B)=10, dist(A,C)=10, dist(B,C)=20
    hash_map = {1: 0, 2: 0x3FF, 3: 0xFFC00}
    result = _split_by_complete_linkage([1, 2, 3], hash_map, threshold=10)
    # A and B must be in the same sub-cluster; C must be separate
    sizes = sorted(len(sc) for sc in result)
    assert sizes == [1, 2]
    two_member = next(sc for sc in result if len(sc) == 2)
    assert set(two_member) == {1, 2}


# ---------------------------------------------------------------------------
# Chaining prevention (integration: build_clusters + complete linkage)
# ---------------------------------------------------------------------------


def test_build_clusters_complete_linkage_prevents_chaining() -> None:
    """Single-linkage would chain A-B-C but complete-linkage splits correctly."""
    conn = _open_in_memory()
    # A=0, B=0x3FF (dist(A,B)=10), C=0xFFC00 (dist(A,C)=10, dist(B,C)=20)
    _add_asset_with_phash(conn, "p/a.jpg", "0000000000000000")
    _add_asset_with_phash(conn, "p/b.jpg", f"{0x3FF:016x}")
    _add_asset_with_phash(conn, "p/c.jpg", f"{0xFFC00:016x}")

    # threshold=10: A-B ok, A-C ok (both 10 bits), B-C not ok (20 bits)
    # Single-linkage: {A,B,C} in one component; complete-linkage post: {A,B} + {C}
    # C is a singleton → filtered by min_cluster_size=2 → only 1 cluster stored
    result = build_clusters(conn, threshold=10)
    assert result == (1, 0)
    assert count_clusters(conn) == 1


# ---------------------------------------------------------------------------
# Cluster diameter
# ---------------------------------------------------------------------------


def test_build_clusters_stores_diameter() -> None:
    """build_clusters must store the intra-cluster diameter on each cluster row."""
    conn = _open_in_memory()
    # Hashes differ by 1 bit → diameter = 1
    _add_asset_with_phash(conn, "p/a.jpg", "0000000000000000")
    _add_asset_with_phash(conn, "p/b.jpg", "0000000000000001")
    build_clusters(conn, threshold=5)

    info = get_cluster_info(conn, 1)
    assert info is not None
    assert info["diameter"] == 1.0


def test_build_clusters_identical_hashes_diameter_zero() -> None:
    """Identical hashes → cluster with diameter 0."""
    conn = _open_in_memory()
    _add_asset_with_phash(conn, "p/a.jpg", "aabbccdd11223344")
    _add_asset_with_phash(conn, "p/b.jpg", "aabbccdd11223344")
    build_clusters(conn, threshold=10)

    info = get_cluster_info(conn, 1)
    assert info is not None
    assert info["diameter"] == 0.0


def test_list_clusters_with_representatives_includes_diameter() -> None:
    """list_clusters_with_representatives must include the diameter field."""
    conn = _open_in_memory()
    _add_asset_with_phash(conn, "p/a.jpg", "0000000000000000")
    _add_asset_with_phash(conn, "p/b.jpg", "0000000000000001")
    build_clusters(conn, threshold=5)

    result = list_clusters_with_representatives(conn)
    assert len(result) == 1
    assert "diameter" in result[0]
    assert result[0]["diameter"] == 1.0
