"""Tests for cluster-related DB query helpers."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from takeout_rater.db.queries import (
    bulk_insert_cluster_members,
    count_clusters,
    delete_clustering_run,
    delete_clusters_by_method_params,
    get_cluster_info,
    get_cluster_member_hashes,
    get_cluster_members,
    get_clustering_run,
    insert_cluster,
    insert_clustering_run,
    list_all_phashes,
    list_clustering_runs,
    list_clusters_for_run,
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


def _make_run(conn: sqlite3.Connection, method: str = "m", params_json: str | None = None) -> int:
    """Insert a clustering_run row and return its id."""
    return insert_clustering_run(conn, method, params_json)


# ---------------------------------------------------------------------------
# list_all_phashes
# ---------------------------------------------------------------------------


def test_list_all_phashes_empty() -> None:
    conn = _open_in_memory()
    assert list_all_phashes(conn) == []


def test_list_all_phashes_returns_stored_values() -> None:
    conn = _open_in_memory()
    id1 = _add_asset(conn, "p/a.jpg")
    id2 = _add_asset(conn, "p/b.jpg")
    upsert_phash(conn, id1, "aabb000000000000")
    upsert_phash(conn, id2, "ccdd000000000000")

    result = list_all_phashes(conn)
    assert len(result) == 2
    ids = [r[0] for r in result]
    hashes = [r[1] for r in result]
    assert id1 in ids
    assert id2 in ids
    assert "aabb000000000000" in hashes
    assert "ccdd000000000000" in hashes


def test_list_all_phashes_ordered_by_asset_id() -> None:
    conn = _open_in_memory()
    id1 = _add_asset(conn, "p/a.jpg")
    id2 = _add_asset(conn, "p/b.jpg")
    id3 = _add_asset(conn, "p/c.jpg")
    for aid in (id3, id1, id2):  # insert in non-id order
        upsert_phash(conn, aid, f"{aid:016x}")

    result = list_all_phashes(conn)
    assert [r[0] for r in result] == sorted([id1, id2, id3])


def test_list_all_phashes_algo_filter() -> None:
    """list_all_phashes with algo='dhash16' returns only matching hashes."""
    conn = _open_in_memory()
    id1 = _add_asset(conn, "p/a.jpg")
    id2 = _add_asset(conn, "p/b.jpg")
    upsert_phash(conn, id1, "aabb000000000000", algo="dhash")
    upsert_phash(conn, id2, "ccdd000000000000", algo="dhash16")

    result_all = list_all_phashes(conn)
    assert len(result_all) == 2

    result_dhash16 = list_all_phashes(conn, algo="dhash16")
    assert len(result_dhash16) == 1
    assert result_dhash16[0][0] == id2

    result_dhash = list_all_phashes(conn, algo="dhash")
    assert len(result_dhash) == 1
    assert result_dhash[0][0] == id1


# ---------------------------------------------------------------------------
# insert_cluster + bulk_insert_cluster_members
# ---------------------------------------------------------------------------


def test_insert_cluster_returns_int() -> None:
    conn = _open_in_memory()
    rid = _make_run(conn, "dhash_hamming", '{"threshold":10}')
    cid = insert_cluster(conn, "dhash_hamming", '{"threshold":10}', run_id=rid)
    assert isinstance(cid, int)
    assert cid > 0


def test_insert_cluster_multiple_ids_unique() -> None:
    conn = _open_in_memory()
    rid = _make_run(conn, "dhash_hamming", '{"threshold":10}')
    cid1 = insert_cluster(conn, "dhash_hamming", '{"threshold":10}', run_id=rid)
    cid2 = insert_cluster(conn, "dhash_hamming", '{"threshold":10}', run_id=rid)
    assert cid1 != cid2


def test_bulk_insert_cluster_members_stores_rows() -> None:
    conn = _open_in_memory()
    id1 = _add_asset(conn, "p/a.jpg")
    id2 = _add_asset(conn, "p/b.jpg")
    rid = _make_run(conn)
    cid = insert_cluster(conn, "dhash_hamming", None, run_id=rid)

    bulk_insert_cluster_members(conn, cid, [(id1, None, 1), (id2, 2.5, 0)])

    members = get_cluster_members(conn, cid)
    assert len(members) == 2
    reps = [a for a, _d, is_rep in members if is_rep]
    assert len(reps) == 1
    assert reps[0].id == id1


def test_bulk_insert_cluster_members_ignores_duplicates() -> None:
    conn = _open_in_memory()
    id1 = _add_asset(conn, "p/a.jpg")
    rid = _make_run(conn)
    cid = insert_cluster(conn, "m", None, run_id=rid)
    bulk_insert_cluster_members(conn, cid, [(id1, None, 1)])
    bulk_insert_cluster_members(conn, cid, [(id1, None, 1)])  # duplicate ignored

    members = get_cluster_members(conn, cid)
    assert len(members) == 1


# ---------------------------------------------------------------------------
# delete_clusters_by_method_params
# ---------------------------------------------------------------------------


def test_delete_clusters_by_method_params_removes_rows() -> None:
    conn = _open_in_memory()
    id1 = _add_asset(conn, "p/a.jpg")
    id2 = _add_asset(conn, "p/b.jpg")
    rid = _make_run(conn, "dhash_hamming", '{"threshold":10}')
    cid = insert_cluster(conn, "dhash_hamming", '{"threshold":10}', run_id=rid)
    bulk_insert_cluster_members(conn, cid, [(id1, None, 1), (id2, None, 0)])

    deleted = delete_clusters_by_method_params(conn, "dhash_hamming", '{"threshold":10}')
    assert deleted == 1
    assert count_clusters(conn) == 0

    # Members should also be deleted (FK cascade handled manually)
    members = get_cluster_members(conn, cid)
    assert members == []


def test_delete_clusters_by_method_params_only_matching() -> None:
    conn = _open_in_memory()
    id1 = _add_asset(conn, "p/a.jpg")
    id2 = _add_asset(conn, "p/b.jpg")
    rid = _make_run(conn, "dhash_hamming")
    cid1 = insert_cluster(conn, "dhash_hamming", '{"threshold":10}', run_id=rid)
    cid2 = insert_cluster(conn, "dhash_hamming", '{"threshold":5}', run_id=rid)
    bulk_insert_cluster_members(conn, cid1, [(id1, None, 1)])
    bulk_insert_cluster_members(conn, cid2, [(id2, None, 1)])

    deleted = delete_clusters_by_method_params(conn, "dhash_hamming", '{"threshold":10}')
    assert deleted == 1
    assert count_clusters(conn) == 1


def test_delete_clusters_by_method_params_none_params() -> None:
    conn = _open_in_memory()
    id1 = _add_asset(conn, "p/a.jpg")
    rid = _make_run(conn, "test_method", None)
    cid = insert_cluster(conn, "test_method", None, run_id=rid)
    bulk_insert_cluster_members(conn, cid, [(id1, None, 1)])

    deleted = delete_clusters_by_method_params(conn, "test_method", None)
    assert deleted == 1
    assert count_clusters(conn) == 0


# ---------------------------------------------------------------------------
# count_clusters
# ---------------------------------------------------------------------------


def test_count_clusters_empty() -> None:
    conn = _open_in_memory()
    assert count_clusters(conn) == 0


def test_count_clusters_after_inserts() -> None:
    conn = _open_in_memory()
    rid = _make_run(conn)
    insert_cluster(conn, "m", None, run_id=rid)
    insert_cluster(conn, "m", None, run_id=rid)
    insert_cluster(conn, "m", None, run_id=rid)
    assert count_clusters(conn) == 3


# ---------------------------------------------------------------------------
# get_cluster_members
# ---------------------------------------------------------------------------


def test_get_cluster_members_nonexistent_returns_empty() -> None:
    conn = _open_in_memory()
    assert get_cluster_members(conn, 999) == []


def test_get_cluster_members_representative_first() -> None:
    conn = _open_in_memory()
    id_a = _add_asset(conn, "p/a.jpg")
    id_b = _add_asset(conn, "p/b.jpg")
    rid = _make_run(conn)
    cid = insert_cluster(conn, "m", None, run_id=rid)
    # Insert non-rep first, then rep
    bulk_insert_cluster_members(conn, cid, [(id_b, 2.0, 0), (id_a, 0.0, 1)])

    members = get_cluster_members(conn, cid)
    assert len(members) == 2
    # Representative must come first
    assert members[0][2] is True  # is_representative
    assert members[0][0].id == id_a


# ---------------------------------------------------------------------------
# list_clusters_with_representatives
# ---------------------------------------------------------------------------


def test_list_clusters_with_representatives_sorted_by_size() -> None:
    conn = _open_in_memory()
    ids = [_add_asset(conn, f"p/{i}.jpg") for i in range(5)]
    rid = _make_run(conn)

    # Cluster 1: 3 members
    cid1 = insert_cluster(conn, "m", None, run_id=rid)
    bulk_insert_cluster_members(
        conn, cid1, [(ids[0], None, 1), (ids[1], None, 0), (ids[2], None, 0)]
    )

    # Cluster 2: 2 members
    cid2 = insert_cluster(conn, "m", None, run_id=rid)
    bulk_insert_cluster_members(conn, cid2, [(ids[3], None, 1), (ids[4], None, 0)])

    result = list_clusters_with_representatives(conn)
    assert len(result) == 2
    # Largest cluster should be first
    assert result[0]["member_count"] == 3
    assert result[1]["member_count"] == 2


# ---------------------------------------------------------------------------
# insert_cluster with diameter
# ---------------------------------------------------------------------------


def test_insert_cluster_with_diameter() -> None:
    conn = _open_in_memory()
    rid = _make_run(conn, "dhash_hamming", '{"threshold":20}')
    cid = insert_cluster(conn, "dhash_hamming", '{"threshold":20}', diameter=7.0, run_id=rid)
    assert cid > 0
    info = get_cluster_info(conn, cid)
    assert info is not None
    assert info["diameter"] == 7.0


def test_insert_cluster_without_diameter_is_none() -> None:
    conn = _open_in_memory()
    rid = _make_run(conn, "dhash_hamming")
    cid = insert_cluster(conn, "dhash_hamming", None, run_id=rid)
    info = get_cluster_info(conn, cid)
    assert info is not None
    assert info["diameter"] is None


# ---------------------------------------------------------------------------
# get_cluster_info
# ---------------------------------------------------------------------------


def test_get_cluster_info_nonexistent_returns_none() -> None:
    conn = _open_in_memory()
    assert get_cluster_info(conn, 999) is None


def test_get_cluster_info_returns_expected_fields() -> None:
    conn = _open_in_memory()
    rid = _make_run(conn, "dhash_hamming", '{"threshold":20}')
    cid = insert_cluster(conn, "dhash_hamming", '{"threshold":20}', diameter=5.0, run_id=rid)
    info = get_cluster_info(conn, cid)
    assert info is not None
    assert info["cluster_id"] == cid
    assert info["method"] == "dhash_hamming"
    assert info["params_json"] == '{"threshold":20}'
    assert info["diameter"] == 5.0
    assert isinstance(info["created_at"], int)
    assert info["run_id"] == rid


# ---------------------------------------------------------------------------
# get_cluster_member_hashes
# ---------------------------------------------------------------------------


def test_get_cluster_member_hashes_returns_phash_hex() -> None:
    conn = _open_in_memory()
    id1 = _add_asset(conn, "p/a.jpg")
    id2 = _add_asset(conn, "p/b.jpg")
    upsert_phash(conn, id1, "aabb000000000000")
    upsert_phash(conn, id2, "ccdd000000000000")
    rid = _make_run(conn)
    cid = insert_cluster(conn, "m", None, run_id=rid)
    bulk_insert_cluster_members(conn, cid, [(id1, None, 1), (id2, None, 0)])

    result = get_cluster_member_hashes(conn, cid)
    assert result[id1] == "aabb000000000000"
    assert result[id2] == "ccdd000000000000"


def test_get_cluster_member_hashes_without_phash_returns_none() -> None:
    conn = _open_in_memory()
    id1 = _add_asset(conn, "p/a.jpg")
    rid = _make_run(conn)
    cid = insert_cluster(conn, "m", None, run_id=rid)
    bulk_insert_cluster_members(conn, cid, [(id1, None, 1)])

    result = get_cluster_member_hashes(conn, cid)
    assert result[id1] is None


def test_get_cluster_member_hashes_nonexistent_cluster_returns_empty() -> None:
    conn = _open_in_memory()
    result = get_cluster_member_hashes(conn, 999)
    assert result == {}


# ---------------------------------------------------------------------------
# insert_clustering_run / list_clustering_runs / get_clustering_run
# ---------------------------------------------------------------------------


def test_insert_clustering_run_returns_int() -> None:
    conn = _open_in_memory()
    rid = insert_clustering_run(conn, "dhash_hamming", '{"threshold":10}')
    assert isinstance(rid, int)
    assert rid > 0


def test_insert_clustering_run_multiple_unique_ids() -> None:
    conn = _open_in_memory()
    rid1 = insert_clustering_run(conn, "dhash_hamming", '{"threshold":10}')
    rid2 = insert_clustering_run(conn, "dhash_hamming", '{"threshold":20}')
    assert rid1 != rid2


def test_list_clustering_runs_empty() -> None:
    conn = _open_in_memory()
    assert list_clustering_runs(conn) == []


def test_list_clustering_runs_returns_runs() -> None:
    conn = _open_in_memory()
    rid = insert_clustering_run(conn, "dhash_hamming", '{"threshold":10}')
    id1 = _add_asset(conn, "p/a.jpg")
    id2 = _add_asset(conn, "p/b.jpg")
    cid = insert_cluster(conn, "dhash_hamming", '{"threshold":10}', run_id=rid)
    bulk_insert_cluster_members(conn, cid, [(id1, None, 1), (id2, None, 0)])

    runs = list_clustering_runs(conn)
    assert len(runs) == 1
    assert runs[0]["run_id"] == rid
    assert runs[0]["n_clusters"] == 1
    assert runs[0]["method"] == "dhash_hamming"


def test_list_clustering_runs_most_recent_first() -> None:
    conn = _open_in_memory()
    rid1 = insert_clustering_run(conn, "m", None)
    rid2 = insert_clustering_run(conn, "m", None)

    runs = list_clustering_runs(conn)
    # Most-recent run appears first.
    assert runs[0]["run_id"] == rid2
    assert runs[1]["run_id"] == rid1


def test_get_clustering_run_returns_fields() -> None:
    conn = _open_in_memory()
    rid = insert_clustering_run(conn, "dhash_hamming", '{"threshold":10}')
    run = get_clustering_run(conn, rid)
    assert run is not None
    assert run["run_id"] == rid
    assert run["method"] == "dhash_hamming"
    assert run["params_json"] == '{"threshold":10}'
    assert isinstance(run["created_at"], int)


def test_get_clustering_run_nonexistent_returns_none() -> None:
    conn = _open_in_memory()
    assert get_clustering_run(conn, 999) is None


# ---------------------------------------------------------------------------
# delete_clustering_run
# ---------------------------------------------------------------------------


def test_delete_clustering_run_removes_run_clusters_members() -> None:
    conn = _open_in_memory()
    id1 = _add_asset(conn, "p/a.jpg")
    id2 = _add_asset(conn, "p/b.jpg")
    rid = insert_clustering_run(conn, "dhash_hamming", None)
    cid = insert_cluster(conn, "dhash_hamming", None, run_id=rid)
    bulk_insert_cluster_members(conn, cid, [(id1, None, 1), (id2, None, 0)])

    result = delete_clustering_run(conn, rid)
    assert result is True
    assert get_clustering_run(conn, rid) is None
    assert count_clusters(conn) == 0
    assert get_cluster_members(conn, cid) == []


def test_delete_clustering_run_nonexistent_returns_false() -> None:
    conn = _open_in_memory()
    assert delete_clustering_run(conn, 999) is False


def test_delete_clustering_run_only_removes_its_clusters() -> None:
    conn = _open_in_memory()
    id1 = _add_asset(conn, "p/a.jpg")
    id2 = _add_asset(conn, "p/b.jpg")
    rid1 = insert_clustering_run(conn, "m", None)
    rid2 = insert_clustering_run(conn, "m", None)
    cid1 = insert_cluster(conn, "m", None, run_id=rid1)
    cid2 = insert_cluster(conn, "m", None, run_id=rid2)
    bulk_insert_cluster_members(conn, cid1, [(id1, None, 1)])
    bulk_insert_cluster_members(conn, cid2, [(id2, None, 1)])

    delete_clustering_run(conn, rid1)
    assert count_clusters(conn) == 1
    assert get_cluster_members(conn, cid2) != []


# ---------------------------------------------------------------------------
# list_clusters_for_run
# ---------------------------------------------------------------------------


def test_list_clusters_for_run_returns_clusters() -> None:
    conn = _open_in_memory()
    ids = [_add_asset(conn, f"p/{i}.jpg") for i in range(3)]
    rid = insert_clustering_run(conn, "m", None)
    cid = insert_cluster(conn, "m", None, run_id=rid)
    bulk_insert_cluster_members(
        conn, cid, [(ids[0], None, 1), (ids[1], None, 0), (ids[2], None, 0)]
    )

    clusters = list_clusters_for_run(conn, rid)
    assert len(clusters) == 1
    assert clusters[0]["cluster_id"] == cid
    assert clusters[0]["member_count"] == 3


def test_list_clusters_for_run_empty_run() -> None:
    conn = _open_in_memory()
    rid = insert_clustering_run(conn, "m", None)
    assert list_clusters_for_run(conn, rid) == []


def test_list_clusters_for_run_only_returns_run_clusters() -> None:
    conn = _open_in_memory()
    id1 = _add_asset(conn, "p/a.jpg")
    id2 = _add_asset(conn, "p/b.jpg")
    rid1 = insert_clustering_run(conn, "m", None)
    rid2 = insert_clustering_run(conn, "m", None)
    cid1 = insert_cluster(conn, "m", None, run_id=rid1)
    cid2 = insert_cluster(conn, "m", None, run_id=rid2)
    bulk_insert_cluster_members(conn, cid1, [(id1, None, 1)])
    bulk_insert_cluster_members(conn, cid2, [(id2, None, 1)])

    assert len(list_clusters_for_run(conn, rid1)) == 1
    assert list_clusters_for_run(conn, rid1)[0]["cluster_id"] == cid1
