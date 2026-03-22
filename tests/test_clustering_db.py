"""Tests for cluster-related DB query helpers."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from takeout_rater.db.queries import (
    bulk_insert_cluster_members,
    count_clusters,
    delete_clusters_by_method_params,
    get_cluster_members,
    insert_cluster,
    list_all_phashes,
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


# ---------------------------------------------------------------------------
# insert_cluster + bulk_insert_cluster_members
# ---------------------------------------------------------------------------


def test_insert_cluster_returns_int() -> None:
    conn = _open_in_memory()
    cid = insert_cluster(conn, "dhash_hamming", '{"threshold":10}')
    assert isinstance(cid, int)
    assert cid > 0


def test_insert_cluster_multiple_ids_unique() -> None:
    conn = _open_in_memory()
    cid1 = insert_cluster(conn, "dhash_hamming", '{"threshold":10}')
    cid2 = insert_cluster(conn, "dhash_hamming", '{"threshold":10}')
    assert cid1 != cid2


def test_bulk_insert_cluster_members_stores_rows() -> None:
    conn = _open_in_memory()
    id1 = _add_asset(conn, "p/a.jpg")
    id2 = _add_asset(conn, "p/b.jpg")
    cid = insert_cluster(conn, "dhash_hamming", None)

    bulk_insert_cluster_members(conn, cid, [(id1, None, 1), (id2, 2.5, 0)])

    members = get_cluster_members(conn, cid)
    assert len(members) == 2
    reps = [a for a, _d, is_rep in members if is_rep]
    assert len(reps) == 1
    assert reps[0].id == id1


def test_bulk_insert_cluster_members_ignores_duplicates() -> None:
    conn = _open_in_memory()
    id1 = _add_asset(conn, "p/a.jpg")
    cid = insert_cluster(conn, "m", None)
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
    cid = insert_cluster(conn, "dhash_hamming", '{"threshold":10}')
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
    cid1 = insert_cluster(conn, "dhash_hamming", '{"threshold":10}')
    cid2 = insert_cluster(conn, "dhash_hamming", '{"threshold":5}')
    bulk_insert_cluster_members(conn, cid1, [(id1, None, 1)])
    bulk_insert_cluster_members(conn, cid2, [(id2, None, 1)])

    deleted = delete_clusters_by_method_params(conn, "dhash_hamming", '{"threshold":10}')
    assert deleted == 1
    assert count_clusters(conn) == 1


def test_delete_clusters_by_method_params_none_params() -> None:
    conn = _open_in_memory()
    id1 = _add_asset(conn, "p/a.jpg")
    cid = insert_cluster(conn, "test_method", None)
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
    insert_cluster(conn, "m", None)
    insert_cluster(conn, "m", None)
    insert_cluster(conn, "m", None)
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
    cid = insert_cluster(conn, "m", None)
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

    # Cluster 1: 3 members
    cid1 = insert_cluster(conn, "m", None)
    bulk_insert_cluster_members(
        conn, cid1, [(ids[0], None, 1), (ids[1], None, 0), (ids[2], None, 0)]
    )

    # Cluster 2: 2 members
    cid2 = insert_cluster(conn, "m", None)
    bulk_insert_cluster_members(conn, cid2, [(ids[3], None, 1), (ids[4], None, 0)])

    result = list_clusters_with_representatives(conn)
    assert len(result) == 2
    # Largest cluster should be first
    assert result[0]["member_count"] == 3
    assert result[1]["member_count"] == 2
