"""Tests for the database schema, connection, and query helpers."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from takeout_rater.db.connection import library_state_dir, open_library_db
from takeout_rater.db.queries import (
    AssetRow,
    count_assets,
    count_assets_deduped,
    get_asset_alias_paths,
    get_asset_by_id,
    get_asset_by_relpath,
    get_duplicate_assets,
    list_asset_ids_without_sha256,
    list_assets,
    list_assets_deduped,
    update_asset_sha256,
    upsert_asset,
)
from takeout_rater.db.schema import CURRENT_SCHEMA_VERSION, SchemaMismatchError, migrate

# ── Helpers ───────────────────────────────────────────────────────────────────


def _minimal_asset(relpath: str = "Photos/img.jpg") -> dict:
    """Return a minimal valid asset dict for upsert_asset."""
    return {
        "relpath": relpath,
        "filename": Path(relpath).name,
        "ext": Path(relpath).suffix.lower(),
        "size_bytes": 1024,
        "mime": "image/jpeg",
        "indexed_at": int(time.time()),
    }


def _open_in_memory() -> sqlite3.Connection:
    """Create an in-memory SQLite DB with the full schema applied."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    return conn


# ── Schema / migration ────────────────────────────────────────────────────────


def test_schema_creates_assets_table() -> None:
    conn = _open_in_memory()
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert "assets" in tables


def test_schema_creates_albums_table() -> None:
    conn = _open_in_memory()
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert "albums" in tables


def test_schema_creates_asset_scores_table() -> None:
    conn = _open_in_memory()
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert "asset_scores" in tables


def test_schema_user_version_is_10() -> None:
    conn = _open_in_memory()
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == CURRENT_SCHEMA_VERSION


def test_migrate_is_idempotent() -> None:
    """Running migrate twice on the same DB must not raise."""
    conn = _open_in_memory()
    migrate(conn)  # second run
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == CURRENT_SCHEMA_VERSION


def test_migrate_raises_on_stale_schema() -> None:
    """migrate() must raise SchemaMismatchError for pre-version-13 databases."""
    for stale_version in range(1, 13):
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA user_version = {stale_version}")
        with pytest.raises(SchemaMismatchError) as exc_info:
            migrate(conn)
        assert exc_info.value.found_version == stale_version
        conn.close()


# ── library_state_dir ────────────────────────────────────────────────────────


def test_library_state_dir_creates_directory(tmp_path: Path) -> None:
    state = library_state_dir(tmp_path)
    assert state.exists()
    assert state.name == "takeout-rater"


def test_library_state_dir_idempotent(tmp_path: Path) -> None:
    library_state_dir(tmp_path)
    library_state_dir(tmp_path)  # must not raise


# ── open_library_db ───────────────────────────────────────────────────────────


def test_open_library_db_creates_db_file(tmp_path: Path) -> None:
    conn = open_library_db(tmp_path)
    conn.close()
    db_path = tmp_path / "takeout-rater" / "library.sqlite"
    assert db_path.exists()


def test_open_library_db_returns_connection(tmp_path: Path) -> None:
    conn = open_library_db(tmp_path)
    assert isinstance(conn, sqlite3.Connection)
    conn.close()


# ── upsert_asset ──────────────────────────────────────────────────────────────


def test_upsert_asset_returns_int() -> None:
    conn = _open_in_memory()
    asset_id = upsert_asset(conn, _minimal_asset())
    assert isinstance(asset_id, int)


def test_upsert_asset_is_idempotent() -> None:
    conn = _open_in_memory()
    id1 = upsert_asset(conn, _minimal_asset("p/a.jpg"))
    id2 = upsert_asset(conn, _minimal_asset("p/a.jpg"))
    assert id1 == id2
    assert count_assets(conn) == 1


def test_upsert_asset_different_relpaths() -> None:
    conn = _open_in_memory()
    upsert_asset(conn, _minimal_asset("p/a.jpg"))
    upsert_asset(conn, _minimal_asset("p/b.jpg"))
    assert count_assets(conn) == 2


def test_upsert_asset_auto_sets_indexed_at() -> None:
    conn = _open_in_memory()
    asset = _minimal_asset()
    del asset["indexed_at"]
    asset_id = upsert_asset(conn, asset)
    row = get_asset_by_id(conn, asset_id)
    assert row is not None
    assert row.indexed_at > 0


# ── get_asset_by_id ───────────────────────────────────────────────────────────


def test_get_asset_by_id_returns_asset_row() -> None:
    conn = _open_in_memory()
    asset_id = upsert_asset(conn, _minimal_asset())
    row = get_asset_by_id(conn, asset_id)
    assert isinstance(row, AssetRow)


def test_get_asset_by_id_missing_returns_none() -> None:
    conn = _open_in_memory()
    assert get_asset_by_id(conn, 999999) is None


def test_get_asset_by_id_relpath_matches() -> None:
    conn = _open_in_memory()
    asset_id = upsert_asset(conn, _minimal_asset("foo/bar.jpg"))
    row = get_asset_by_id(conn, asset_id)
    assert row is not None
    assert row.relpath == "foo/bar.jpg"


# ── get_asset_by_relpath ─────────────────────────────────────────────────────


def test_get_asset_by_relpath_found() -> None:
    conn = _open_in_memory()
    upsert_asset(conn, _minimal_asset("a/b.png"))
    row = get_asset_by_relpath(conn, "a/b.png")
    assert row is not None
    assert row.relpath == "a/b.png"


def test_get_asset_by_relpath_missing_returns_none() -> None:
    conn = _open_in_memory()
    assert get_asset_by_relpath(conn, "no/such.jpg") is None


# ── list_assets ───────────────────────────────────────────────────────────────


def test_list_assets_empty_db() -> None:
    conn = _open_in_memory()
    assert list_assets(conn) == []


def test_list_assets_returns_all() -> None:
    conn = _open_in_memory()
    for i in range(3):
        upsert_asset(conn, _minimal_asset(f"p/{i}.jpg"))
    assert len(list_assets(conn, limit=10)) == 3


def test_list_assets_pagination() -> None:
    conn = _open_in_memory()
    for i in range(5):
        upsert_asset(conn, _minimal_asset(f"p/{i}.jpg"))
    page1 = list_assets(conn, limit=3, offset=0)
    page2 = list_assets(conn, limit=3, offset=3)
    assert len(page1) == 3
    assert len(page2) == 2


def test_list_assets_filter_favorited() -> None:
    conn = _open_in_memory()
    upsert_asset(conn, {**_minimal_asset("a.jpg"), "favorited": 1})
    upsert_asset(conn, {**_minimal_asset("b.jpg"), "favorited": 0})
    upsert_asset(conn, _minimal_asset("c.jpg"))
    result = list_assets(conn, favorited=True)
    assert len(result) == 1
    assert result[0].relpath == "a.jpg"


def test_list_assets_filter_ext() -> None:
    conn = _open_in_memory()
    upsert_asset(conn, _minimal_asset("a.jpg"))
    upsert_asset(conn, {**_minimal_asset("b.png"), "ext": ".png", "mime": "image/png"})
    result = list_assets(conn, ext=".png")
    assert len(result) == 1
    assert result[0].relpath == "b.png"


# ── count_assets ─────────────────────────────────────────────────────────────


def test_count_assets_empty() -> None:
    conn = _open_in_memory()
    assert count_assets(conn) == 0


def test_count_assets_all() -> None:
    conn = _open_in_memory()
    for i in range(4):
        upsert_asset(conn, _minimal_asset(f"p/{i}.jpg"))
    assert count_assets(conn) == 4


def test_count_assets_filter_favorited() -> None:
    conn = _open_in_memory()
    upsert_asset(conn, {**_minimal_asset("a.jpg"), "favorited": 1})
    upsert_asset(conn, {**_minimal_asset("b.jpg"), "favorited": 0})
    assert count_assets(conn, favorited=True) == 1
    assert count_assets(conn, favorited=False) == 1


# ── AssetRow field types ──────────────────────────────────────────────────────


def test_asset_row_favorited_true() -> None:
    conn = _open_in_memory()
    asset_id = upsert_asset(conn, {**_minimal_asset(), "favorited": 1})
    row = get_asset_by_id(conn, asset_id)
    assert row is not None
    assert row.favorited is True


def test_asset_row_favorited_false() -> None:
    conn = _open_in_memory()
    asset_id = upsert_asset(conn, {**_minimal_asset(), "favorited": 0})
    row = get_asset_by_id(conn, asset_id)
    assert row is not None
    assert row.favorited is False


def test_asset_row_favorited_none_when_not_set() -> None:
    conn = _open_in_memory()
    asset_id = upsert_asset(conn, _minimal_asset())
    row = get_asset_by_id(conn, asset_id)
    assert row is not None
    assert row.favorited is None


# ── SHA-256 / deduplication helpers ──────────────────────────────────────────


def test_asset_row_sha256_field_defaults_to_none() -> None:
    conn = _open_in_memory()
    asset_id = upsert_asset(conn, _minimal_asset())
    row = get_asset_by_id(conn, asset_id)
    assert row is not None
    assert row.sha256 is None


def test_asset_row_sha256_stored_and_retrieved() -> None:
    conn = _open_in_memory()
    asset_id = upsert_asset(conn, {**_minimal_asset(), "sha256": "abc123"})
    row = get_asset_by_id(conn, asset_id)
    assert row is not None
    assert row.sha256 == "abc123"


def test_update_asset_sha256() -> None:
    conn = _open_in_memory()
    asset_id = upsert_asset(conn, _minimal_asset())
    update_asset_sha256(conn, asset_id, "deadbeef")
    row = get_asset_by_id(conn, asset_id)
    assert row is not None
    assert row.sha256 == "deadbeef"


def test_list_asset_ids_without_sha256_returns_unhashed() -> None:
    conn = _open_in_memory()
    id_no_hash = upsert_asset(conn, _minimal_asset("p/a.jpg"))
    upsert_asset(conn, {**_minimal_asset("p/b.jpg"), "sha256": "abc123"})
    missing = list_asset_ids_without_sha256(conn)
    assert id_no_hash in missing
    # The asset that already has a sha256 should NOT appear
    for asset_id in missing:
        row = get_asset_by_id(conn, asset_id)
        assert row is not None
        assert row.sha256 is None


def test_list_asset_ids_without_sha256_empty_when_all_hashed() -> None:
    conn = _open_in_memory()
    upsert_asset(conn, {**_minimal_asset("p/a.jpg"), "sha256": "aaa"})
    upsert_asset(conn, {**_minimal_asset("p/b.jpg"), "sha256": "bbb"})
    assert list_asset_ids_without_sha256(conn) == []


def test_get_duplicate_assets_returns_all_copies() -> None:
    # With write-time dedup, only the canonical asset (first inserted) ends up in
    # assets; the second path is stored in asset_paths.  get_duplicate_assets now
    # returns just the canonical row.
    conn = _open_in_memory()
    canonical_id = upsert_asset(conn, {**_minimal_asset("album1/img.jpg"), "sha256": "cafebabe"})
    alias_id = upsert_asset(conn, {**_minimal_asset("album2/img.jpg"), "sha256": "cafebabe"})
    upsert_asset(conn, {**_minimal_asset("album3/other.jpg"), "sha256": "deadbeef"})
    # Both upserts return the same (canonical) id; only one assets row was created.
    assert alias_id == canonical_id
    dups = get_duplicate_assets(conn, "cafebabe")
    assert len(dups) == 1
    assert dups[0].relpath == "album1/img.jpg"


def test_get_duplicate_assets_returns_empty_for_unknown_hash() -> None:
    conn = _open_in_memory()
    upsert_asset(conn, {**_minimal_asset(), "sha256": "aaa"})
    assert get_duplicate_assets(conn, "zzz") == []


# ── upsert_asset sha256 dedup ─────────────────────────────────────────────────


def test_upsert_asset_sha256_duplicate_returns_canonical_id() -> None:
    """Second upsert with same SHA-256 returns the canonical asset's ID."""
    conn = _open_in_memory()
    id1 = upsert_asset(conn, {**_minimal_asset("photos/img.jpg"), "sha256": "abc123"})
    id2 = upsert_asset(conn, {**_minimal_asset("album/img.jpg"), "sha256": "abc123"})
    assert id2 == id1


def test_upsert_asset_sha256_duplicate_stored_in_asset_paths() -> None:
    """Alias path from duplicate is stored in asset_paths."""
    conn = _open_in_memory()
    id1 = upsert_asset(conn, {**_minimal_asset("photos/img.jpg"), "sha256": "abc123"})
    upsert_asset(conn, {**_minimal_asset("album/img.jpg"), "sha256": "abc123"})
    aliases = get_asset_alias_paths(conn, id1)
    assert aliases == ["album/img.jpg"]


def test_upsert_asset_sha256_duplicate_not_in_assets_table() -> None:
    """Alias path must NOT appear as a separate row in assets."""
    conn = _open_in_memory()
    upsert_asset(conn, {**_minimal_asset("photos/img.jpg"), "sha256": "abc123"})
    upsert_asset(conn, {**_minimal_asset("album/img.jpg"), "sha256": "abc123"})
    total = conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
    assert total == 1


def test_upsert_asset_no_sha256_creates_separate_rows() -> None:
    """Without SHA-256, each relpath still gets its own assets row."""
    conn = _open_in_memory()
    id1 = upsert_asset(conn, _minimal_asset("a/img.jpg"))
    id2 = upsert_asset(conn, _minimal_asset("b/img.jpg"))
    assert id1 != id2
    total = conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
    assert total == 2


def test_upsert_asset_reindex_canonical_updates_in_place() -> None:
    """Re-indexing the canonical path still updates the assets row."""
    conn = _open_in_memory()
    id1 = upsert_asset(conn, {**_minimal_asset("photos/img.jpg"), "sha256": "abc123"})
    id2 = upsert_asset(
        conn, {**_minimal_asset("photos/img.jpg"), "sha256": "abc123", "title": "New"}
    )
    assert id2 == id1
    row = conn.execute("SELECT title FROM assets WHERE id = ?", (id1,)).fetchone()
    assert row["title"] == "New"


def test_upsert_asset_multiple_aliases() -> None:
    """Three files with same SHA-256: one canonical, two aliases."""
    conn = _open_in_memory()
    id1 = upsert_asset(conn, {**_minimal_asset("p/img.jpg"), "sha256": "fff"})
    upsert_asset(conn, {**_minimal_asset("a1/img.jpg"), "sha256": "fff"})
    upsert_asset(conn, {**_minimal_asset("a2/img.jpg"), "sha256": "fff"})
    aliases = get_asset_alias_paths(conn, id1)
    assert sorted(aliases) == ["a1/img.jpg", "a2/img.jpg"]
    total = conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
    assert total == 1


# ── get_asset_by_relpath (updated to check asset_paths) ───────────────────────


def test_get_asset_by_relpath_alias_returns_canonical() -> None:
    """Looking up an alias relpath returns the canonical asset."""
    conn = _open_in_memory()
    canonical_id = upsert_asset(conn, {**_minimal_asset("photos/img.jpg"), "sha256": "abc123"})
    upsert_asset(conn, {**_minimal_asset("album/img.jpg"), "sha256": "abc123"})
    row = get_asset_by_relpath(conn, "album/img.jpg")
    assert row is not None
    assert row.id == canonical_id
    assert row.relpath == "photos/img.jpg"


def test_count_assets_deduped_no_duplicates() -> None:
    conn = _open_in_memory()
    upsert_asset(conn, {**_minimal_asset("p/a.jpg"), "sha256": "aaa"})
    upsert_asset(conn, {**_minimal_asset("p/b.jpg"), "sha256": "bbb"})
    # Two unique hashes → still 2
    assert count_assets_deduped(conn) == 2


def test_count_assets_deduped_with_exact_duplicates() -> None:
    conn = _open_in_memory()
    upsert_asset(conn, {**_minimal_asset("album1/img.jpg"), "sha256": "cafebabe"})
    upsert_asset(conn, {**_minimal_asset("album2/img.jpg"), "sha256": "cafebabe"})
    upsert_asset(conn, {**_minimal_asset("p/unique.jpg"), "sha256": "deadbeef"})
    # Write-time dedup: album2 → asset_paths; only 2 rows in assets
    assert count_assets_deduped(conn) == 2


def test_count_assets_deduped_null_sha256_each_counted() -> None:
    conn = _open_in_memory()
    upsert_asset(conn, _minimal_asset("p/a.jpg"))
    upsert_asset(conn, _minimal_asset("p/b.jpg"))
    # Both have no sha256 → treated as separate unique images
    assert count_assets_deduped(conn) == 2


def test_list_assets_deduped_hides_duplicate() -> None:
    conn = _open_in_memory()
    upsert_asset(conn, {**_minimal_asset("album1/img.jpg"), "sha256": "cafebabe"})
    upsert_asset(conn, {**_minimal_asset("album2/img.jpg"), "sha256": "cafebabe"})
    upsert_asset(conn, {**_minimal_asset("p/other.jpg"), "sha256": "deadbeef"})
    rows = list_assets_deduped(conn)
    assert len(rows) == 2
    relpaths = {r.relpath for r in rows}
    # Write-time dedup: only album1 is in assets; album2 is in asset_paths
    assert "album1/img.jpg" in relpaths
    assert "album2/img.jpg" not in relpaths


def test_list_assets_deduped_shows_all_when_no_sha256() -> None:
    conn = _open_in_memory()
    upsert_asset(conn, _minimal_asset("p/a.jpg"))
    upsert_asset(conn, _minimal_asset("p/b.jpg"))
    rows = list_assets_deduped(conn)
    assert len(rows) == 2


def test_count_assets_deduped_favorited_filter() -> None:
    conn = _open_in_memory()
    upsert_asset(conn, {**_minimal_asset("a/fav.jpg"), "sha256": "aaa", "favorited": 1})
    upsert_asset(conn, {**_minimal_asset("b/fav.jpg"), "sha256": "aaa", "favorited": 1})
    upsert_asset(conn, {**_minimal_asset("c/other.jpg"), "sha256": "bbb", "favorited": 0})
    assert count_assets_deduped(conn, favorited=True) == 1
    assert count_assets_deduped(conn, favorited=False) == 1


# ── Browse paging: dedup + stable order ───────────────────────────────────────


def _insert_raw(
    conn: sqlite3.Connection,
    asset_id: int,
    sha: str | None,
    taken_at: int | None,
    favorited: int = 0,
) -> None:
    # Raw insert: upsert_asset merges equal sha256 at write time, but older
    # libraries (and lazily hashed rows) can still hold duplicate rows.
    conn.execute(
        "INSERT INTO assets (id, relpath, filename, ext, sha256, taken_at, favorited, indexed_at)"
        " VALUES (?, ?, ?, '.jpg', ?, ?, ?, 0)",
        (asset_id, f"p/{asset_id}.jpg", f"{asset_id}.jpg", sha, taken_at, favorited),
    )


def _window_dedup_reference(conn: sqlite3.Connection, where: str = "") -> list[int]:
    rows = conn.execute(
        "SELECT id FROM (SELECT id, taken_at, ROW_NUMBER() OVER ("
        " PARTITION BY COALESCE(sha256, CAST(id AS TEXT)) ORDER BY id) AS rn"
        f" FROM assets {where}) WHERE rn = 1 ORDER BY taken_at DESC, id DESC"
    ).fetchall()
    return [r[0] for r in rows]


def test_list_assets_deduped_matches_window_reference_across_pages() -> None:
    conn = _open_in_memory()
    # Many equal timestamps, some NULL timestamps, NULL hashes and duplicate
    # hashes whose lowest-id copy is not favorited.
    for i in range(1, 121):
        sha = None if i % 11 == 0 else f"dup{i % 7}" if i % 5 == 0 else f"s{i}"
        taken = None if i % 13 == 0 else 1_700_000_000 - (i // 4)
        _insert_raw(conn, i, sha, taken, favorited=1 if i % 3 == 0 else 0)
    conn.commit()

    for fav, where in ((None, ""), (True, "WHERE favorited = 1")):
        expected = _window_dedup_reference(conn, where)
        paged: list[int] = []
        for offset in range(0, 200, 7):
            paged += [
                a.id for a in list_assets_deduped(conn, limit=7, offset=offset, favorited=fav)
            ]
        assert paged == expected
        assert len(paged) == count_assets_deduped(conn, favorited=fav)


def test_list_assets_orders_equal_timestamps_by_id() -> None:
    conn = _open_in_memory()
    for i in range(1, 11):
        _insert_raw(conn, i, f"s{i}", 1_700_000_000)
    conn.commit()
    pages = [list_assets(conn, limit=3, offset=o) for o in range(0, 12, 3)]
    ids = [a.id for page in pages for a in page]
    assert ids == list(range(10, 0, -1))


def test_migration_19_adds_score_value_index() -> None:
    conn = sqlite3.connect(":memory:")
    migrate(conn)
    # Put the database back into its v18 shape, then migrate forward.
    conn.execute("DROP INDEX idx_asset_scores_metric_value")
    conn.execute(
        "CREATE INDEX idx_asset_scores_scorer ON asset_scores (scorer_id, variant_id, metric_key)"
    )
    conn.execute("PRAGMA user_version = 18")
    migrate(conn)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
    assert "idx_asset_scores_metric_value" in names
    assert "idx_asset_scores_scorer" not in names
    assert conn.execute("PRAGMA user_version").fetchone()[0] == CURRENT_SCHEMA_VERSION
    plan = " ".join(
        str(r[3])
        for r in conn.execute(
            "EXPLAIN QUERY PLAN SELECT s.asset_id FROM asset_scores s"
            " WHERE s.scorer_id = 'x' AND s.variant_id = 'y' AND s.metric_key = 'z'"
            " ORDER BY s.value DESC, s.asset_id DESC LIMIT 5"
        )
    )
    assert "idx_asset_scores_metric_value" in plan
    assert "TEMP B-TREE" not in plan
