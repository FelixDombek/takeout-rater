"""Tests for the database schema, connection, and query helpers."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from takeout_rater.db.connection import library_state_dir, open_library_db
from takeout_rater.db.queries import (
    AssetRow,
    count_assets,
    get_asset_by_id,
    get_asset_by_relpath,
    list_assets,
    upsert_asset,
)
from takeout_rater.db.schema import migrate

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


def test_schema_creates_scorer_runs_table() -> None:
    conn = _open_in_memory()
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert "scorer_runs" in tables


def test_schema_creates_asset_scores_table() -> None:
    conn = _open_in_memory()
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert "asset_scores" in tables


def test_schema_user_version_is_2() -> None:
    conn = _open_in_memory()
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == 2


def test_migrate_is_idempotent() -> None:
    """Running migrate twice on the same DB must not raise."""
    conn = _open_in_memory()
    migrate(conn)  # second run
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == 2


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
