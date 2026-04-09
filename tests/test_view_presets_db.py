"""Tests for view-preset DB helpers (migration 0002)."""

from __future__ import annotations

import sqlite3
import time

import pytest

from takeout_rater.db.queries import (
    ViewPreset,
    delete_view_preset,
    get_view_preset,
    list_view_presets,
    upsert_view_preset,
)
from takeout_rater.db.schema import migrate

# ── Helpers ───────────────────────────────────────────────────────────────────


def _open_in_memory() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    return conn


# ── list_view_presets ─────────────────────────────────────────────────────────


def test_list_view_presets_empty_initially() -> None:
    conn = _open_in_memory()
    assert list_view_presets(conn) == []


def test_list_view_presets_returns_presets() -> None:
    conn = _open_in_memory()
    upsert_view_preset(conn, "My Preset", sort_by="blur:sharpness")
    presets = list_view_presets(conn)
    assert len(presets) == 1
    assert presets[0].name == "My Preset"


def test_list_view_presets_ordered_by_name() -> None:
    conn = _open_in_memory()
    upsert_view_preset(conn, "Zed")
    upsert_view_preset(conn, "Alpha")
    upsert_view_preset(conn, "Middle")
    names = [p.name for p in list_view_presets(conn)]
    assert names == sorted(names)


# ── upsert_view_preset ────────────────────────────────────────────────────────


def test_upsert_view_preset_returns_int_id() -> None:
    conn = _open_in_memory()
    pid = upsert_view_preset(conn, "Test")
    assert isinstance(pid, int)
    assert pid > 0


def test_upsert_view_preset_stores_fields() -> None:
    conn = _open_in_memory()
    pid = upsert_view_preset(
        conn,
        "Aesthetic High",
        sort_by="aesthetic:aesthetic",
        favorited=1,
        min_score=7.0,
        max_score=None,
    )
    preset = get_view_preset(conn, pid)
    assert preset is not None
    assert preset.sort_by == "aesthetic:aesthetic"
    assert preset.favorited == 1
    assert preset.min_score == pytest.approx(7.0)
    assert preset.max_score is None


def test_upsert_view_preset_update_on_duplicate_name() -> None:
    conn = _open_in_memory()
    pid1 = upsert_view_preset(conn, "Shared", sort_by="blur:sharpness")
    pid2 = upsert_view_preset(conn, "Shared", sort_by="aesthetic:aesthetic")
    assert pid1 == pid2
    preset = get_view_preset(conn, pid1)
    assert preset is not None
    assert preset.sort_by == "aesthetic:aesthetic"


def test_upsert_view_preset_sets_timestamps() -> None:
    conn = _open_in_memory()
    before = int(time.time())
    pid = upsert_view_preset(conn, "Timestamped")
    after = int(time.time())
    preset = get_view_preset(conn, pid)
    assert preset is not None
    assert before <= preset.created_at <= after
    assert before <= preset.updated_at <= after


# ── get_view_preset ───────────────────────────────────────────────────────────


def test_get_view_preset_none_for_missing() -> None:
    conn = _open_in_memory()
    assert get_view_preset(conn, 99999) is None


def test_get_view_preset_returns_view_preset_instance() -> None:
    conn = _open_in_memory()
    pid = upsert_view_preset(conn, "Check Type")
    preset = get_view_preset(conn, pid)
    assert isinstance(preset, ViewPreset)


# ── delete_view_preset ────────────────────────────────────────────────────────


def test_delete_view_preset_returns_true_when_found() -> None:
    conn = _open_in_memory()
    pid = upsert_view_preset(conn, "To Delete")
    assert delete_view_preset(conn, pid) is True


def test_delete_view_preset_returns_false_when_missing() -> None:
    conn = _open_in_memory()
    assert delete_view_preset(conn, 99999) is False


def test_delete_view_preset_removes_row() -> None:
    conn = _open_in_memory()
    pid = upsert_view_preset(conn, "Gone")
    delete_view_preset(conn, pid)
    assert get_view_preset(conn, pid) is None
    assert list_view_presets(conn) == []


# ── migration creates view_presets table ──────────────────────────────────────


def test_view_presets_table_exists_after_migration() -> None:
    conn = _open_in_memory()
    tables = [
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    ]
    assert "view_presets" in tables


def test_schema_version_is_5() -> None:
    conn = _open_in_memory()
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == 5
