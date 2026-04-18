"""Integration tests for the index CLI command – fast error-path cases only.

Tests that exercise the full pipeline (scan → thumbnails → DB) live in
tests/slow/test_indexer_slow.py because thumbnail generation hangs in
environments without a display or GPU.
"""

from __future__ import annotations

from pathlib import Path

from takeout_rater.cli import main
from takeout_rater.db.connection import open_library_db

# ── index command: error cases ────────────────────────────────────────────────


def test_index_command_nonexistent_path_returns_nonzero(tmp_path: Path) -> None:
    rc = main(["index", "--db-root", str(tmp_path / "state"), str(tmp_path / "does_not_exist")])
    assert rc == 1


def test_index_command_empty_dir_returns_zero(tmp_path: Path) -> None:
    """An empty photos directory should succeed (just index 0 photos)."""
    photos = tmp_path / "photos"
    photos.mkdir()
    rc = main(["index", "--db-root", str(tmp_path / "state"), str(photos)])
    assert rc == 0


def test_index_command_db_root_option(tmp_path: Path) -> None:
    """--db-root places the state directory in the specified folder."""
    from takeout_rater.db.connection import library_db_path  # noqa: PLC0415

    photos = tmp_path / "photos"
    photos.mkdir()
    state = tmp_path / "state"
    state.mkdir()

    rc = main(["index", "--db-root", str(state), str(photos)])
    assert rc == 0
    # DB must be inside state/, not photos/
    assert library_db_path(state).exists()
    assert not library_db_path(photos).exists()


# ── run_index function: separate db_root ─────────────────────────────────────


def test_run_index_separate_db_root(tmp_path: Path) -> None:
    """run_index with db_root different from photos_root stores DB in db_root."""
    from takeout_rater.db.connection import library_db_path  # noqa: PLC0415
    from takeout_rater.indexing.run import run_index  # noqa: PLC0415

    photos_root = tmp_path / "photos"
    photos_root.mkdir()
    db_root = tmp_path / "state"
    db_root.mkdir()

    conn = open_library_db(db_root)
    progress = run_index(photos_root, conn, db_root=db_root)
    conn.close()

    assert progress.done is True
    assert progress.error is None
    assert library_db_path(db_root).exists()
    # DB must NOT be inside photos_root
    assert not library_db_path(photos_root).exists()
