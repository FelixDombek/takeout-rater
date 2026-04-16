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
    rc = main(["index", str(tmp_path / "does_not_exist")])
    assert rc == 1


def test_index_command_missing_takeout_dir_returns_nonzero(tmp_path: Path) -> None:
    """A library root that has no Takeout/ subdirectory should fail."""
    rc = main(["index", str(tmp_path)])
    assert rc == 1


# ── run_index function: error cases ──────────────────────────────────────────


def test_run_index_missing_takeout_returns_error(tmp_path: Path) -> None:
    """run_index must return an error when Takeout/ does not exist."""
    from takeout_rater.indexing.run import run_index  # noqa: PLC0415

    conn = open_library_db(tmp_path)
    progress = run_index(tmp_path, conn)
    conn.close()

    assert progress.done is True
    assert progress.running is False
    assert progress.error is not None
    assert "Takeout" in progress.error
