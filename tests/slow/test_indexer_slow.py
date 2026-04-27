"""Slow integration tests for the index CLI command."""

from __future__ import annotations

from pathlib import Path

import pytest

from takeout_rater.cli import main

FIXTURE_TAKEOUT = Path(__file__).parent.parent / "fixtures" / "takeout_tree" / "Takeout"


@pytest.fixture()
def library_root(tmp_path: Path) -> Path:
    photos_link = tmp_path / "photos"
    photos_link.symlink_to(FIXTURE_TAKEOUT.resolve(), target_is_directory=True)
    return photos_link


def _db_root_for(photos_root: Path) -> Path:
    return photos_root.parent / "state"


def _index_args(photos_root: Path) -> list[str]:
    return ["index", "--db-root", str(_db_root_for(photos_root)), str(photos_root)]


def test_index_command_returns_zero(library_root: Path) -> None:
    rc = main(_index_args(library_root))
    assert rc == 0
