"""Integration tests for the index CLI command.

These tests exercise the full pipeline: scan → parse sidecar → upsert to DB →
generate thumbnails.  They use the fixture Takeout tree and a temporary
library root.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from takeout_rater.cli import main
from takeout_rater.db.connection import open_library_db
from takeout_rater.db.queries import count_assets, get_asset_by_relpath, list_assets

FIXTURE_TAKEOUT = Path(__file__).parent / "fixtures" / "takeout_tree" / "Takeout"


@pytest.fixture()
def library_root(tmp_path: Path) -> Path:
    """A temporary library root that already contains a Takeout directory."""
    # Symlink the fixture Takeout into the temp directory so scanning works
    # without copying large files.
    takeout_link = tmp_path / "Takeout"
    takeout_link.symlink_to(FIXTURE_TAKEOUT.resolve(), target_is_directory=True)
    return tmp_path


# ── index command: basic operation ───────────────────────────────────────────


def test_index_command_returns_zero(library_root: Path) -> None:
    rc = main(["index", str(library_root)])
    assert rc == 0


def test_index_command_creates_db(library_root: Path) -> None:
    main(["index", str(library_root)])
    db_path = library_root / "takeout-rater" / "library.sqlite"
    assert db_path.exists()


def test_index_command_populates_assets(library_root: Path) -> None:
    main(["index", str(library_root)])
    conn = open_library_db(library_root)
    total = count_assets(conn)
    conn.close()
    assert total >= 2  # fixture tree has at least a JPEG and a PNG


def test_index_command_jpg_has_sidecar_data(library_root: Path) -> None:
    main(["index", str(library_root)])
    conn = open_library_db(library_root)
    row = get_asset_by_relpath(conn, "Photos from 2023/IMG_20230615_142301.jpg")
    conn.close()
    assert row is not None
    assert row.taken_at == 1686836381
    assert row.title == "IMG_20230615_142301.jpg"
    assert row.origin_type == "mobileUpload"
    assert row.geo_lat == pytest.approx(48.0)


def test_index_command_is_idempotent(library_root: Path) -> None:
    """Running index twice must not duplicate assets."""
    main(["index", str(library_root)])
    main(["index", str(library_root)])
    conn = open_library_db(library_root)
    total = count_assets(conn)
    conn.close()
    # Should still be only the assets from the fixture tree
    assert total >= 2  # at least the fixture assets
    # Uniqueness is enforced by the DB; check no duplicate relpaths
    rows = list_assets(conn=open_library_db(library_root), limit=1000)
    relpaths = [r.relpath for r in rows]
    assert len(relpaths) == len(set(relpaths))


def test_index_command_generates_thumbnails(library_root: Path) -> None:
    main(["index", str(library_root)])
    thumbs_dir = library_root / "takeout-rater" / "thumbs"
    assert thumbs_dir.exists()
    thumb_files = list(thumbs_dir.rglob("*.jpg"))
    assert len(thumb_files) >= 1


def test_index_command_no_thumbs_flag(library_root: Path) -> None:
    rc = main(["index", "--no-thumbs", str(library_root)])
    assert rc == 0
    thumbs_dir = library_root / "takeout-rater" / "thumbs"
    # Directory is created but no files inside
    thumb_files = list(thumbs_dir.rglob("*.jpg")) if thumbs_dir.exists() else []
    assert len(thumb_files) == 0


# ── index command: error cases ────────────────────────────────────────────────


def test_index_command_nonexistent_path_returns_nonzero(tmp_path: Path) -> None:
    rc = main(["index", str(tmp_path / "does_not_exist")])
    assert rc == 1


def test_index_command_missing_takeout_dir_returns_nonzero(tmp_path: Path) -> None:
    """A library root that has no Takeout/ subdirectory should fail."""
    rc = main(["index", str(tmp_path)])
    assert rc == 1
