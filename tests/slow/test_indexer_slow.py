"""Slow integration tests for the index CLI command.

These tests exercise the full pipeline: scan → parse sidecar → upsert to DB →
generate thumbnails.  They use the fixture Takeout tree and a temporary
library root.  They are excluded from the default test run (--ignore=tests/slow)
because thumbnail generation hangs in environments without a display or GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from takeout_rater.cli import main
from takeout_rater.db.connection import open_library_db
from takeout_rater.db.queries import count_assets, get_asset_by_relpath, list_assets

FIXTURE_TAKEOUT = Path(__file__).parent.parent / "fixtures" / "takeout_tree" / "Takeout"


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


# ── index command: Google Photos subdirectory layout ─────────────────────────


@pytest.fixture()
def library_root_google_photos_subdir(tmp_path: Path) -> Path:
    """Library root whose Takeout contains a 'Google Photos' subdirectory.

    Simulates the layout where Google Takeout nests all albums inside
    ``Takeout/Google Photos/`` instead of directly in ``Takeout/``.
    Also adds an unrelated image under a sibling directory to verify it is
    not indexed.
    """
    takeout = tmp_path / "Takeout"
    google_photos = takeout / "Google Photos"
    album = google_photos / "Photos from 2023"
    album.mkdir(parents=True)
    (album / "img.jpg").write_bytes(b"\xff\xd8\xff")

    # Unrelated Google product directory that must NOT be indexed
    other = takeout / "Google Drive"
    other.mkdir()
    (other / "drive_image.jpg").write_bytes(b"\xff\xd8\xff")

    return tmp_path


def test_index_google_photos_subdir_returns_zero(
    library_root_google_photos_subdir: Path,
) -> None:
    rc = main(["index", str(library_root_google_photos_subdir)])
    assert rc == 0


def test_index_google_photos_subdir_indexes_only_photos(
    library_root_google_photos_subdir: Path,
) -> None:
    """Only the image inside Google Photos/ should be indexed; Drive image must not be."""
    main(["index", str(library_root_google_photos_subdir)])
    conn = open_library_db(library_root_google_photos_subdir)
    rows = list_assets(conn, limit=100)
    conn.close()

    relpaths = [r.relpath for r in rows]
    assert len(relpaths) == 1, f"expected 1 asset, got {len(relpaths)}: {relpaths}"
    # relpath is relative to the Google Photos root, so it should NOT include
    # 'Google Photos/' as a prefix
    assert relpaths[0] == "Photos from 2023/img.jpg"


def test_index_google_photos_subdir_relpath_excludes_google_photos_prefix(
    library_root_google_photos_subdir: Path,
) -> None:
    """relpaths must be relative to 'Google Photos/', not to 'Takeout/'."""
    main(["index", str(library_root_google_photos_subdir)])
    conn = open_library_db(library_root_google_photos_subdir)
    row = get_asset_by_relpath(conn, "Photos from 2023/img.jpg")
    conn.close()
    assert row is not None


@pytest.fixture()
def library_root_google_fotos_subdir(tmp_path: Path) -> Path:
    """Library root with the German-localized 'Google Fotos' subdirectory."""
    takeout = tmp_path / "Takeout"
    google_fotos = takeout / "Google Fotos"
    album = google_fotos / "Fotos aus 2023"
    album.mkdir(parents=True)
    (album / "bild.jpg").write_bytes(b"\xff\xd8\xff")
    return tmp_path


def test_index_google_fotos_subdir_returns_zero(
    library_root_google_fotos_subdir: Path,
) -> None:
    rc = main(["index", str(library_root_google_fotos_subdir)])
    assert rc == 0


def test_index_google_fotos_subdir_indexes_photo(
    library_root_google_fotos_subdir: Path,
) -> None:
    main(["index", str(library_root_google_fotos_subdir)])
    conn = open_library_db(library_root_google_fotos_subdir)
    rows = list_assets(conn, limit=100)
    conn.close()
    assert len(rows) == 1
    assert rows[0].relpath == "Fotos aus 2023/bild.jpg"


# ── run_index function (background-callable API) ──────────────────────────────


def test_run_index_returns_progress_with_indexed_count(library_root: Path) -> None:
    """run_index must upsert assets and return the correct count."""
    from takeout_rater.indexing.run import run_index  # noqa: PLC0415

    conn = open_library_db(library_root)
    progress = run_index(library_root, conn)
    conn.close()

    assert progress.done is True
    assert progress.running is False
    assert progress.error is None
    assert progress.indexed >= 2  # fixture has at least a JPEG and a PNG
    assert progress.found == progress.indexed


def test_run_index_populates_db(library_root: Path) -> None:
    """After run_index the DB must contain the indexed assets."""
    from takeout_rater.indexing.run import run_index  # noqa: PLC0415

    conn = open_library_db(library_root)
    run_index(library_root, conn)
    total = count_assets(conn)
    conn.close()

    assert total >= 2


def test_run_index_is_idempotent(library_root: Path) -> None:
    """Calling run_index twice must not duplicate assets."""
    from takeout_rater.indexing.run import run_index  # noqa: PLC0415

    conn = open_library_db(library_root)
    run_index(library_root, conn)
    run_index(library_root, conn)
    total = count_assets(conn)
    conn.close()

    conn2 = open_library_db(library_root)
    rows = list_assets(conn=conn2, limit=1000)
    conn2.close()
    relpaths = [r.relpath for r in rows]
    assert len(relpaths) == len(set(relpaths))
    assert total == len(relpaths)


# ── run_index progress fields ─────────────────────────────────────────────────


def test_run_index_final_progress_phase_is_processing(library_root: Path) -> None:
    """The final IndexProgress must have phase='processing' (not 'scanning')."""
    from takeout_rater.indexing.run import run_index  # noqa: PLC0415

    conn = open_library_db(library_root)
    progress = run_index(library_root, conn)
    conn.close()

    assert progress.phase == "processing"


def test_run_index_final_progress_dirs_scanned(library_root: Path) -> None:
    """dirs_scanned must be positive after a successful run."""
    from takeout_rater.indexing.run import run_index  # noqa: PLC0415

    conn = open_library_db(library_root)
    progress = run_index(library_root, conn)
    conn.close()

    assert progress.dirs_scanned > 0
    assert progress.dirs_scanned == progress.total_dirs


def test_run_index_on_progress_called_during_scanning(library_root: Path) -> None:
    """on_progress must be invoked while scanning (phase='scanning')."""
    from takeout_rater.indexing.run import run_index  # noqa: PLC0415

    scanning_calls: list[object] = []

    def _cb(p: object) -> None:
        from takeout_rater.indexing.run import IndexProgress  # noqa: PLC0415

        assert isinstance(p, IndexProgress)
        if p.phase == "scanning":
            scanning_calls.append(p)

    conn = open_library_db(library_root)
    run_index(library_root, conn, on_progress=_cb)
    conn.close()

    assert len(scanning_calls) > 0, "on_progress was never called with phase='scanning'"


def test_run_index_on_progress_called_during_processing(library_root: Path) -> None:
    """on_progress must be invoked while processing (phase='processing')."""
    from takeout_rater.indexing.run import run_index  # noqa: PLC0415

    processing_calls: list[object] = []

    def _cb(p: object) -> None:
        from takeout_rater.indexing.run import IndexProgress  # noqa: PLC0415

        if isinstance(p, IndexProgress) and p.phase == "processing" and not p.done:
            processing_calls.append(p)

    conn = open_library_db(library_root)
    run_index(library_root, conn, on_progress=_cb)
    conn.close()

    assert len(processing_calls) > 0, "on_progress was never called with phase='processing'"


# ── SHA-256 computation during indexing ──────────────────────────────────────


def test_index_command_computes_sha256(library_root: Path) -> None:
    """Assets indexed via the CLI should have a sha256 hash set."""
    main(["index", str(library_root)])
    conn = open_library_db(library_root)
    rows = list_assets(conn=conn, limit=1000)
    conn.close()
    # At least one asset should have a sha256
    sha256_values = [r.sha256 for r in rows if r.sha256 is not None]
    assert len(sha256_values) > 0


def test_index_command_sha256_is_valid_hex(library_root: Path) -> None:
    """SHA-256 values stored during indexing must be 64-character hex strings."""
    main(["index", str(library_root)])
    conn = open_library_db(library_root)
    rows = list_assets(conn=conn, limit=1000)
    conn.close()
    for row in rows:
        if row.sha256 is not None:
            assert len(row.sha256) == 64
            int(row.sha256, 16)  # raises ValueError if not valid hex


def test_index_command_identical_files_deduplicated(tmp_path: Path) -> None:
    """Two physically identical files (same bytes) produce a single assets row.

    The second path is stored as an alias in asset_paths.
    """
    from takeout_rater.db.queries import get_asset_alias_paths  # noqa: PLC0415

    takeout = tmp_path / "Takeout" / "Photos from 2024"
    takeout.mkdir(parents=True)
    content = b"\xff\xd8\xff" + b"\x00" * 100
    (takeout / "copy1.jpg").write_bytes(content)
    (takeout / "copy2.jpg").write_bytes(content)

    main(["index", str(tmp_path)])
    conn = open_library_db(tmp_path)
    rows = list_assets(conn=conn, limit=1000)
    # Only one assets row created; the duplicate is in asset_paths.
    assert len(rows) == 1
    assert rows[0].sha256 is not None
    canonical_id = rows[0].id
    aliases = get_asset_alias_paths(conn, canonical_id)
    conn.close()
    # Exactly one alias (whichever of copy1/copy2 was indexed second).
    assert len(aliases) == 1
    all_relpaths = {rows[0].relpath} | set(aliases)
    assert "Photos from 2024/copy1.jpg" in all_relpaths
    assert "Photos from 2024/copy2.jpg" in all_relpaths
