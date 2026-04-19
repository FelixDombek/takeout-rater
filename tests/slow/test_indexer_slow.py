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
    """A temporary photos root pointing directly at the fixture albums."""
    # Symlink the fixture photos directory into the temp directory so scanning works
    # without copying large files.
    photos_link = tmp_path / "photos"
    photos_link.symlink_to(FIXTURE_TAKEOUT.resolve(), target_is_directory=True)
    return photos_link


def _db_root_for(photos_root: Path) -> Path:
    return photos_root.parent / "state"


def _index_args(photos_root: Path) -> list[str]:
    return ["index", "--db-root", str(_db_root_for(photos_root)), str(photos_root)]


def _open_test_db(photos_root: Path):
    return open_library_db(_db_root_for(photos_root))


# ── index command: basic operation ───────────────────────────────────────────


def test_index_command_returns_zero(library_root: Path) -> None:
    rc = main(_index_args(library_root))
    assert rc == 0


def test_index_command_creates_db(library_root: Path) -> None:
    main(_index_args(library_root))
    db_path = _db_root_for(library_root) / "takeout-rater" / "library.sqlite"
    assert db_path.exists()


def test_index_command_populates_assets(library_root: Path) -> None:
    main(_index_args(library_root))
    conn = _open_test_db(library_root)
    total = count_assets(conn)
    conn.close()
    assert total >= 2  # fixture tree has at least a JPEG and a PNG


def test_index_command_jpg_has_sidecar_data(library_root: Path) -> None:
    main(_index_args(library_root))
    conn = _open_test_db(library_root)
    row = get_asset_by_relpath(conn, "Photos from 2023/IMG_20230615_142301.jpg")
    conn.close()
    assert row is not None
    assert row.taken_at == 1686836381
    assert row.title == "IMG_20230615_142301.jpg"
    assert row.origin_type == "mobileUpload"
    assert row.geo_lat == pytest.approx(48.0)


def test_index_command_is_idempotent(library_root: Path) -> None:
    """Running index twice must not duplicate assets."""
    main(_index_args(library_root))
    main(_index_args(library_root))
    conn = _open_test_db(library_root)
    total = count_assets(conn)
    conn.close()
    # Should still be only the assets from the fixture tree
    assert total >= 2  # at least the fixture assets
    # Uniqueness is enforced by the DB; check no duplicate relpaths
    rows = list_assets(conn=_open_test_db(library_root), limit=1000)
    relpaths = [r.relpath for r in rows]
    assert len(relpaths) == len(set(relpaths))


def test_index_command_generates_thumbnails(library_root: Path) -> None:
    main(_index_args(library_root))
    thumbs_dir = _db_root_for(library_root) / "takeout-rater" / "thumbs"
    assert thumbs_dir.exists()
    thumb_files = list(thumbs_dir.rglob("*.jpg"))
    assert len(thumb_files) >= 1


# ── index command: Google Photos subdirectory layout ─────────────────────────


@pytest.fixture()
def library_root_google_photos_subdir(tmp_path: Path) -> Path:
    """Photos root named 'Google Photos'."""
    takeout = tmp_path / "Takeout"
    google_photos = takeout / "Google Photos"
    album = google_photos / "Photos from 2023"
    album.mkdir(parents=True)
    (album / "img.jpg").write_bytes(b"\xff\xd8\xff")

    # Unrelated Google product directory that must NOT be indexed
    other = takeout / "Google Drive"
    other.mkdir()
    (other / "drive_image.jpg").write_bytes(b"\xff\xd8\xff")

    return google_photos


def test_index_google_photos_subdir_returns_zero(
    library_root_google_photos_subdir: Path,
) -> None:
    rc = main(_index_args(library_root_google_photos_subdir))
    assert rc == 0


def test_index_google_photos_subdir_indexes_only_photos(
    library_root_google_photos_subdir: Path,
) -> None:
    """Only the selected photos root should be indexed."""
    main(_index_args(library_root_google_photos_subdir))
    conn = _open_test_db(library_root_google_photos_subdir)
    rows = list_assets(conn, limit=100)
    conn.close()

    relpaths = [r.relpath for r in rows]
    assert len(relpaths) == 1, f"expected 1 asset, got {len(relpaths)}: {relpaths}"
    assert relpaths[0] == "Photos from 2023/img.jpg"


def test_index_google_photos_subdir_relpath_excludes_google_photos_prefix(
    library_root_google_photos_subdir: Path,
) -> None:
    """relpaths must be relative to the selected photos root."""
    main(_index_args(library_root_google_photos_subdir))
    conn = _open_test_db(library_root_google_photos_subdir)
    row = get_asset_by_relpath(conn, "Photos from 2023/img.jpg")
    conn.close()
    assert row is not None


@pytest.fixture()
def library_root_google_fotos_subdir(tmp_path: Path) -> Path:
    """Photos root named 'Google Fotos'."""
    takeout = tmp_path / "Takeout"
    google_fotos = takeout / "Google Fotos"
    album = google_fotos / "Fotos aus 2023"
    album.mkdir(parents=True)
    (album / "bild.jpg").write_bytes(b"\xff\xd8\xff")
    return google_fotos


def test_index_google_fotos_subdir_returns_zero(
    library_root_google_fotos_subdir: Path,
) -> None:
    rc = main(_index_args(library_root_google_fotos_subdir))
    assert rc == 0


def test_index_google_fotos_subdir_indexes_photo(
    library_root_google_fotos_subdir: Path,
) -> None:
    main(_index_args(library_root_google_fotos_subdir))
    conn = _open_test_db(library_root_google_fotos_subdir)
    rows = list_assets(conn, limit=100)
    conn.close()
    assert len(rows) == 1
    assert rows[0].relpath == "Fotos aus 2023/bild.jpg"


# ── run_index function (background-callable API) ──────────────────────────────


def test_run_index_returns_progress_with_indexed_count(library_root: Path) -> None:
    """run_index must upsert assets and return the correct count."""
    from takeout_rater.indexing.run import run_index

    conn = _open_test_db(library_root)
    progress = run_index(library_root, conn, db_root=_db_root_for(library_root))
    conn.close()

    assert progress.done is True
    assert progress.running is False
    assert progress.error is None
    assert progress.indexed >= 2  # fixture has at least a JPEG and a PNG
    assert progress.found == progress.indexed


def test_run_index_populates_db(library_root: Path) -> None:
    """After run_index the DB must contain the indexed assets."""
    from takeout_rater.indexing.run import run_index

    conn = _open_test_db(library_root)
    run_index(library_root, conn, db_root=_db_root_for(library_root))
    total = count_assets(conn)
    conn.close()

    assert total >= 2


def test_run_index_is_idempotent(library_root: Path) -> None:
    """Calling run_index twice must not duplicate assets."""
    from takeout_rater.indexing.run import run_index

    conn = _open_test_db(library_root)
    run_index(library_root, conn, db_root=_db_root_for(library_root))
    run_index(library_root, conn, db_root=_db_root_for(library_root))
    total = count_assets(conn)
    conn.close()

    conn2 = _open_test_db(library_root)
    rows = list_assets(conn=conn2, limit=1000)
    conn2.close()
    relpaths = [r.relpath for r in rows]
    assert len(relpaths) == len(set(relpaths))
    assert total == len(relpaths)


# ── run_index progress fields ─────────────────────────────────────────────────


def test_run_index_final_progress_phase_is_processing(library_root: Path) -> None:
    """The final IndexProgress must have phase='processing' (not 'scanning')."""
    from takeout_rater.indexing.run import run_index

    conn = _open_test_db(library_root)
    progress = run_index(library_root, conn, db_root=_db_root_for(library_root))
    conn.close()

    assert progress.phase == "processing"


def test_run_index_final_progress_dirs_scanned(library_root: Path) -> None:
    """dirs_scanned must be positive after a successful run."""
    from takeout_rater.indexing.run import run_index

    conn = _open_test_db(library_root)
    progress = run_index(library_root, conn, db_root=_db_root_for(library_root))
    conn.close()

    assert progress.dirs_scanned > 0
    assert progress.dirs_scanned == progress.total_dirs


def test_run_index_on_progress_called_during_scanning(library_root: Path) -> None:
    """on_progress must be invoked while scanning (phase='scanning')."""
    from takeout_rater.indexing.run import run_index

    scanning_calls: list[object] = []

    def _cb(p: object) -> None:
        from takeout_rater.indexing.run import IndexProgress

        assert isinstance(p, IndexProgress)
        if p.phase == "scanning":
            scanning_calls.append(p)

    conn = _open_test_db(library_root)
    run_index(library_root, conn, db_root=_db_root_for(library_root), on_progress=_cb)
    conn.close()

    assert len(scanning_calls) > 0, "on_progress was never called with phase='scanning'"


def test_run_index_on_progress_called_during_processing(library_root: Path) -> None:
    """on_progress must be invoked while processing (phase='processing')."""
    from takeout_rater.indexing.run import run_index

    processing_calls: list[object] = []

    def _cb(p: object) -> None:
        from takeout_rater.indexing.run import IndexProgress

        if isinstance(p, IndexProgress) and p.phase == "processing" and not p.done:
            processing_calls.append(p)

    conn = _open_test_db(library_root)
    run_index(library_root, conn, db_root=_db_root_for(library_root), on_progress=_cb)
    conn.close()

    assert len(processing_calls) > 0, "on_progress was never called with phase='processing'"


# ── SHA-256 computation during indexing ──────────────────────────────────────


def test_index_command_computes_sha256(library_root: Path) -> None:
    """Assets indexed via the CLI should have a sha256 hash set."""
    main(_index_args(library_root))
    conn = _open_test_db(library_root)
    rows = list_assets(conn=conn, limit=1000)
    conn.close()
    # At least one asset should have a sha256
    sha256_values = [r.sha256 for r in rows if r.sha256 is not None]
    assert len(sha256_values) > 0


def test_index_command_sha256_is_valid_hex(library_root: Path) -> None:
    """SHA-256 values stored during indexing must be 64-character hex strings."""
    main(_index_args(library_root))
    conn = _open_test_db(library_root)
    rows = list_assets(conn=conn, limit=1000)
    conn.close()
    for row in rows:
        if row.sha256 is not None:
            assert len(row.sha256) == 64
            int(row.sha256, 16)  # raises ValueError if not valid hex


def test_stale_thumbnail_overwritten_after_db_reset(tmp_path: Path) -> None:
    """Stale thumbnails left over from a deleted DB must be overwritten.

    Regression test for: thumbnails appearing wrong after creating a fresh
    database without clearing the thumbs directory.  When auto-increment IDs
    restart from 1, a new asset may be assigned the same ID that a different
    photo held in the previous database.  If the old thumbnail file still
    exists on disk the indexer must not reuse it — it must overwrite it with
    a thumbnail generated from the actual current asset.
    """
    from PIL import Image

    from takeout_rater.db.connection import library_state_dir
    from takeout_rater.indexing.thumbnailer import thumb_path_for_id

    # ── First database: index "photo_A" so asset_id=1 maps to a red image ─────
    photos_root = tmp_path / "photos"
    album = photos_root / "Album"
    album.mkdir(parents=True)
    db_root = tmp_path / "state"
    db_root.mkdir()

    red_img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    red_img.save(album / "photo_A.jpg", "JPEG")

    main(["index", "--db-root", str(db_root), str(photos_root)])

    # Confirm the thumbnail for asset_id=1 exists and looks red.
    thumbs_dir = library_state_dir(db_root) / "thumbs"
    thumb_1 = thumb_path_for_id(thumbs_dir, 1)
    assert thumb_1.exists(), "thumbnail for asset_id=1 should be created"
    with Image.open(thumb_1) as t:
        r, g, b = t.getpixel((0, 0))
    assert r > 200 and g < 50 and b < 50, "first thumbnail should be red"

    # ── Simulate a DB reset: delete the database but keep the thumbs dir ──────
    db_file = db_root / "takeout-rater" / "library.sqlite"
    db_file.unlink()

    # Replace photo_A with a completely different (blue) photo.
    (album / "photo_A.jpg").unlink()
    blue_img = Image.new("RGB", (64, 64), color=(0, 0, 255))
    blue_img.save(album / "photo_B.jpg", "JPEG")

    # ── Second database: re-index; photo_B gets asset_id=1 again ─────────────
    main(["index", "--db-root", str(db_root), str(photos_root)])

    # The thumbnail at asset_id=1 must now reflect photo_B (blue), not the
    # stale red thumbnail from the old database.
    with Image.open(thumb_1) as t:
        r, g, b = t.getpixel((0, 0))
    assert b > 200 and r < 50 and g < 50, (
        "stale thumbnail was not overwritten: expected blue pixel after re-index with new DB"
    )


def test_index_command_identical_files_deduplicated(tmp_path: Path) -> None:
    """Two physically identical files (same bytes) produce a single assets row.

    The second path is stored as an alias in asset_paths.
    """
    from takeout_rater.db.queries import get_asset_alias_paths

    photos_root = tmp_path / "photos"
    photos = photos_root / "Photos from 2024"
    photos.mkdir(parents=True)
    content = b"\xff\xd8\xff" + b"\x00" * 100
    (photos / "copy1.jpg").write_bytes(content)
    (photos / "copy2.jpg").write_bytes(content)

    db_root = tmp_path / "state"
    main(["index", "--db-root", str(db_root), str(photos_root)])
    conn = open_library_db(db_root)
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
