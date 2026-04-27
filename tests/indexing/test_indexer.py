"""Integration tests for the index CLI command – fast error-path cases only.

Tests that exercise the full pipeline (scan → thumbnails → DB) live in
tests/slow/test_indexer_slow.py because thumbnail generation hangs in
environments without a display or GPU.
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
    photos_link = tmp_path / "photos"
    photos_link.symlink_to(FIXTURE_TAKEOUT.resolve(), target_is_directory=True)
    return photos_link


@pytest.fixture(autouse=True)
def disable_clip_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("takeout_rater.scoring.scorers.clip_backbone.is_available", lambda: False)


def _db_root_for(photos_root: Path) -> Path:
    return photos_root.parent / "state"


def _index_args(photos_root: Path) -> list[str]:
    return ["index", "--db-root", str(_db_root_for(photos_root)), str(photos_root)]


def _open_test_db(photos_root: Path):
    return open_library_db(_db_root_for(photos_root))


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
    from takeout_rater.db.connection import library_db_path

    photos = tmp_path / "photos"
    photos.mkdir()
    state = tmp_path / "state"
    state.mkdir()

    rc = main(["index", "--db-root", str(state), str(photos)])
    assert rc == 0
    # DB must be inside state/, not photos/
    assert library_db_path(state).exists()
    assert not library_db_path(photos).exists()


# ── index command: fixture pipeline ──────────────────────────────────────────


def test_index_command_creates_db(library_root: Path) -> None:
    main(_index_args(library_root))
    db_path = _db_root_for(library_root) / "takeout-rater" / "library.sqlite"
    assert db_path.exists()


def test_index_command_populates_assets(library_root: Path) -> None:
    main(_index_args(library_root))
    conn = _open_test_db(library_root)
    total = count_assets(conn)
    conn.close()
    assert total >= 2


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
    main(_index_args(library_root))
    main(_index_args(library_root))
    conn = _open_test_db(library_root)
    total = count_assets(conn)
    conn.close()
    assert total >= 2

    conn = _open_test_db(library_root)
    rows = list_assets(conn=conn, limit=1000)
    conn.close()
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
    takeout = tmp_path / "Takeout"
    google_photos = takeout / "Google Photos"
    album = google_photos / "Photos from 2023"
    album.mkdir(parents=True)
    (album / "img.jpg").write_bytes(b"\xff\xd8\xff")

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
    main(_index_args(library_root_google_photos_subdir))
    conn = _open_test_db(library_root_google_photos_subdir)
    row = get_asset_by_relpath(conn, "Photos from 2023/img.jpg")
    conn.close()
    assert row is not None


@pytest.fixture()
def library_root_google_fotos_subdir(tmp_path: Path) -> Path:
    google_fotos = tmp_path / "Takeout" / "Google Fotos"
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


# ── run_index function: separate db_root ─────────────────────────────────────


def test_run_index_separate_db_root(tmp_path: Path) -> None:
    """run_index with db_root different from photos_root stores DB in db_root."""
    from takeout_rater.db.connection import library_db_path
    from takeout_rater.indexing.run import run_index

    photos_root = tmp_path / "photos"
    photos_root.mkdir()
    db_root = tmp_path / "state"
    db_root.mkdir()

    conn = open_library_db(db_root)
    progress = run_index(photos_root, conn, db_root=db_root)
    conn.close()

    assert progress.done is True
    assert progress.error is None
    assert progress.diagnostics["assets_found"] == 0
    assert "scan_seconds" in progress.diagnostics
    assert library_db_path(db_root).exists()
    # DB must NOT be inside photos_root
    assert not library_db_path(photos_root).exists()


def test_run_index_returns_progress_with_indexed_count(library_root: Path) -> None:
    from takeout_rater.indexing.run import run_index

    conn = _open_test_db(library_root)
    progress = run_index(library_root, conn, db_root=_db_root_for(library_root))
    conn.close()

    assert progress.done is True
    assert progress.running is False
    assert progress.error is None
    assert progress.indexed >= 2
    assert progress.found == progress.indexed


def test_run_index_populates_db(library_root: Path) -> None:
    from takeout_rater.indexing.run import run_index

    conn = _open_test_db(library_root)
    run_index(library_root, conn, db_root=_db_root_for(library_root))
    total = count_assets(conn)
    conn.close()

    assert total >= 2


def test_run_index_is_idempotent(library_root: Path) -> None:
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


def test_run_index_final_progress_phase_is_processing(library_root: Path) -> None:
    from takeout_rater.indexing.run import run_index

    conn = _open_test_db(library_root)
    progress = run_index(library_root, conn, db_root=_db_root_for(library_root))
    conn.close()

    assert progress.phase == "processing"


def test_run_index_final_progress_dirs_scanned(library_root: Path) -> None:
    from takeout_rater.indexing.run import run_index

    conn = _open_test_db(library_root)
    progress = run_index(library_root, conn, db_root=_db_root_for(library_root))
    conn.close()

    assert progress.dirs_scanned > 0
    assert progress.dirs_scanned == progress.total_dirs


def test_run_index_on_progress_called_during_scanning(library_root: Path) -> None:
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


def test_index_command_computes_sha256(library_root: Path) -> None:
    main(_index_args(library_root))
    conn = _open_test_db(library_root)
    rows = list_assets(conn=conn, limit=1000)
    conn.close()
    sha256_values = [r.sha256 for r in rows if r.sha256 is not None]
    assert len(sha256_values) > 0


def test_index_command_sha256_is_valid_hex(library_root: Path) -> None:
    main(_index_args(library_root))
    conn = _open_test_db(library_root)
    rows = list_assets(conn=conn, limit=1000)
    conn.close()
    for row in rows:
        if row.sha256 is not None:
            assert len(row.sha256) == 64
            int(row.sha256, 16)


def test_stale_thumbnail_overwritten_after_db_reset(tmp_path: Path) -> None:
    from PIL import Image

    from takeout_rater.db.connection import library_state_dir
    from takeout_rater.indexing.thumbnailer import thumb_path_for_id

    photos_root = tmp_path / "photos"
    album = photos_root / "Album"
    album.mkdir(parents=True)
    db_root = tmp_path / "state"
    db_root.mkdir()

    Image.new("RGB", (64, 64), color=(255, 0, 0)).save(album / "photo_A.jpg", "JPEG")
    main(["index", "--db-root", str(db_root), str(photos_root)])

    thumbs_dir = library_state_dir(db_root) / "thumbs"
    thumb_1 = thumb_path_for_id(thumbs_dir, 1)
    assert thumb_1.exists(), "thumbnail for asset_id=1 should be created"
    with Image.open(thumb_1) as t:
        r, g, b = t.getpixel((0, 0))
    assert r > 200 and g < 50 and b < 50, "first thumbnail should be red"

    db_file = db_root / "takeout-rater" / "library.sqlite"
    db_file.unlink()
    (album / "photo_A.jpg").unlink()
    Image.new("RGB", (64, 64), color=(0, 0, 255)).save(album / "photo_B.jpg", "JPEG")

    main(["index", "--db-root", str(db_root), str(photos_root)])

    with Image.open(thumb_1) as t:
        r, g, b = t.getpixel((0, 0))
    assert b > 200 and r < 50 and g < 50, (
        "stale thumbnail was not overwritten: expected blue pixel after re-index with new DB"
    )


def test_index_command_identical_files_deduplicated(tmp_path: Path) -> None:
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
    assert len(rows) == 1
    assert rows[0].sha256 is not None
    canonical_id = rows[0].id
    aliases = get_asset_alias_paths(conn, canonical_id)
    conn.close()
    assert len(aliases) == 1
    all_relpaths = {rows[0].relpath} | set(aliases)
    assert "Photos from 2024/copy1.jpg" in all_relpaths
    assert "Photos from 2024/copy2.jpg" in all_relpaths


def test_run_index_batches_clip_embeddings(tmp_path: Path, monkeypatch) -> None:
    import torch
    from PIL import Image

    from takeout_rater.indexing.run import run_index

    photos_root = tmp_path / "photos"
    photos_root.mkdir()
    for idx in range(8):
        Image.new("RGB", (24, 24), color=(idx * 20, 40, 80)).save(
            photos_root / f"img{idx}.jpg",
            "JPEG",
        )

    db_root = tmp_path / "state"
    db_root.mkdir()
    conn = open_library_db(db_root)
    batch_sizes: list[int] = []

    class _FakeClipModel:
        def encode_image(self, batch):  # type: ignore[no-untyped-def]
            batch_sizes.append(int(batch.shape[0]))
            embeddings = torch.zeros(batch.shape[0], 768)
            embeddings[:, 0] = 1.0
            return embeddings

    def _fake_get_clip_model():  # type: ignore[no-untyped-def]
        return (
            _FakeClipModel(),
            lambda _img: torch.zeros(3, 224, 224),
            None,
            torch.device("cpu"),
        )

    monkeypatch.setattr("takeout_rater.scoring.scorers.clip_backbone.is_available", lambda: True)
    monkeypatch.setattr(
        "takeout_rater.scoring.scorers.clip_backbone.get_clip_model", _fake_get_clip_model
    )
    monkeypatch.setenv("TAKEOUT_RATER_CLIP_ACCELERATOR", "torch")

    progress = run_index(photos_root, conn, db_root=db_root)

    assert conn.execute("SELECT COUNT(*) FROM clip_embeddings").fetchone()[0] == 8
    assert batch_sizes
    assert max(batch_sizes) > 1
    assert progress.diagnostics["assets_found"] == 8
    assert progress.diagnostics["assets_indexed"] == 8
    assert progress.diagnostics["clip_embeddings_computed"] == 8
    assert progress.diagnostics["clip_batches"] >= 1
    assert progress.diagnostics["clip_preprocess_workers"] >= 1
    assert progress.diagnostics["clip_batch_last_size"] >= 1
    assert progress.diagnostics["clip_batch_inference_last_seconds"] >= 0
    assert progress.diagnostics["clip_batch_inference_max_seconds"] >= 0
    assert progress.diagnostics["clip_image_queue_max"] == 128
    assert progress.diagnostics["clip_tensor_queue_max"] >= 1
    assert "clip_inference_seconds" in progress.diagnostics
    conn.close()
