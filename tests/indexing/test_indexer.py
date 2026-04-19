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
    assert library_db_path(db_root).exists()
    # DB must NOT be inside photos_root
    assert not library_db_path(photos_root).exists()


def test_run_index_batches_clip_embeddings(
    tmp_path: Path, monkeypatch
) -> None:
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
    monkeypatch.setattr("takeout_rater.scoring.scorers.clip_backbone.get_clip_model", _fake_get_clip_model)

    run_index(photos_root, conn, db_root=db_root)

    assert conn.execute("SELECT COUNT(*) FROM clip_embeddings").fetchone()[0] == 8
    assert batch_sizes
    assert max(batch_sizes) > 1
    conn.close()
