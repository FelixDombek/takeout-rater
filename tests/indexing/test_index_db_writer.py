"""Robustness of the importer's batched DB writer."""

from __future__ import annotations

from pathlib import Path

import pytest

from takeout_rater.db.connection import open_library_db


@pytest.fixture(autouse=True)
def disable_clip_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("takeout_rater.scoring.scorers.clip_backbone.is_available", lambda: False)


def _make_photos(root: Path, n: int) -> None:
    from PIL import Image

    album = root / "Album"
    album.mkdir(parents=True)
    for i in range(n):
        # Distinct content so every file gets its own asset row.
        Image.new("RGB", (40 + i, 30), (i * 7 % 256, 0, 0)).save(album / f"img{i}.jpg", "JPEG")


def test_transaction_aborting_phash_error_keeps_claims_consistent(tmp_path: Path) -> None:
    """An error that makes SQLite abort the whole batch transaction (as SQLITE_FULL
    or SQLITE_IOERR would) must not leave claims reported as saved but missing.

    ``RAISE(ROLLBACK)`` in a trigger rolls back the entire transaction, like
    those errors do.  Every asset must still end up in the database.
    """
    from takeout_rater.indexing.run import run_index

    photos = tmp_path / "photos"
    _make_photos(photos, 12)
    db_root = tmp_path / "state"
    conn = open_library_db(db_root)
    conn.execute(
        "CREATE TRIGGER phash_abort BEFORE INSERT ON phash"
        " BEGIN SELECT RAISE(ROLLBACK, 'simulated disk full'); END"
    )
    conn.commit()

    progress = run_index(photos, conn, db_root=db_root)

    assert progress.done is True
    assert progress.indexed == 12
    ids = [r[0] for r in conn.execute("SELECT id FROM assets ORDER BY id")]
    assert len(ids) == 12
    assert conn.execute("SELECT COUNT(*) FROM phash").fetchone()[0] == 0
    # Thumbnails were written for the ids the claims returned, so those ids
    # must be the committed rows.
    thumbs = sorted(int(p.stem) for p in (db_root / "takeout-rater" / "thumbs").rglob("*.jpg"))
    assert thumbs == ids
    conn.close()


def test_final_progress_callback_has_final_counters(tmp_path: Path) -> None:
    from takeout_rater.indexing.run import run_index

    photos = tmp_path / "photos"
    _make_photos(photos, 5)
    db_root = tmp_path / "state"
    conn = open_library_db(db_root)
    seen: list[dict] = []
    run_index(photos, conn, db_root=db_root, on_progress=lambda p: seen.append(dict(p.diagnostics)))
    conn.close()

    assert seen
    last = seen[-1]
    assert last["assets_indexed"] == 5
    assert last["thumbnails_generated"] == 5
    assert last["phashes_computed"] == 5
