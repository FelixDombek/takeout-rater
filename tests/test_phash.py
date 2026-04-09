"""Tests for pHash computation (compute_dhash and compute_phash_all)."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from takeout_rater.db.queries import get_phash, list_asset_ids_without_phash, upsert_asset
from takeout_rater.db.schema import migrate
from takeout_rater.scoring.phash import compute_dhash, compute_phash_all, hamming_distance

# ── Helpers ───────────────────────────────────────────────────────────────────


def _open_in_memory() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    return conn


def _add_asset(conn: sqlite3.Connection, relpath: str = "Photos/img.jpg") -> int:
    return upsert_asset(
        conn,
        {
            "relpath": relpath,
            "filename": Path(relpath).name,
            "ext": Path(relpath).suffix.lower(),
            "size_bytes": 512,
            "mime": "image/jpeg",
            "indexed_at": int(time.time()),
        },
    )


def _make_thumbnail(
    thumbs_dir: Path, asset_id: int, color: tuple[int, int, int] = (100, 150, 200)
) -> Path:
    """Create a minimal JPEG thumbnail and return its path."""
    from PIL import Image  # noqa: PLC0415

    from takeout_rater.indexing.thumbnailer import thumb_path_for_id  # noqa: PLC0415

    thumb = thumb_path_for_id(thumbs_dir, asset_id)
    thumb.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=color).save(thumb, "JPEG")
    return thumb


# ── compute_dhash ─────────────────────────────────────────────────────────────


def test_compute_dhash_returns_hex_string(tmp_path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64), color=(100, 150, 200)).save(img_path, "JPEG")

    result = compute_dhash(img_path)
    assert isinstance(result, str)
    assert len(result) == 16  # 64-bit hash → 16 hex chars


def test_compute_dhash_hex_chars_only(tmp_path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64)).save(img_path, "JPEG")

    result = compute_dhash(img_path)
    int(result, 16)  # should not raise


def test_compute_dhash_identical_images_match(tmp_path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    img.save(tmp_path / "a.jpg", "JPEG")
    img.save(tmp_path / "b.jpg", "JPEG")

    h1 = compute_dhash(tmp_path / "a.jpg")
    h2 = compute_dhash(tmp_path / "b.jpg")
    assert h1 == h2


def test_compute_dhash_different_images_differ(tmp_path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    Image.new("RGB", (64, 64), color=(0, 0, 0)).save(tmp_path / "black.jpg", "JPEG")
    # White-to-black gradient so that left > right in some pixels
    img = Image.new("RGB", (64, 64))
    for x in range(64):
        for y in range(64):
            img.putpixel((x, y), (255 - x * 4, 255 - y * 4, 128))
    img.save(tmp_path / "gradient.jpg", "JPEG")

    h1 = compute_dhash(tmp_path / "black.jpg")
    h2 = compute_dhash(tmp_path / "gradient.jpg")
    assert h1 != h2


def test_compute_dhash_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        compute_dhash(tmp_path / "nonexistent.jpg")


# ── hamming_distance ──────────────────────────────────────────────────────────


def test_hamming_distance_identical() -> None:
    assert hamming_distance("deadbeef00000000", "deadbeef00000000") == 0


def test_hamming_distance_one_bit() -> None:
    # Differ in LSB only: 0 vs 1 → distance = 1
    assert hamming_distance("0000000000000000", "0000000000000001") == 1


def test_hamming_distance_all_bits() -> None:
    # All 64 bits flipped
    assert hamming_distance("0000000000000000", "ffffffffffffffff") == 64


def test_hamming_distance_symmetric() -> None:
    a, b = "abcdef1234567890", "fedcba0987654321"
    assert hamming_distance(a, b) == hamming_distance(b, a)


# ── compute_phash_all ─────────────────────────────────────────────────────────


def test_compute_phash_all_returns_count(tmp_path: Path) -> None:
    conn = _open_in_memory()
    thumbs_dir = tmp_path / "thumbs"
    ids = [_add_asset(conn, f"p/{i}.jpg") for i in range(3)]
    for aid in ids:
        _make_thumbnail(thumbs_dir, aid)

    count = compute_phash_all(conn, thumbs_dir)
    assert count == 3


def test_compute_phash_all_stores_hashes(tmp_path: Path) -> None:
    conn = _open_in_memory()
    thumbs_dir = tmp_path / "thumbs"
    asset_id = _add_asset(conn)
    _make_thumbnail(thumbs_dir, asset_id)

    compute_phash_all(conn, thumbs_dir)

    result = get_phash(conn, asset_id)
    assert result is not None
    assert len(result["phash_hex"]) == 16
    assert result["algo"] == "dhash"


def test_compute_phash_all_skips_missing_thumbnails(tmp_path: Path) -> None:
    conn = _open_in_memory()
    thumbs_dir = tmp_path / "thumbs"
    _add_asset(conn)  # no thumbnail created

    count = compute_phash_all(conn, thumbs_dir)
    assert count == 0


def test_compute_phash_all_skips_already_hashed(tmp_path: Path) -> None:
    conn = _open_in_memory()
    thumbs_dir = tmp_path / "thumbs"
    ids = [_add_asset(conn, f"p/{i}.jpg") for i in range(3)]
    for aid in ids:
        _make_thumbnail(thumbs_dir, aid)

    # First pass: hash all
    compute_phash_all(conn, thumbs_dir)
    # Second pass: all already hashed, should write 0 (default: only missing)
    remaining = list_asset_ids_without_phash(conn)
    count = compute_phash_all(conn, thumbs_dir)
    assert count == 0
    assert remaining == []


def test_compute_phash_all_progress_callback(tmp_path: Path) -> None:
    conn = _open_in_memory()
    thumbs_dir = tmp_path / "thumbs"
    for i in range(4):
        aid = _add_asset(conn, f"p/{i}.jpg")
        _make_thumbnail(thumbs_dir, aid, color=(i * 50, 100, 200))

    calls: list[tuple[int, int]] = []
    compute_phash_all(conn, thumbs_dir, batch_size=2, on_progress=lambda d, t: calls.append((d, t)))

    # on_progress is now fired once per item, so we expect exactly 4 calls.
    assert len(calls) == 4
    assert calls[-1] == (4, 4)
    # processed counter must be monotonically increasing.
    for i, (done, total) in enumerate(calls):
        assert done == i + 1
        assert total == 4


def test_compute_phash_all_on_item_callback(tmp_path: Path) -> None:
    conn = _open_in_memory()
    thumbs_dir = tmp_path / "thumbs"
    aids = [_add_asset(conn, f"p/{i}.jpg") for i in range(3)]
    for aid in aids:
        _make_thumbnail(thumbs_dir, aid)

    items: list[tuple[int, int, int]] = []
    compute_phash_all(
        conn,
        thumbs_dir,
        on_item=lambda aid, done, total: items.append((aid, done, total)),
    )

    assert len(items) == 3
    # asset IDs must appear in the callback
    assert {item[0] for item in items} == set(aids)
    # processed counter must be 1-based and monotonically increasing
    for i, (_aid, done, total) in enumerate(items):
        assert done == i + 1
        assert total == 3
