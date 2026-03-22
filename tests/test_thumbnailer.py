"""Tests for the thumbnail generator."""

from __future__ import annotations

from pathlib import Path

import pytest

from takeout_rater.indexing.thumbnailer import (
    THUMB_MAX_PX,
    generate_thumbnail,
    thumb_path_for_id,
)

FIXTURE_JPEG = (
    Path(__file__).parent
    / "fixtures"
    / "takeout_tree"
    / "Takeout"
    / "Photos from 2023"
    / "IMG_20230615_142301.jpg"
)
FIXTURE_PNG = (
    Path(__file__).parent
    / "fixtures"
    / "takeout_tree"
    / "Takeout"
    / "Photos from 2023"
    / "screenshot.png"
)


# ── thumb_path_for_id ─────────────────────────────────────────────────────────


def test_thumb_path_asset_0(tmp_path: Path) -> None:
    p = thumb_path_for_id(tmp_path, 0)
    assert p == tmp_path / "0000" / "0.jpg"


def test_thumb_path_asset_999(tmp_path: Path) -> None:
    p = thumb_path_for_id(tmp_path, 999)
    assert p == tmp_path / "0000" / "999.jpg"


def test_thumb_path_asset_1000(tmp_path: Path) -> None:
    p = thumb_path_for_id(tmp_path, 1000)
    assert p == tmp_path / "0001" / "1000.jpg"


def test_thumb_path_extension_is_jpg(tmp_path: Path) -> None:
    p = thumb_path_for_id(tmp_path, 42)
    assert p.suffix == ".jpg"


# ── generate_thumbnail ────────────────────────────────────────────────────────


def test_generate_thumbnail_creates_file(tmp_path: Path) -> None:
    out = tmp_path / "thumb.jpg"
    generate_thumbnail(FIXTURE_JPEG, out)
    assert out.exists()


def test_generate_thumbnail_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "deep" / "thumb.jpg"
    generate_thumbnail(FIXTURE_JPEG, out)
    assert out.exists()


def test_generate_thumbnail_produces_jpeg(tmp_path: Path) -> None:
    """Output file must start with JPEG magic bytes."""
    out = tmp_path / "thumb.jpg"
    generate_thumbnail(FIXTURE_JPEG, out)
    magic = out.read_bytes()[:3]
    assert magic == b"\xff\xd8\xff"


def test_generate_thumbnail_from_png(tmp_path: Path) -> None:
    """PNG input must be converted to JPEG output."""
    out = tmp_path / "from_png.jpg"
    generate_thumbnail(FIXTURE_PNG, out)
    assert out.exists()
    magic = out.read_bytes()[:3]
    assert magic == b"\xff\xd8\xff"


def test_generate_thumbnail_respects_max_dimension(tmp_path: Path) -> None:
    """Output dimensions must not exceed THUMB_MAX_PX in either axis."""
    pytest.importorskip("PIL")
    from PIL import Image  # noqa: PLC0415

    # Create a large image: 1024×768
    large_path = tmp_path / "large.jpg"
    Image.new("RGB", (1024, 768), color=(128, 64, 32)).save(large_path, "JPEG")

    out = tmp_path / "thumb.jpg"
    generate_thumbnail(large_path, out)

    with Image.open(out) as img:
        w, h = img.size
    assert w <= THUMB_MAX_PX
    assert h <= THUMB_MAX_PX


def test_generate_thumbnail_small_image_not_upscaled(tmp_path: Path) -> None:
    """An image smaller than THUMB_MAX_PX must not be upscaled."""
    pytest.importorskip("PIL")
    from PIL import Image  # noqa: PLC0415

    small_path = tmp_path / "small.jpg"
    Image.new("RGB", (100, 80), color=(200, 200, 200)).save(small_path, "JPEG")

    out = tmp_path / "thumb.jpg"
    generate_thumbnail(small_path, out)

    with Image.open(out) as img:
        w, h = img.size
    assert w <= 100
    assert h <= 80


def test_generate_thumbnail_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        generate_thumbnail(tmp_path / "nonexistent.jpg", tmp_path / "out.jpg")
