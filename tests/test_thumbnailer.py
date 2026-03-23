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


# ── EXIF orientation ──────────────────────────────────────────────────────────


def _make_jpeg_with_exif_orientation(path: Path, width: int, height: int, orientation: int) -> None:
    """Save a plain-colour JPEG at *path* with the given EXIF orientation tag."""
    pytest.importorskip("PIL")
    from PIL import Image  # noqa: PLC0415

    img = Image.new("RGB", (width, height), color=(200, 100, 50))
    exif = img.getexif()
    exif[0x0112] = orientation  # 0x0112 = EXIF Orientation tag
    img.save(path, "JPEG", exif=exif.tobytes())


@pytest.mark.parametrize(
    "orientation, w, h, expected_w, expected_h",
    [
        # orientation 1 – no rotation; pixel dimensions unchanged
        (1, 100, 200, 100, 200),
        # orientation 6 – 90° CW rotation; a tall image becomes wide
        (6, 100, 200, 200, 100),
        # orientation 8 – 90° CCW rotation; a wide image becomes tall
        (8, 200, 100, 100, 200),
        # orientation 3 – 180° rotation; dimensions unchanged
        (3, 100, 200, 100, 200),
        # orientation 2 – horizontal flip; dimensions unchanged
        (2, 100, 200, 100, 200),
        # orientation 4 – vertical flip; dimensions unchanged
        (4, 100, 200, 100, 200),
        # orientation 5 – transpose (flip + 90° CCW); wide becomes tall
        (5, 200, 100, 100, 200),
        # orientation 7 – transverse (flip + 90° CW); tall becomes wide
        (7, 100, 200, 200, 100),
    ],
)
def test_generate_thumbnail_exif_orientation(
    tmp_path: Path,
    orientation: int,
    w: int,
    h: int,
    expected_w: int,
    expected_h: int,
) -> None:
    """Thumbnail dimensions must reflect the EXIF orientation, not raw pixel order."""
    pytest.importorskip("PIL")
    from PIL import Image  # noqa: PLC0415

    src = tmp_path / f"orient_{orientation}.jpg"
    _make_jpeg_with_exif_orientation(src, w, h, orientation)

    out = tmp_path / f"thumb_{orientation}.jpg"
    generate_thumbnail(src, out)

    with Image.open(out) as thumb:
        tw, th = thumb.size

    # The thumbnail must have the correct aspect ratio after orientation is applied.
    # Both dimensions are within THUMB_MAX_PX and match expected_w × expected_h ratio.
    assert tw <= THUMB_MAX_PX
    assert th <= THUMB_MAX_PX
    if expected_w != expected_h:
        # Non-square: confirm the orientation of the thumbnail (landscape vs. portrait)
        assert (tw > th) == (expected_w > expected_h), (
            f"orientation={orientation}: expected {'landscape' if expected_w > expected_h else 'portrait'}, "
            f"got {tw}×{th}"
        )
