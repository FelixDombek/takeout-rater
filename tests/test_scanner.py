"""Tests for the Takeout directory scanner."""

from __future__ import annotations

from pathlib import Path

import pytest

from takeout_rater.indexing.scanner import (
    IMAGE_EXTENSIONS,
    SIDECAR_SUFFIX,
    AssetFile,
    scan_takeout,
)

TREE_FIXTURES = Path(__file__).parent / "fixtures" / "takeout_tree" / "Takeout"


# ── AssetFile data structure ───────────────────────────────────────────────────


def test_asset_file_is_frozen(tmp_path: Path) -> None:
    """AssetFile must be immutable (frozen dataclass)."""
    asset = AssetFile(
        relpath="test.jpg",
        abspath=tmp_path / "test.jpg",
        sidecar_path=None,
        mime="image/jpeg",
        size_bytes=1024,
    )
    with pytest.raises(AttributeError):
        asset.relpath = "other.jpg"  # type: ignore[misc]


# ── IMAGE_EXTENSIONS constant ─────────────────────────────────────────────────


def test_image_extensions_includes_jpg() -> None:
    assert ".jpg" in IMAGE_EXTENSIONS


def test_image_extensions_includes_png() -> None:
    assert ".png" in IMAGE_EXTENSIONS


def test_image_extensions_includes_heic() -> None:
    assert ".heic" in IMAGE_EXTENSIONS


# ── scan_takeout with fixture tree ────────────────────────────────────────────


def test_scan_fixture_tree_returns_list() -> None:
    result = scan_takeout(TREE_FIXTURES)
    assert isinstance(result, list)


def test_scan_fixture_tree_finds_images() -> None:
    result = scan_takeout(TREE_FIXTURES)
    assert len(result) >= 2


def test_scan_fixture_tree_all_asset_files() -> None:
    for asset in scan_takeout(TREE_FIXTURES):
        assert isinstance(asset, AssetFile)


def test_scan_fixture_tree_finds_sidecar_for_jpg() -> None:
    assets = scan_takeout(TREE_FIXTURES)
    jpg_asset = next((a for a in assets if a.relpath.endswith(".jpg")), None)
    assert jpg_asset is not None
    assert jpg_asset.sidecar_path is not None


def test_scan_fixture_tree_no_sidecar_for_png() -> None:
    """The PNG in the fixture tree has no sidecar."""
    assets = scan_takeout(TREE_FIXTURES)
    png_asset = next((a for a in assets if a.relpath.endswith(".png")), None)
    assert png_asset is not None
    assert png_asset.sidecar_path is None


def test_scan_fixture_tree_mime_jpeg() -> None:
    assets = scan_takeout(TREE_FIXTURES)
    jpg_asset = next((a for a in assets if a.relpath.endswith(".jpg")), None)
    assert jpg_asset is not None
    assert jpg_asset.mime == "image/jpeg"


def test_scan_fixture_tree_mime_png() -> None:
    assets = scan_takeout(TREE_FIXTURES)
    png_asset = next((a for a in assets if a.relpath.endswith(".png")), None)
    assert png_asset is not None
    assert png_asset.mime == "image/png"


def test_scan_fixture_tree_size_bytes_positive() -> None:
    for asset in scan_takeout(TREE_FIXTURES):
        assert asset.size_bytes > 0


def test_scan_fixture_tree_abspath_exists() -> None:
    for asset in scan_takeout(TREE_FIXTURES):
        assert asset.abspath.exists()


def test_scan_fixture_tree_no_sidecar_files_included() -> None:
    """Sidecar JSON files must never be returned as assets."""
    for asset in scan_takeout(TREE_FIXTURES):
        assert SIDECAR_SUFFIX not in asset.relpath


def test_scan_fixture_tree_sorted() -> None:
    """Results should be in sorted relpath order."""
    assets = scan_takeout(TREE_FIXTURES)
    relpaths = [a.relpath for a in assets]
    assert relpaths == sorted(relpaths)


# ── scan_takeout with temporary directory ─────────────────────────────────────


def test_scan_empty_dir_returns_empty(tmp_path: Path) -> None:
    assert scan_takeout(tmp_path) == []


def test_scan_dir_with_one_image(tmp_path: Path) -> None:
    (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8\xff")  # minimal JPEG header
    result = scan_takeout(tmp_path)
    assert len(result) == 1
    assert result[0].relpath == "photo.jpg"


def test_scan_dir_skips_non_image_files(tmp_path: Path) -> None:
    (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8\xff")
    (tmp_path / "document.pdf").write_bytes(b"%PDF")
    (tmp_path / "video.mp4").write_bytes(b"\x00")
    result = scan_takeout(tmp_path)
    assert len(result) == 1


def test_scan_dir_finds_nested_image(tmp_path: Path) -> None:
    sub = tmp_path / "Photos from 2023"
    sub.mkdir()
    (sub / "img.png").write_bytes(b"\x89PNG")
    result = scan_takeout(tmp_path)
    assert len(result) == 1
    assert result[0].relpath == "Photos from 2023/img.png"


def test_scan_dir_associates_sidecar_primary(tmp_path: Path) -> None:
    """Primary sidecar naming: img.jpg + img.jpg.supplemental-metadata.json"""
    (tmp_path / "img.jpg").write_bytes(b"\xff\xd8\xff")
    sidecar = tmp_path / "img.jpg.supplemental-metadata.json"
    sidecar.write_text("{}", encoding="utf-8")
    result = scan_takeout(tmp_path)
    assert result[0].sidecar_path == sidecar


def test_scan_dir_associates_sidecar_fallback(tmp_path: Path) -> None:
    """Fallback sidecar naming: img.jpg + img.supplemental-metadata.json"""
    (tmp_path / "img.jpg").write_bytes(b"\xff\xd8\xff")
    sidecar = tmp_path / "img.supplemental-metadata.json"
    sidecar.write_text("{}", encoding="utf-8")
    result = scan_takeout(tmp_path)
    assert result[0].sidecar_path == sidecar


def test_scan_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        scan_takeout(tmp_path / "nonexistent")
