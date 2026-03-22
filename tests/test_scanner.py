"""Tests for the Takeout directory scanner."""

from __future__ import annotations

from pathlib import Path

import pytest

from takeout_rater.indexing.scanner import (
    GOOGLE_PHOTOS_DIR_NAMES,
    IMAGE_EXTENSIONS,
    SIDECAR_SUFFIX,
    AssetFile,
    find_google_photos_root,
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


# ── find_google_photos_root ───────────────────────────────────────────────────


def test_find_google_photos_root_returns_dir_unchanged_when_no_subdir(
    tmp_path: Path,
) -> None:
    """When no known localized subdir exists, the input dir is returned as-is."""
    (tmp_path / "Photos from 2023").mkdir()
    assert find_google_photos_root(tmp_path) == tmp_path


def test_find_google_photos_root_finds_google_photos_subdir(tmp_path: Path) -> None:
    """English 'Google Photos' subdirectory is detected."""
    subdir = tmp_path / "Google Photos"
    subdir.mkdir()
    assert find_google_photos_root(tmp_path) == subdir


def test_find_google_photos_root_finds_google_fotos_subdir(tmp_path: Path) -> None:
    """German/Spanish/Portuguese 'Google Fotos' subdirectory is detected."""
    subdir = tmp_path / "Google Fotos"
    subdir.mkdir()
    assert find_google_photos_root(tmp_path) == subdir


def test_find_google_photos_root_finds_google_foto_subdir(tmp_path: Path) -> None:
    """Italian 'Google Foto' subdirectory is detected."""
    subdir = tmp_path / "Google Foto"
    subdir.mkdir()
    assert find_google_photos_root(tmp_path) == subdir


def test_find_google_photos_root_empty_dir(tmp_path: Path) -> None:
    """An empty directory returns itself (no crash)."""
    assert find_google_photos_root(tmp_path) == tmp_path


def test_find_google_photos_root_ignores_file_with_same_name(tmp_path: Path) -> None:
    """A *file* named 'Google Photos' is not treated as the photos root."""
    (tmp_path / "Google Photos").write_text("not a dir", encoding="utf-8")
    assert find_google_photos_root(tmp_path) == tmp_path


def test_google_photos_dir_names_constant_is_non_empty() -> None:
    assert len(GOOGLE_PHOTOS_DIR_NAMES) > 0


def test_scan_skips_non_photos_sibling_when_google_photos_subdir_present(
    tmp_path: Path,
) -> None:
    """Images outside the Google Photos subdir are not returned when that subdir exists."""
    google_photos = tmp_path / "Google Photos"
    google_photos.mkdir()
    (google_photos / "album").mkdir()
    (google_photos / "album" / "photo.jpg").write_bytes(b"\xff\xd8\xff")

    # A sibling directory simulating another Google product (e.g. Drive)
    drive = tmp_path / "Google Drive"
    drive.mkdir()
    (drive / "file.jpg").write_bytes(b"\xff\xd8\xff")

    photos_root = find_google_photos_root(tmp_path)
    result = scan_takeout(photos_root)

    relpaths = [a.relpath for a in result]
    assert any("photo.jpg" in r for r in relpaths), "expected photo from Google Photos"
    assert not any("Google Drive" in r for r in relpaths), "Drive image must be excluded"
