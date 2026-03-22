"""Tests for the sidecar JSON parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from takeout_rater.indexing.sidecar import SidecarData, parse_sidecar

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "takeout_sidecars"


# ── parse_sidecar happy paths ──────────────────────────────────────────────────


def test_parse_full_sidecar_title() -> None:
    result = parse_sidecar(FIXTURES_DIR / "photo_full.supplemental-metadata.json")
    assert result.title == "IMG_20230615_142301.jpg"


def test_parse_full_sidecar_description() -> None:
    result = parse_sidecar(FIXTURES_DIR / "photo_full.supplemental-metadata.json")
    assert result.description == ""


def test_parse_full_sidecar_taken_at() -> None:
    result = parse_sidecar(FIXTURES_DIR / "photo_full.supplemental-metadata.json")
    assert result.taken_at == 1686836381


def test_parse_full_sidecar_created_at_sidecar() -> None:
    result = parse_sidecar(FIXTURES_DIR / "photo_full.supplemental-metadata.json")
    assert result.created_at_sidecar == 1686836581


def test_parse_full_sidecar_image_views() -> None:
    result = parse_sidecar(FIXTURES_DIR / "photo_full.supplemental-metadata.json")
    assert result.image_views == 42


def test_parse_full_sidecar_geo_data() -> None:
    result = parse_sidecar(FIXTURES_DIR / "photo_full.supplemental-metadata.json")
    assert result.geo_lat == pytest.approx(48.0)
    assert result.geo_lon == pytest.approx(11.0)
    assert result.geo_alt == pytest.approx(520.0)


def test_parse_full_sidecar_geo_exif() -> None:
    result = parse_sidecar(FIXTURES_DIR / "photo_full.supplemental-metadata.json")
    assert result.geo_exif_lat == pytest.approx(48.0)
    assert result.geo_exif_lon == pytest.approx(11.0)
    assert result.geo_exif_alt == pytest.approx(520.0)


def test_parse_full_sidecar_url() -> None:
    result = parse_sidecar(FIXTURES_DIR / "photo_full.supplemental-metadata.json")
    assert result.google_photos_url == "https://example.invalid/photo/EXAMPLE"


def test_parse_full_sidecar_origin_type() -> None:
    result = parse_sidecar(FIXTURES_DIR / "photo_full.supplemental-metadata.json")
    assert result.origin_type == "mobileUpload"
    assert result.origin_device_type == "ANDROID_PHONE"


def test_parse_full_sidecar_returns_sidecar_data() -> None:
    result = parse_sidecar(FIXTURES_DIR / "photo_full.supplemental-metadata.json")
    assert isinstance(result, SidecarData)


# ── parse_sidecar with missing optional fields ─────────────────────────────────


def test_parse_no_geo_sidecar_geo_is_none() -> None:
    result = parse_sidecar(FIXTURES_DIR / "photo_no_geo.supplemental-metadata.json")
    assert result.geo_lat is None
    assert result.geo_lon is None
    assert result.geo_alt is None


def test_parse_no_geo_sidecar_geo_exif_is_none() -> None:
    result = parse_sidecar(FIXTURES_DIR / "photo_no_geo.supplemental-metadata.json")
    assert result.geo_exif_lat is None
    assert result.geo_exif_lon is None
    assert result.geo_exif_alt is None


def test_parse_no_photo_taken_time() -> None:
    result = parse_sidecar(FIXTURES_DIR / "video_no_photo_taken_time.supplemental-metadata.json")
    assert result.taken_at is None


def test_parse_no_photo_taken_time_has_created_at() -> None:
    result = parse_sidecar(FIXTURES_DIR / "video_no_photo_taken_time.supplemental-metadata.json")
    assert result.created_at_sidecar == 1646136000


def test_parse_web_upload_origin() -> None:
    result = parse_sidecar(FIXTURES_DIR / "video_no_photo_taken_time.supplemental-metadata.json")
    assert result.origin_type == "webUpload"
    assert result.origin_device_type is None  # webUpload has no deviceType


def test_parse_ios_origin() -> None:
    result = parse_sidecar(FIXTURES_DIR / "photo_no_geo.supplemental-metadata.json")
    assert result.origin_type == "mobileUpload"
    assert result.origin_device_type == "IOS_PHONE"


def test_parse_boolean_flags_absent() -> None:
    """favorited/archived/trashed absent in all current fixtures → should be None."""
    result = parse_sidecar(FIXTURES_DIR / "photo_full.supplemental-metadata.json")
    assert result.favorited is None
    assert result.archived is None
    assert result.trashed is None


def test_parse_app_source_absent() -> None:
    result = parse_sidecar(FIXTURES_DIR / "photo_full.supplemental-metadata.json")
    assert result.app_source_package is None


# ── parse_sidecar with inline content via tmp_path ───────────────────────────


def test_parse_sidecar_with_favorited_flag(tmp_path: Path) -> None:
    sidecar = tmp_path / "img.jpg.supplemental-metadata.json"
    sidecar.write_text(
        '{"title":"t","description":"","url":"u","creationTime":{"timestamp":"100"},'
        '"imageViews":"0","favorited":true}',
        encoding="utf-8",
    )
    result = parse_sidecar(sidecar)
    assert result.favorited is True


def test_parse_sidecar_with_archived_flag(tmp_path: Path) -> None:
    sidecar = tmp_path / "img.jpg.supplemental-metadata.json"
    sidecar.write_text(
        '{"title":"t","description":"","url":"u","creationTime":{"timestamp":"100"},'
        '"imageViews":"0","archived":true}',
        encoding="utf-8",
    )
    result = parse_sidecar(sidecar)
    assert result.archived is True


def test_parse_sidecar_with_app_source(tmp_path: Path) -> None:
    sidecar = tmp_path / "img.jpg.supplemental-metadata.json"
    sidecar.write_text(
        '{"title":"t","description":"","url":"u","creationTime":{"timestamp":"100"},'
        '"imageViews":"0","appSource":{"androidPackageName":"com.example.app"}}',
        encoding="utf-8",
    )
    result = parse_sidecar(sidecar)
    assert result.app_source_package == "com.example.app"


def test_parse_sidecar_with_partner_sharing(tmp_path: Path) -> None:
    sidecar = tmp_path / "img.jpg.supplemental-metadata.json"
    sidecar.write_text(
        '{"title":"t","description":"","url":"u","creationTime":{"timestamp":"100"},'
        '"imageViews":"0","googlePhotosOrigin":{"fromPartnerSharing":{}}}',
        encoding="utf-8",
    )
    result = parse_sidecar(sidecar)
    assert result.origin_type == "fromPartnerSharing"


# ── parse_sidecar error cases ─────────────────────────────────────────────────


def test_parse_sidecar_missing_file_raises_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Cannot parse sidecar"):
        parse_sidecar(tmp_path / "nonexistent.json")


def test_parse_sidecar_invalid_json_raises_value_error(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not json!", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot parse sidecar"):
        parse_sidecar(bad_file)
