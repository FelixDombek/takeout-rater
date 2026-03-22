"""Parse ``*.supplemental-metadata.json`` sidecar files from Google Photos Takeout."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Keys that may appear under googlePhotosOrigin (in priority order)
_ORIGIN_KEYS = (
    "mobileUpload",
    "driveSync",
    "fromPartnerSharing",
    "fromSharedAlbum",
    "webUpload",
    "composition",
)


@dataclass(frozen=True)
class SidecarData:
    """Parsed fields from a ``*.supplemental-metadata.json`` sidecar file.

    All fields correspond directly to columns in the ``assets`` table.
    Optional fields are ``None`` when absent or not applicable.
    """

    title: str
    description: str
    google_photos_url: str
    taken_at: int | None  # Unix timestamp from photoTakenTime.timestamp
    created_at_sidecar: int | None  # Unix timestamp from creationTime.timestamp
    image_views: int | None  # imageViews (sidecar encodes as string)
    # geoData — always present in sidecar; 0.0/0.0 when no location data
    geo_lat: float | None
    geo_lon: float | None
    geo_alt: float | None
    # geoDataExif — optional; present in ~51 % of sidecars
    geo_exif_lat: float | None
    geo_exif_lon: float | None
    geo_exif_alt: float | None
    # Boolean flags (optional in sidecar)
    favorited: bool | None
    archived: bool | None
    trashed: bool | None
    # Upload / origin info (from googlePhotosOrigin, optional)
    origin_type: str | None
    origin_device_type: str | None
    origin_device_folder: str | None
    # App source (optional)
    app_source_package: str | None


def _parse_geo(geo: dict) -> tuple[float, float, float]:
    """Extract latitude, longitude, and altitude from a geoData dict."""
    return (
        float(geo.get("latitude", 0.0)),
        float(geo.get("longitude", 0.0)),
        float(geo.get("altitude", 0.0)),
    )


def _parse_origin(origin: dict) -> tuple[str | None, str | None, str | None]:
    """Return ``(origin_type, device_type, device_folder)`` from a googlePhotosOrigin dict."""
    for key in _ORIGIN_KEYS:
        if key in origin:
            origin_data = origin[key]
            if isinstance(origin_data, dict):
                device_type: str | None = origin_data.get("deviceType")
                folder_data = origin_data.get("deviceFolder", {})
                device_folder: str | None = (
                    folder_data.get("localFolderName") if isinstance(folder_data, dict) else None
                )
            else:
                device_type = None
                device_folder = None
            return key, device_type, device_folder
    return None, None, None


def parse_sidecar(path: Path) -> SidecarData:
    """Parse a ``*.supplemental-metadata.json`` sidecar file.

    Args:
        path: Absolute (or relative) path to the sidecar JSON file.

    Returns:
        A :class:`SidecarData` with all available fields populated.

    Raises:
        ValueError: If the file cannot be parsed as JSON or is unreadable.
    """
    try:
        raw: dict = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot parse sidecar {path}: {exc}") from exc

    # Required top-level fields (default to empty string if missing)
    title: str = raw.get("title", "")
    description: str = raw.get("description", "")
    url: str = raw.get("url", "")

    # Timestamps (optional)
    photo_taken = raw.get("photoTakenTime", {})
    taken_at: int | None = int(photo_taken["timestamp"]) if "timestamp" in photo_taken else None

    creation = raw.get("creationTime", {})
    created_at_sidecar: int | None = int(creation["timestamp"]) if "timestamp" in creation else None

    # imageViews is encoded as a string in the sidecar
    image_views_raw = raw.get("imageViews")
    image_views: int | None = int(image_views_raw) if image_views_raw is not None else None

    # geoData — always present; 0.0/0.0 when no location
    geo = raw.get("geoData")
    if geo is not None:
        geo_lat, geo_lon, geo_alt = _parse_geo(geo)
    else:
        geo_lat = geo_lon = geo_alt = None

    # geoDataExif — optional
    geo_exif = raw.get("geoDataExif")
    if geo_exif is not None:
        geo_exif_lat, geo_exif_lon, geo_exif_alt = _parse_geo(geo_exif)
    else:
        geo_exif_lat = geo_exif_lon = geo_exif_alt = None

    # Boolean flags
    favorited: bool | None = bool(raw["favorited"]) if "favorited" in raw else None
    archived: bool | None = bool(raw["archived"]) if "archived" in raw else None
    trashed: bool | None = bool(raw["trashed"]) if "trashed" in raw else None

    # Origin info
    origin = raw.get("googlePhotosOrigin", {})
    origin_type, origin_device_type, origin_device_folder = _parse_origin(origin)

    # App source
    app_source = raw.get("appSource", {})
    app_source_package: str | None = (
        app_source.get("androidPackageName") if isinstance(app_source, dict) else None
    )

    return SidecarData(
        title=title,
        description=description,
        google_photos_url=url,
        taken_at=taken_at,
        created_at_sidecar=created_at_sidecar,
        image_views=image_views,
        geo_lat=geo_lat,
        geo_lon=geo_lon,
        geo_alt=geo_alt,
        geo_exif_lat=geo_exif_lat,
        geo_exif_lon=geo_exif_lon,
        geo_exif_alt=geo_exif_alt,
        favorited=favorited,
        archived=archived,
        trashed=trashed,
        origin_type=origin_type,
        origin_device_type=origin_device_type,
        origin_device_folder=origin_device_folder,
        app_source_package=app_source_package,
    )
