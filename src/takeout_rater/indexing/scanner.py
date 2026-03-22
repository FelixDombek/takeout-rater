"""Walk a Google Photos Takeout directory tree and enumerate image assets."""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path

# Image extensions that the indexer will recognise
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".gif", ".bmp", ".tiff", ".tif"}
)

# Sidecar suffix used by Google Photos Takeout
SIDECAR_SUFFIX = ".supplemental-metadata.json"

# Known localized names for the Google Photos subdirectory inside a Takeout export.
# Google Takeout sometimes nests all photo albums under one of these subdirectories
# instead of placing them directly inside ``Takeout/``.  Scanning only this
# subdirectory avoids accidentally indexing images from other Google products
# (Drive, Chat, etc.) that may also be present in the same Takeout archive.
GOOGLE_PHOTOS_DIR_NAMES: tuple[str, ...] = (
    "Google Photos",  # English
    "Google Fotos",  # German, Spanish, Portuguese
    "Google Foto",  # Italian
)


def find_google_photos_root(takeout_dir: Path) -> Path:
    """Return the directory that contains the Google Photos album folders.

    A Google Takeout export may place all photo albums directly inside the
    ``Takeout/`` directory (old format with ``Photos from YYYY/`` subdirs), or
    it may nest them inside a localized subdirectory such as ``Google Photos/``
    (English) or ``Google Fotos/`` (German).  Scanning the entire ``Takeout/``
    tree would incorrectly include images from other Google products like Drive
    or Chat.  This function finds the narrowest root that covers only Google
    Photos content.

    Args:
        takeout_dir: The ``Takeout/`` directory (or the top-level scan root).

    Returns:
        The subdirectory to pass to :func:`scan_takeout`.  Returns a known
        localized Google Photos subdirectory when one exists, otherwise returns
        *takeout_dir* unchanged (for old-format exports).
    """
    for name in GOOGLE_PHOTOS_DIR_NAMES:
        candidate = takeout_dir / name
        if candidate.is_dir():
            return candidate
    return takeout_dir


@dataclass(frozen=True)
class AssetFile:
    """One image asset discovered during scanning.

    Attributes:
        relpath: Path relative to the scan root (e.g. ``Photos from 2023/img.jpg``).
        abspath: Absolute path to the image file.
        sidecar_path: Absolute path to the sidecar JSON file, or ``None`` if absent.
        mime: MIME type inferred from the file extension (e.g. ``"image/jpeg"``).
        size_bytes: File size in bytes.
    """

    relpath: str
    abspath: Path
    sidecar_path: Path | None
    mime: str
    size_bytes: int


def _find_sidecar(image_path: Path) -> Path | None:
    """Return the sidecar path for *image_path*, or ``None`` if not found.

    Google Photos Takeout typically names sidecars in one of two ways:

    1. ``<filename>.supplemental-metadata.json``
       (full filename including extension — the most common form)
    2. ``<stem>.supplemental-metadata.json``
       (stem only, seen occasionally)
    """
    # Primary: IMG_20230615.jpg → IMG_20230615.jpg.supplemental-metadata.json
    primary = image_path.parent / (image_path.name + SIDECAR_SUFFIX)
    if primary.exists():
        return primary
    # Fallback: IMG_20230615.jpg → IMG_20230615.supplemental-metadata.json
    fallback = image_path.parent / (image_path.stem + SIDECAR_SUFFIX)
    if fallback.exists():
        return fallback
    return None


def scan_takeout(takeout_root: Path) -> list[AssetFile]:
    """Walk *takeout_root* recursively and return all image assets found.

    The function skips sidecar JSON files and any non-image files.  Results
    are sorted by their relative path for deterministic ordering.

    Args:
        takeout_root: The directory to scan.  Typically the ``Takeout/``
            directory itself, or the directory that *contains* ``Takeout/``.

    Returns:
        Sorted list of :class:`AssetFile` instances.

    Raises:
        FileNotFoundError: If *takeout_root* does not exist.
    """
    if not takeout_root.exists():
        raise FileNotFoundError(f"Takeout root does not exist: {takeout_root}")

    assets: list[AssetFile] = []

    for abspath in sorted(takeout_root.rglob("*")):
        if not abspath.is_file():
            continue
        if abspath.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        # Sidecar files may have a double extension like ".jpg.supplemental-metadata.json"
        if SIDECAR_SUFFIX in abspath.name:
            continue

        relpath = str(abspath.relative_to(takeout_root))
        mime, _ = mimetypes.guess_type(str(abspath))
        mime = mime or "application/octet-stream"
        sidecar_path = _find_sidecar(abspath)
        size_bytes = abspath.stat().st_size

        assets.append(
            AssetFile(
                relpath=relpath,
                abspath=abspath,
                sidecar_path=sidecar_path,
                mime=mime,
                size_bytes=size_bytes,
            )
        )

    return assets
