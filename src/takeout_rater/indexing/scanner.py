"""Walk a Google Photos Takeout directory tree and enumerate image assets."""

from __future__ import annotations

import mimetypes
import os
from collections.abc import Callable
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


def scan_takeout(
    takeout_root: Path,
    on_dir_scanned: Callable[[int, int, str], None] | None = None,
) -> list[AssetFile]:
    """Walk *takeout_root* recursively and return all image assets found.

    The scan runs in two phases to enable progress reporting on large libraries:

    **Phase 1 – directory enumeration (no additional per-file ``stat`` calls):**
    :func:`os.walk` is used to collect every ``(directory, image_filenames)``
    pair using the filename metadata provided by the OS directory listing.
    Only filename extensions are checked; no file content is read.

    **Phase 2 – metadata collection (one ``stat`` + sidecar probe per file):**
    Each collected image path is stat'd and its sidecar is located.
    ``on_dir_scanned`` is called once per directory after all its files have
    been processed.

    Args:
        takeout_root: The directory to scan.  Typically the ``Takeout/``
            directory itself, or the directory that *contains* ``Takeout/``.
        on_dir_scanned: Optional callback invoked once per directory during
            **phase 2**, after that directory's files have been stat'd.
            Receives ``(dirs_done: int, total_dirs: int, dir_name: str)``.
            Can be used to drive a progress bar during scanning.

    Returns:
        Sorted list of :class:`AssetFile` instances.

    Raises:
        FileNotFoundError: If *takeout_root* does not exist.
    """
    if not takeout_root.exists():
        raise FileNotFoundError(f"Takeout root does not exist: {takeout_root}")

    # ------------------------------------------------------------------
    # Phase 1: enumerate directories and image filenames via os.walk.
    # os.walk yields DirEntry metadata from the OS directory listing, so
    # no extra stat() call is needed per file at this stage.
    # ------------------------------------------------------------------
    dir_images: list[tuple[Path, list[str]]] = []
    for dirpath_str, _subdirs, filenames in os.walk(takeout_root):
        dp = Path(dirpath_str)
        images = sorted(
            f
            for f in filenames
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS and SIDECAR_SUFFIX not in f
        )
        dir_images.append((dp, images))

    # Sort directories for deterministic ordering.
    dir_images.sort(key=lambda t: str(t[0]))
    total_dirs = len(dir_images)

    # ------------------------------------------------------------------
    # Phase 2: stat each image file, locate sidecars, build AssetFile
    # objects.  Call on_dir_scanned after finishing each directory.
    # ------------------------------------------------------------------
    assets: list[AssetFile] = []
    for i, (dp, image_names) in enumerate(dir_images):
        for fname in image_names:
            abspath = dp / fname
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
        if on_dir_scanned is not None:
            on_dir_scanned(i + 1, total_dirs, dp.name)

    # Final sort by relpath for deterministic output (existing contract).
    assets.sort(key=lambda a: a.relpath)
    return assets
