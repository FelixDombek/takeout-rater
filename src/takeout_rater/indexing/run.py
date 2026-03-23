"""Reusable indexing function callable from both the CLI and the web API.

This module provides :func:`run_index` which scans a Google Photos Takeout
directory, upserts assets into the library database, and (optionally)
generates thumbnails.  It can be invoked from a background thread to avoid
blocking the web server while indexing is in progress.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


def _compute_sha256(path: Path) -> str:
    """Return the hex-encoded SHA-256 digest of the file at *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _bool_to_int(v: bool | None) -> int | None:
    """Convert an optional bool to the 0/1 integer stored in SQLite."""
    if v is None:
        return None
    return 1 if v else 0


@dataclass
class IndexProgress:
    """Tracks the progress of an indexing run.

    Attributes:
        running: ``True`` while the indexer is still working.
        done: ``True`` once the indexer has finished (successfully or not).
        error: Human-readable error message, or *None* on success.
        found: Total number of image files discovered during scanning.
        indexed: Number of assets upserted into the database so far.
        thumbs_ok: Number of thumbnails successfully generated.
        thumbs_skip: Number of thumbnails skipped (already existed or error).
        phase: Current phase — ``"scanning"`` while :func:`scan_takeout` is
            running; ``"indexing"`` once the DB-upsert loop has started.
        total_dirs: Total number of directories to scan (known after the fast
            first pass inside :func:`scan_takeout`).
        dirs_scanned: Number of directories fully processed so far.
        current_dir: Name of the directory most recently processed.
    """

    running: bool = False
    done: bool = False
    error: str | None = None
    found: int = 0
    indexed: int = 0
    thumbs_ok: int = 0
    thumbs_skip: int = 0
    phase: str = "scanning"
    total_dirs: int = 0
    dirs_scanned: int = 0
    current_dir: str = ""


def run_index(
    library_root: Path,
    conn: sqlite3.Connection,
    generate_thumbs: bool = True,
    on_progress: Callable[[IndexProgress], None] | None = None,
) -> IndexProgress:
    """Scan *library_root* and upsert discovered assets into *conn*.

    This function is safe to call from a background thread because the
    database connection is opened with ``check_same_thread=False``.

    Args:
        library_root: Directory that *contains* the ``Takeout/`` folder.
        conn: Open :class:`sqlite3.Connection` for the library database.
        generate_thumbs: When ``True`` (default), generate JPEG thumbnails for
            every newly discovered asset.  Skipped silently if Pillow is not
            installed.
        on_progress: Optional callback invoked after each asset is processed.
            Receives the current :class:`IndexProgress` instance.

    Returns:
        The final :class:`IndexProgress` describing what was indexed.
    """
    from takeout_rater.db.connection import library_state_dir  # noqa: PLC0415
    from takeout_rater.db.queries import upsert_asset  # noqa: PLC0415
    from takeout_rater.indexing.scanner import (  # noqa: PLC0415
        GOOGLE_PHOTOS_DIR_NAMES,
        find_google_photos_root,
        scan_takeout,
    )
    from takeout_rater.indexing.sidecar import parse_sidecar  # noqa: PLC0415

    progress = IndexProgress(running=True)

    takeout_dir = library_root / "Takeout"
    if not takeout_dir.exists():
        # Accept the user passing the Takeout/ dir directly (old-format exports).
        if list(library_root.glob("Photos from *")) or any(
            (library_root / name).is_dir() for name in GOOGLE_PHOTOS_DIR_NAMES
        ):
            takeout_dir = library_root
        else:
            progress.running = False
            progress.done = True
            progress.error = (
                f"No Takeout/ directory found inside {library_root}. "
                "Pass the directory that *contains* your Takeout/ folder."
            )
            return progress

    photos_root = find_google_photos_root(takeout_dir)

    # Wire the scanning-phase progress: on_dir_scanned is called by
    # scan_takeout after each directory's files are stat'd.
    def _on_dir_scanned(dirs_done: int, total_dirs: int, dir_name: str) -> None:
        progress.total_dirs = total_dirs
        progress.dirs_scanned = dirs_done
        progress.current_dir = dir_name
        if on_progress:
            on_progress(progress)

    assets = scan_takeout(photos_root, on_dir_scanned=_on_dir_scanned)
    progress.found = len(assets)
    progress.phase = "indexing"

    if on_progress:
        on_progress(progress)

    if not assets:
        progress.running = False
        progress.done = True
        return progress

    if generate_thumbs:
        thumbs_dir = library_state_dir(library_root) / "thumbs"
        thumbs_dir.mkdir(parents=True, exist_ok=True)
    else:
        thumbs_dir = None

    now = int(time.time())

    for asset_file in assets:
        progress.current_dir = os.path.dirname(asset_file.relpath)
        sidecar = None
        if asset_file.sidecar_path is not None:
            with contextlib.suppress(ValueError):
                sidecar = parse_sidecar(asset_file.sidecar_path)

        row: dict = {
            "relpath": asset_file.relpath,
            "filename": Path(asset_file.relpath).name,
            "ext": Path(asset_file.relpath).suffix.lower(),
            "size_bytes": asset_file.size_bytes,
            "mime": asset_file.mime,
            "sidecar_relpath": (
                str(asset_file.sidecar_path.relative_to(photos_root))
                if asset_file.sidecar_path
                else None
            ),
            "indexed_at": now,
        }

        # Compute SHA-256 content hash; skip silently on read errors.
        with contextlib.suppress(OSError):
            row["sha256"] = _compute_sha256(asset_file.abspath)

        if sidecar is not None:
            row.update(
                {
                    "title": sidecar.title,
                    "description": sidecar.description,
                    "google_photos_url": sidecar.google_photos_url,
                    "taken_at": sidecar.taken_at,
                    "created_at_sidecar": sidecar.created_at_sidecar,
                    "image_views": sidecar.image_views,
                    "geo_lat": sidecar.geo_lat,
                    "geo_lon": sidecar.geo_lon,
                    "geo_alt": sidecar.geo_alt,
                    "geo_exif_lat": sidecar.geo_exif_lat,
                    "geo_exif_lon": sidecar.geo_exif_lon,
                    "geo_exif_alt": sidecar.geo_exif_alt,
                    "favorited": _bool_to_int(sidecar.favorited),
                    "archived": _bool_to_int(sidecar.archived),
                    "trashed": _bool_to_int(sidecar.trashed),
                    "origin_type": sidecar.origin_type,
                    "origin_device_type": sidecar.origin_device_type,
                    "origin_device_folder": sidecar.origin_device_folder,
                    "app_source_package": sidecar.app_source_package,
                }
            )

        asset_id = upsert_asset(conn, row)
        progress.indexed += 1

        if generate_thumbs and thumbs_dir is not None:
            from takeout_rater.indexing.thumbnailer import (  # noqa: PLC0415
                generate_thumbnail,
                thumb_path_for_id,
            )

            thumb = thumb_path_for_id(thumbs_dir, asset_id)
            if not thumb.exists():
                try:
                    generate_thumbnail(asset_file.abspath, thumb)
                    progress.thumbs_ok += 1
                except (ImportError, OSError):
                    progress.thumbs_skip += 1
            else:
                progress.thumbs_skip += 1

        if on_progress:
            on_progress(progress)

    progress.running = False
    progress.done = True
    return progress
