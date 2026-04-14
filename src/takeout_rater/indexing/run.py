"""Reusable indexing function callable from both the CLI and the web API.

This module provides :func:`run_index` which scans a Google Photos Takeout
directory and processes it in two phases:

1. **Index phase** – fast scan of filesystem metadata; upserts each asset into
   the library database with only the basic file properties (relpath, filename,
   extension, size, MIME type, etc.).  No image file is opened and no sidecar
   JSON is read in this phase.

2. **Thumbnail phase** – parallel workers process each asset: they compute the
   SHA-256 content hash, parse the sidecar JSON (if present), and generate a
   JPEG thumbnail.  Results are written back to the database sequentially from
   the main thread.  A deduplication pass then merges byte-identical files
   (same SHA-256) into the ``asset_paths`` table.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import sqlite3
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from takeout_rater.db.queries import CURRENT_INDEXER_VERSION as _CURRENT_INDEXER_VERSION


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
            running; ``"indexing"`` once the DB-upsert loop has started;
            ``"thumbnailing"`` during parallel thumbnail/sha256/sidecar work.
        total_dirs: Total number of directories to scan (known after the fast
            first pass inside :func:`scan_takeout`).
        dirs_scanned: Number of directories fully processed so far.
        current_dir: Name of the directory most recently processed.
        thumbs_total: Total number of assets to process in the
            ``"thumbnailing"`` phase (0 until that phase begins).
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
    thumbs_total: int = 0


def run_index(
    library_root: Path,
    conn: sqlite3.Connection,
    on_progress: Callable[[IndexProgress], None] | None = None,
) -> IndexProgress:
    """Scan *library_root* and upsert discovered assets into *conn*.

    Processing happens in two phases for maximum throughput:

    * **Index phase** – inserts basic file metadata into the database without
      reading image files or sidecar JSONs.
    * **Thumbnail phase** – a thread pool processes each asset in parallel:
      SHA-256 computation, sidecar parsing, and JPEG thumbnail generation all
      happen concurrently.  DB updates are applied from the main thread as
      results arrive.  A final deduplication pass merges byte-identical files.

    This function is safe to call from a background thread because the
    database connection is opened with ``check_same_thread=False``.

    Args:
        library_root: Directory that *contains* the ``Takeout/`` folder.
        conn: Open :class:`sqlite3.Connection` for the library database.
        on_progress: Optional callback invoked after each asset is processed.
            Receives the current :class:`IndexProgress` instance.

    Returns:
        The final :class:`IndexProgress` describing what was indexed.
    """
    from takeout_rater.db.connection import library_state_dir  # noqa: PLC0415
    from takeout_rater.db.queries import (  # noqa: PLC0415
        dedup_assets_by_sha256,
        update_asset_metadata,
        upsert_asset,
    )
    from takeout_rater.indexing.scanner import (  # noqa: PLC0415
        GOOGLE_PHOTOS_DIR_NAMES,
        find_google_photos_root,
        scan_takeout,
    )
    from takeout_rater.indexing.sidecar import parse_sidecar  # noqa: PLC0415
    from takeout_rater.indexing.thumbnailer import (  # noqa: PLC0415
        generate_thumbnail,
        thumb_path_for_id,
    )

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

    thumbs_dir = library_state_dir(library_root) / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    now = int(time.time())

    # ── Phase 1: Fast index ───────────────────────────────────────────────────
    # Insert only the basic filesystem metadata for each asset.  No image file
    # is opened and no sidecar is read.  This loop is intentionally lightweight
    # so the database is populated quickly even for very large libraries.
    #
    # We also build the work-list for Phase 2 (thumbnail + sidecar + sha256).
    _thumb_work: list[tuple[int, Path, Path | None, Path]] = []

    for asset_file in assets:
        progress.current_dir = os.path.dirname(asset_file.relpath)
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
            "indexer_version": _CURRENT_INDEXER_VERSION,
        }

        asset_id = upsert_asset(conn, row)
        progress.indexed += 1

        thumb = thumb_path_for_id(thumbs_dir, asset_id)
        _thumb_work.append((asset_id, asset_file.abspath, asset_file.sidecar_path, thumb))

        if on_progress:
            on_progress(progress)

    # ── Phase 2: Parallel thumbnail + sidecar + SHA-256 ──────────────────────
    # Each worker computes the sha256, parses the sidecar, and generates the
    # thumbnail.  Workers are I/O-bound so the GIL is released for most of the
    # time, providing genuine parallelism across CPU cores.  Database updates
    # are applied from the *main thread* as futures complete to keep SQLite
    # writes serial and consistent.

    progress.phase = "thumbnailing"
    progress.thumbs_total = len(_thumb_work)

    if on_progress:
        on_progress(progress)

    def _do_thumb(
        item: tuple[int, Path, Path | None, Path],
    ) -> tuple[int, str | None, dict, bool | None]:
        """Worker: compute sha256, parse sidecar, generate thumbnail.

        Returns ``(asset_id, sha256, sidecar_updates, thumb_ok)`` where
        ``thumb_ok`` is ``True`` on success, ``False`` on error, and ``None``
        when the thumbnail already existed and was skipped.
        """
        asset_id, abspath, sidecar_path, thumb = item

        # Compute SHA-256
        sha256: str | None = None
        with contextlib.suppress(OSError):
            sha256 = _compute_sha256(abspath)

        # Parse sidecar
        sidecar_updates: dict = {}
        if sidecar_path is not None:
            with contextlib.suppress(ValueError):
                sidecar = parse_sidecar(sidecar_path)
                sidecar_updates = {
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

        # Generate thumbnail
        if thumb.exists():
            thumb_ok: bool | None = None  # skip — already present
        else:
            try:
                generate_thumbnail(abspath, thumb)
                thumb_ok = True
            except (ImportError, OSError):
                thumb_ok = False

        return asset_id, sha256, sidecar_updates, thumb_ok

    num_workers = os.cpu_count() or 1

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_do_thumb, item) for item in _thumb_work]
        for future in as_completed(futures):
            asset_id, sha256, sidecar_updates, thumb_ok = future.result()

            # Apply metadata back to the DB from the main thread.
            updates: dict = {}
            if sha256 is not None:
                updates["sha256"] = sha256
            updates.update(sidecar_updates)
            if updates:
                update_asset_metadata(conn, asset_id, updates)

            if thumb_ok is True:
                progress.thumbs_ok += 1
            else:
                progress.thumbs_skip += 1

            if on_progress:
                on_progress(progress)

    # Commit all metadata updates before the dedup pass.
    conn.commit()

    # ── Phase 3: Deduplication ────────────────────────────────────────────────
    # Merge byte-identical files (same SHA-256) so each unique image has
    # exactly one ``assets`` row; duplicate paths become ``asset_paths`` aliases.
    dedup_assets_by_sha256(conn)

    progress.running = False
    progress.done = True
    return progress
