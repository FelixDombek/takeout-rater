"""Reusable indexing function callable from both the CLI and the web API.

This module provides :func:`run_index` which processes a Google Photos
Takeout directory in two distinct phases:

1. **Scan phase** – pure filesystem walk; collects file-stat metadata into an
   in-memory list without opening any image file or reading any sidecar JSON.
   This is extremely fast and produces no database I/O.

2. **Thumbnailing phase** – does all the file-reading work in two parallel
   passes followed by a serialised database step:

   * **Pass A** (parallel) – each worker computes the SHA-256 content hash and
     parses the sidecar JSON for its assigned asset.  No database access.
   * **DB step** (main thread, serialised) – results from completed Pass A
     futures are applied to the database one-by-one as they arrive.
     :func:`~takeout_rater.db.queries.upsert_asset` is called with the full
     row (including ``sha256``), so its built-in content-deduplication logic
     handles exact duplicates by recording alias paths in ``asset_paths``
     instead of creating a second ``assets`` row.  After the upsert, the
     canonical thumbnail path is looked up; if it does not yet exist the
     asset is added to the work-list for Pass B.
   * **Pass B** (parallel) – each worker generates the JPEG thumbnail for one
     canonical asset.  Assets that were merged as aliases, or whose thumbnail
     was already on disk, are automatically skipped.

Progress is reported via the :class:`IndexProgress` dataclass.  The
:attr:`IndexProgress.pct` property exposes a unified 0–100 % figure that
spans all phases and never resets at phase transitions.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import sqlite3
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
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
    """Tracks the progress of an indexing run across all phases.

    Attributes:
        running: ``True`` while the indexer is still working.
        done: ``True`` once the indexer has finished (successfully or not).
        error: Human-readable error message, or *None* on success.
        found: Total number of image files discovered during scanning.
        indexed: Number of assets upserted into the database so far.
        thumbs_ok: Number of thumbnails successfully generated (Pass B).
        thumbs_skip: Number of thumbnails skipped (already existed, error, or
            the asset was a duplicate alias).
        phase: Current phase — ``"scanning"`` while :func:`scan_takeout` is
            running; ``"thumbnailing"`` for Pass A + DB step + Pass B.
        total_dirs: Total number of directories to scan (filled during scan).
        dirs_scanned: Number of directories fully processed so far.
        current_dir: Name of the directory most recently processed.
        thumbs_total: Total number of assets to process in the thumbnailing
            phase (set to ``found`` as soon as the phase begins).
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

    @property
    def pct(self) -> float:
        """Unified progress percentage (0–100) across all phases.

        The bar is partitioned as follows, so it never resets at phase
        transitions:

        * **0 – 5 %** – scanning phase (proportional to dirs scanned).
        * **5 – 52.5 %** – Pass A of thumbnailing (sha256 + sidecar,
          proportional to assets processed through ``indexed``).
        * **52.5 – 100 %** – Pass B of thumbnailing (thumbnail generation,
          proportional to ``thumbs_ok + thumbs_skip``).
        """
        if self.phase == "scanning":
            if self.total_dirs > 0:
                return (self.dirs_scanned / self.total_dirs) * 5.0
            return 0.0
        # thumbnailing phase: both passes combined fill the remaining 95 %.
        # Each asset contributes two "units" (one for sha256+sidecar, one for
        # the thumbnail decision), so total_units = 2 * found.
        if self.found > 0:
            a_done = self.indexed  # completed Pass A units
            b_done = self.thumbs_ok + self.thumbs_skip  # completed Pass B units
            return 5.0 + ((a_done + b_done) / (2 * self.found)) * 95.0
        return 5.0


def run_index(
    library_root: Path,
    conn: sqlite3.Connection,
    on_progress: Callable[[IndexProgress], None] | None = None,
) -> IndexProgress:
    """Scan *library_root* and upsert discovered assets into *conn*.

    Processing happens in two phases (scan → thumbnailing) for maximum
    throughput.  See the module docstring for a detailed description of each
    sub-step.

    Args:
        library_root: Directory that *contains* the ``Takeout/`` folder.
        conn: Open :class:`sqlite3.Connection` for the library database.
        on_progress: Optional callback invoked after each asset is processed.
            Receives the current :class:`IndexProgress` instance.  Will be
            called from the main thread; implementations must not block.

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
    from takeout_rater.indexing.thumbnailer import (  # noqa: PLC0415
        generate_thumbnail,
        thumb_path_for_id,
    )

    progress = IndexProgress(running=True)

    # ── Resolve the photos root ───────────────────────────────────────────────
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

    # ── Phase 1: Scan ─────────────────────────────────────────────────────────
    # Pure filesystem walk — no file reads, no DB access.
    # scan_takeout calls _on_dir_scanned after completing each album directory.

    def _on_dir_scanned(dirs_done: int, total_dirs: int, dir_name: str) -> None:
        progress.total_dirs = total_dirs
        progress.dirs_scanned = dirs_done
        progress.current_dir = dir_name
        if on_progress:
            on_progress(progress)

    assets = scan_takeout(photos_root, on_dir_scanned=_on_dir_scanned)
    progress.found = len(assets)

    if not assets:
        progress.running = False
        progress.done = True
        return progress

    # ── Phase 2: Thumbnailing ─────────────────────────────────────────────────
    progress.phase = "thumbnailing"
    progress.thumbs_total = len(assets)
    if on_progress:
        on_progress(progress)

    thumbs_dir = library_state_dir(library_root) / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    now = int(time.time())
    num_workers = os.cpu_count() or 1

    # ── Pass A: parallel sha256 + sidecar ─────────────────────────────────────
    # Workers read each file and its sidecar but do NOT touch the DB.

    def _do_sha_sidecar(asset_file: object) -> tuple:
        """Compute sha256 and parse sidecar; return a row-ready dict."""
        sha256: str | None = None
        with contextlib.suppress(OSError):
            sha256 = _compute_sha256(asset_file.abspath)  # type: ignore[union-attr]

        sidecar_updates: dict = {}
        if asset_file.sidecar_path is not None:  # type: ignore[union-attr]
            with contextlib.suppress(ValueError):
                sidecar = parse_sidecar(asset_file.sidecar_path)  # type: ignore[union-attr]
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

        return (asset_file, sha256, sidecar_updates)

    # Work-list for Pass B: (asset_id, abspath, thumb_path).
    # A set of already-queued asset_ids prevents duplicate thumbnail work when
    # the same sha256 appears more than once in the same indexing run.
    _needs_thumb: list[tuple[int, Path, Path]] = []
    _thumb_queued: set[int] = set()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all Pass A workers up-front so they run in parallel.
        futures: list[Future] = [executor.submit(_do_sha_sidecar, af) for af in assets]

        # Process results as they complete.  DB access is serialised here in
        # the main thread; workers only do pure I/O.
        for future in as_completed(futures):
            asset_file, sha256, sidecar_updates = future.result()

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
            if sha256 is not None:
                row["sha256"] = sha256
            row.update(sidecar_updates)

            # upsert_asset handles sha256-based deduplication: if another
            # asset row already carries the same hash the new relpath is
            # recorded as an alias in asset_paths and the canonical id is
            # returned.
            asset_id = upsert_asset(conn, row)
            progress.indexed += 1

            # Check whether a thumbnail already exists for this canonical
            # asset.  This covers both re-indexing runs (thumb was generated
            # on the previous run) and the case where an alias was merged into
            # an existing canonical whose thumb is already present.
            thumb = thumb_path_for_id(thumbs_dir, asset_id)
            if not thumb.exists() and asset_id not in _thumb_queued:
                _needs_thumb.append((asset_id, asset_file.abspath, thumb))
                _thumb_queued.add(asset_id)
            else:
                progress.thumbs_skip += 1

            if on_progress:
                on_progress(progress)

    # ── Pass B: parallel thumbnail generation ─────────────────────────────────
    # Only runs for canonical assets whose thumbnail does not yet exist.

    def _gen_thumb(item: tuple[int, Path, Path]) -> tuple[int, bool]:
        asset_id, abspath, thumb = item
        try:
            generate_thumbnail(abspath, thumb)
            return asset_id, True
        except (ImportError, OSError):
            return asset_id, False

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures_b: list[Future] = [executor.submit(_gen_thumb, item) for item in _needs_thumb]
        for future in as_completed(futures_b):
            _asset_id, ok = future.result()
            if ok:
                progress.thumbs_ok += 1
            else:
                progress.thumbs_skip += 1
            if on_progress:
                on_progress(progress)

    progress.running = False
    progress.done = True
    return progress
