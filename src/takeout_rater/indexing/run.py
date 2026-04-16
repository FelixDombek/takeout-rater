"""Reusable indexing function callable from both the CLI and the web API.

This module provides :func:`run_index` which processes a Google Photos
Takeout directory in two distinct phases:

1. **Scan phase** – pure filesystem walk; collects file-stat metadata into an
   in-memory list without opening any image file or reading any sidecar JSON.
   This is extremely fast and produces no database I/O.

2. **Processing phase** – fully parallel per-file workers.  Each worker:

   * Reads file bytes and computes SHA-256 in parallel
   * Parses the sidecar JSON in parallel (if present)
   * Acquires a single shared mutex (:data:`_claim_lock`) only for the critical section:
     check ``lookup_sha256`` → ``upsert_asset`` (which records aliases if needed)
   * After claiming ownership: computes phash, then CLIP embedding (serialised
     via a per-run lock to prevent PyTorch deadlocks), and thumbnail in
     parallel with other workers' non-CLIP steps

The mutex guards only the minimal critical section required to prevent two
workers from both claiming the same sha256-identified asset.  CLIP inference
is additionally serialised (one worker at a time) to avoid a PyTorch
thread-pool deadlock that occurs when multiple Python threads call
``encode_image()`` concurrently.  All other per-asset work (SHA-256, sidecar
parsing, phash, thumbnails) proceeds fully in parallel.

Progress is reported via the :class:`IndexProgress` dataclass.  The
:attr:`IndexProgress.pct` property exposes a unified 0–100 % figure that
spans all phases and never resets at phase transitions.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import sqlite3
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from takeout_rater.db.queries import CURRENT_INDEXER_VERSION as _CURRENT_INDEXER_VERSION

_log = logging.getLogger(__name__)


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
        phase: Current phase — ``"scanning"`` while :func:`scan_takeout` is
            running; ``"processing"`` for the parallel per-file worker phase.
        total_dirs: Total number of directories to scan (filled during scan).
        dirs_scanned: Number of directories fully processed so far.
        current_dir: Name of the directory most recently processed.
    """

    running: bool = False
    done: bool = False
    cancelled: bool = False
    error: str | None = None
    found: int = 0
    indexed: int = 0
    phase: str = "scanning"
    total_dirs: int = 0
    dirs_scanned: int = 0
    current_dir: str = ""

    @property
    def pct(self) -> float:
        """Unified progress percentage (0–100) across all phases.

        The bar is partitioned as follows, so it never resets at phase
        transitions:

        * **0 – 5 %** – scanning phase (proportional to dirs scanned).
        * **5 %** – loading_models phase (held at 5 % while models warm up).
        * **5 – 100 %** – processing phase (proportional to assets indexed).
        """
        if self.phase == "scanning":
            if self.total_dirs > 0:
                return (self.dirs_scanned / self.total_dirs) * 5.0
            return 0.0
        if self.phase == "loading_models":
            return 5.0
        # processing phase: remaining 95% proportional to indexed
        if self.found > 0:
            return 5.0 + (self.indexed / self.found) * 95.0
        return 5.0


def run_index(
    photos_root: Path,
    conn: sqlite3.Connection,
    db_root: Path | None = None,
    on_progress: Callable[[IndexProgress], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> IndexProgress:
    """Scan *photos_root* and upsert discovered assets into *conn*.

    Processing happens in two phases (scan → processing) for maximum
    throughput.  See the module docstring for a detailed description of each
    sub-step.

    Args:
        photos_root: The directory that directly contains the album sub-folders
            (e.g. ``Google Photos/``, or any arbitrary photo folder).  No
            ``Takeout/`` wrapper is assumed.
        conn: Open :class:`sqlite3.Connection` for the library database.
        db_root: Directory where the ``takeout-rater/`` state directory (thumbs,
            DB) should be written.  Defaults to *photos_root* when not given.
        on_progress: Optional callback invoked after each asset is processed.
            Receives the current :class:`IndexProgress` instance.  Will be
            called from the main thread; implementations must not block.
        cancel_check: Optional callable that returns ``True`` when the run
            should be aborted.  Checked before each asset is processed; when
            it returns ``True`` the worker skips the remaining work and the
            run finishes early.

    Returns:
        The final :class:`IndexProgress` describing what was indexed.
    """
    import threading  # noqa: PLC0415

    from takeout_rater.db.connection import (  # noqa: PLC0415
        library_db_path,
        library_state_dir,
        open_db,
    )
    from takeout_rater.db.queries import lookup_sha256, upsert_asset  # noqa: PLC0415
    from takeout_rater.indexing.scanner import scan_takeout  # noqa: PLC0415
    from takeout_rater.indexing.sidecar import parse_sidecar  # noqa: PLC0415
    from takeout_rater.indexing.thumbnailer import (  # noqa: PLC0415
        generate_thumbnail,
        generate_thumbnail_from_image,
        thumb_path_for_id,
    )
    from takeout_rater.scoring.phash import DHASH_ALGO, compute_dhash_from_image  # noqa: PLC0415

    if db_root is None:
        db_root = photos_root

    progress = IndexProgress(running=True)

    # ── Phase 1: Scan + concurrent model pre-load ─────────────────────────────
    # The filesystem walk is pure I/O — no file reads, no DB access.
    # We kick off a background thread to download/warm the CLIP backbone at
    # the same time so that model loading (which can take several minutes on a
    # first run) overlaps with the scan rather than blocking the processing
    # phase.
    #
    # _clip_warmup_ok is set ONLY after get_clip_model() returns
    # successfully.  Workers check this event before attempting CLIP inference,
    # so they never compete for the model-loading lock.  If the warm-up times
    # out (e.g. because the model host is unreachable and the request hangs),
    # workers skip CLIP entirely rather than blocking indefinitely — which
    # would also prevent Ctrl-C from working (Python's ThreadPoolExecutor
    # atexit handler waits for all workers to finish).
    _clip_warmup_ok = threading.Event()

    def _warmup_clip() -> None:
        try:
            from takeout_rater.scorers.adapters.clip_backbone import (  # noqa: PLC0415
                get_clip_model,
                is_available,
            )

            if is_available():
                get_clip_model()
                _clip_warmup_ok.set()  # only set after successful return
        except Exception:  # noqa: BLE001
            pass

    _warmup_thread = threading.Thread(target=_warmup_clip, daemon=True, name="clip-warmup")
    _warmup_thread.start()

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

    # ── Phase 1b: Wait for model warm-up if still loading ─────────────────────
    # If the scan finished before the model loaded, block here (up to 5 min)
    # so that workers skip CLIP rather than block.  The UI shows
    # "Loading CLIP model…" during this window.
    if _warmup_thread.is_alive():
        progress.phase = "loading_models"
        if on_progress:
            on_progress(progress)
        _warmup_thread.join(timeout=300)

    # ── Phase 2: Processing ───────────────────────────────────────────────────
    progress.phase = "processing"
    if on_progress:
        on_progress(progress)

    thumbs_dir = library_state_dir(db_root) / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    db_path = library_db_path(db_root)
    now = int(time.time())
    num_workers = os.cpu_count() or 1

    # _claim_lock guards the critical section: lookup_sha256 + upsert_asset.
    # This prevents two workers from both claiming the same hash as "new".
    _claim_lock = threading.Lock()
    _progress_lock = threading.Lock()
    # _clip_lock serialises CLIP inference to ONE worker at a time.
    # PyTorch's internal C++ thread pool can deadlock when multiple Python
    # threads all call encode_image() simultaneously (especially on CPU with
    # MKL/OpenBLAS).  Serialising inference avoids that deadlock entirely while
    # still allowing all other per-asset work (SHA-256, sidecar, phash,
    # thumbnails) to proceed in parallel.
    _clip_lock = threading.Lock()

    def _process_one(asset_file: object) -> None:
        """Process a single asset: hash, sidecar, claim, then thumb+phash+embed."""
        if cancel_check is not None and cancel_check():
            return
        relpath: str = asset_file.relpath  # type: ignore[union-attr]
        try:
            _process_one_inner(asset_file, relpath)
        except Exception:
            _log.exception("Unexpected error processing asset %r – skipping", relpath)

    def _process_one_inner(asset_file: object, relpath: str) -> None:  # noqa: PLR0912,PLR0915
        # Step 1: Read file bytes + compute sha256 (parallel, no locking)
        sha256: str | None = None
        file_bytes: bytes | None = None
        try:
            with open(asset_file.abspath, "rb") as f:  # type: ignore[union-attr]
                file_bytes = f.read()
            sha256 = hashlib.sha256(file_bytes).hexdigest()
        except OSError:
            _log.debug("Could not read file %r – skipping SHA-256", relpath)

        # Step 2: Parse sidecar if present (parallel, no locking)
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

        # Step 3: Critical section – check hash + upsert (mutex-guarded).
        # Use open_db (no migrations) since the DB is already initialised.
        # Each iteration opens and immediately closes a short-lived connection
        # so that no connection is shared across threads.
        is_new: bool = False
        asset_id: int = 0
        _log.debug("Claiming asset %r (sha256=%s)", relpath, sha256 and sha256[:8])
        with _claim_lock:
            wconn = open_db(db_path)
            try:
                existing = lookup_sha256(wconn, sha256) if sha256 else None
                is_new = existing is None

                row: dict = {
                    "relpath": relpath,
                    "filename": Path(relpath).name,
                    "ext": Path(relpath).suffix.lower(),
                    "size_bytes": asset_file.size_bytes,  # type: ignore[union-attr]
                    "mime": asset_file.mime,  # type: ignore[union-attr]
                    "sidecar_relpath": (
                        str(asset_file.sidecar_path.relative_to(photos_root))  # type: ignore[union-attr]
                        if asset_file.sidecar_path  # type: ignore[union-attr]
                        else None
                    ),
                    "indexed_at": now,
                    "indexer_version": _CURRENT_INDEXER_VERSION,
                }
                if sha256 is not None:
                    row["sha256"] = sha256
                row.update(sidecar_updates)

                asset_id = upsert_asset(wconn, row)

                # Link the asset to its album (the top-level directory it lives in).
                parts = Path(relpath).parts
                if len(parts) > 1:
                    from takeout_rater.db.queries import (  # noqa: PLC0415
                        link_asset_to_album,
                        upsert_album,
                    )

                    album_name = parts[0]
                    album_id = upsert_album(wconn, album_name, album_name)
                    link_asset_to_album(wconn, album_id, asset_id)
            finally:
                wconn.close()
        _log.debug("Claimed asset %r → id=%d is_new=%s", relpath, asset_id, is_new)

        # Update progress (guarded by separate lock)
        with _progress_lock:
            progress.indexed += 1
            if on_progress:
                on_progress(progress)

        # Step 4: Thumbnail first, then phash + CLIP on the thumbnail image.
        #
        # Thumbnail generation is always attempted (for both new and known
        # assets).  When the thumbnail is generated from in-memory bytes we
        # get back a small PIL Image that is immediately reused for phash and
        # CLIP — avoiding a second round-trip to disk for new assets.
        thumb = thumb_path_for_id(thumbs_dir, asset_id)
        thumb_img = None  # PIL thumbnail image, reused for phash + CLIP

        if not thumb.exists():
            if file_bytes:
                try:
                    import io  # noqa: PLC0415

                    from PIL import Image  # noqa: PLC0415

                    full_img = Image.open(io.BytesIO(file_bytes))
                    thumb_img = generate_thumbnail_from_image(full_img, thumb)
                except ImportError:
                    pass  # Pillow not available
                except Exception:
                    _log.debug("Thumbnail generation failed for %r", relpath, exc_info=True)
                    with contextlib.suppress(OSError):
                        thumb.unlink(missing_ok=True)
            else:
                # No in-memory bytes; fall back to reading from disk.
                with contextlib.suppress(ImportError, OSError):
                    generate_thumbnail(asset_file.abspath, thumb)  # type: ignore[union-attr]

        # For new assets: compute phash + CLIP using the thumbnail image.
        # If the thumbnail was already on disk (rare re-index case), load it.
        if is_new:
            if thumb_img is None and thumb.exists():
                try:
                    import io  # noqa: PLC0415

                    from PIL import Image  # noqa: PLC0415

                    thumb_img = Image.open(io.BytesIO(thumb.read_bytes()))
                except ImportError:
                    pass
                except Exception:
                    _log.debug("Could not load thumbnail for phash/CLIP %r", relpath, exc_info=True)

            if thumb_img is not None:
                # Compute phash from thumbnail.
                try:
                    from takeout_rater.db.queries import upsert_phash  # noqa: PLC0415

                    dhash_hex = compute_dhash_from_image(thumb_img)
                    wconn2 = open_db(db_path)
                    try:
                        upsert_phash(wconn2, asset_id, dhash_hex, DHASH_ALGO)
                    finally:
                        wconn2.close()
                except ImportError:
                    pass
                except Exception:
                    _log.warning("phash failed for %r", relpath, exc_info=True)

                # Compute CLIP embedding from thumbnail.
                if _clip_warmup_ok.is_set():
                    _log.debug("CLIP inference start for %r", relpath)
                    try:
                        import struct  # noqa: PLC0415

                        import torch  # noqa: PLC0415

                        from takeout_rater.scorers.adapters.clip_backbone import (
                            get_clip_model,  # noqa: PLC0415
                        )

                        model, preprocess, _tokenizer, device = get_clip_model()
                        img_rgb = thumb_img.convert("RGB")
                        img_tensor = preprocess(img_rgb).unsqueeze(0).to(device)
                        with _clip_lock, torch.no_grad():
                            embedding = model.encode_image(img_tensor)
                            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                            embedding = embedding.cpu().float().numpy()[0]
                        blob = struct.pack(f"{embedding.shape[0]}f", *embedding)
                        wconn3 = open_db(db_path)
                        try:
                            wconn3.execute(
                                "INSERT OR REPLACE INTO clip_embeddings"
                                " (asset_id, embedding, computed_at) VALUES (?, ?, ?)",
                                (asset_id, blob, now),
                            )
                            wconn3.commit()
                        finally:
                            wconn3.close()
                        _log.debug("CLIP inference done for %r", relpath)
                    except ImportError:
                        pass  # torch / open_clip not available
                    except Exception:
                        _log.warning("CLIP embedding failed for %r", relpath, exc_info=True)

    # Submit all workers in parallel; _process_one swallows every exception
    # internally (logging it) so future.result() never re-raises and the
    # executor always shuts down cleanly.
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures: list[Future] = [executor.submit(_process_one, af) for af in assets]
        for future in as_completed(futures):
            future.result()

    progress.running = False
    progress.done = True
    if cancel_check is not None and cancel_check():
        progress.cancelled = True
    return progress
