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
   * After claiming ownership: computes thumbnail and pHash in parallel, then
     queues the thumbnail for a dedicated batched CLIP embedding worker

The mutex guards only the minimal critical section required to prevent two
workers from both claiming the same sha256-identified asset.  CLIP inference
runs on one dedicated worker thread that batches queued thumbnails, avoiding
the PyTorch thread-pool deadlock that occurs when multiple Python threads call
``encode_image()`` concurrently while still giving the GPU real image batches.
All other per-asset work (SHA-256, sidecar parsing, phash, thumbnails)
proceeds fully in parallel.

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
        phase: Current phase — ``"scanning"`` while :func:`scan_photos_tree` is
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
    db_root: Path,
    on_progress: Callable[[IndexProgress], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> IndexProgress:
    """Scan *photos_root* and upsert discovered assets into *conn*.

    Processing happens in two phases (scan → processing) for maximum
    throughput.  See the module docstring for a detailed description of each
    sub-step.

    Args:
        photos_root: The library root directory.  May be the directory that
            directly contains album sub-folders.
        conn: Open :class:`sqlite3.Connection` for the library database.
        db_root: Directory where the ``takeout-rater/`` state directory (thumbs,
            DB) should be written.  This must be supplied by callers; it is not
            inferred from *photos_root*.
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
    import queue
    import threading

    from src.takeout_rater.clustering.phash import (
        DHASH_ALGO,
        compute_dhash_from_image,
    )
    from takeout_rater.db.connection import (
        library_db_path,
        library_state_dir,
        open_db,
    )
    from takeout_rater.db.queries import lookup_sha256, upsert_asset
    from takeout_rater.indexing.scanner import scan_photos_tree
    from takeout_rater.indexing.sidecar import parse_sidecar
    from takeout_rater.indexing.thumbnailer import (
        generate_thumbnail,
        generate_thumbnail_from_image,
        thumb_path_for_id,
    )

    if db_root is None:
        raise ValueError("db_root is required")

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
            from takeout_rater.scoring.scorers.clip_backbone import (
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

    assets = scan_photos_tree(photos_root, on_dir_scanned=_on_dir_scanned)
    progress.found = len(assets)

    if not assets:
        # No assets to index; the warmup thread is a daemon and will be cleaned
        # up by the process, but attempt a very short join in case it already
        # finished to avoid leaving unnecessary background work.
        _warmup_thread.join(timeout=0.5)
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

    # Pre-fetch the sets of asset IDs that already have phash / CLIP records.
    # Workers share these read-only sets to check whether phash/CLIP need to be
    # computed without opening an extra DB connection per asset.  Assets that
    # are genuinely new (is_new=True) always have phash/CLIP computed; the sets
    # are only consulted for existing assets (is_new=False) so that a
    # previously-aborted indexing run — where the asset row was committed but
    # phash/CLIP were not saved — is corrected on the next run.
    _ids_with_phash: frozenset[int] = frozenset(
        r[0] for r in conn.execute("SELECT asset_id FROM phash").fetchall()
    )
    _ids_with_clip: frozenset[int] = frozenset(
        r[0] for r in conn.execute("SELECT asset_id FROM clip_embeddings").fetchall()
    )

    # _claim_lock guards the critical section: lookup_sha256 + upsert_asset.
    # This prevents two workers from both claiming the same hash as "new".
    _claim_lock = threading.Lock()
    _progress_lock = threading.Lock()
    # CLIP inference is handled by one dedicated worker that forms batches.
    # Calling encode_image() from many Python workers can deadlock PyTorch's
    # thread pool, but calling it serially one image at a time underutilises the
    # GPU.  A bounded queue gives us backpressure and real CLIP batches.
    _clip_batch_size = 32
    _clip_queue_max = 128
    _clip_queue: queue.Queue[object] = queue.Queue(maxsize=_clip_queue_max)
    _clip_sentinel = object()

    def _clip_batch_worker() -> None:
        import struct

        try:
            import torch

            from takeout_rater.scoring.scorers.clip_backbone import (
                get_clip_model,
            )
        except ImportError:
            return

        try:
            model, preprocess, _tokenizer, device = get_clip_model()
        except Exception:  # noqa: BLE001
            _log.warning("CLIP batch worker failed to load model", exc_info=True)
            return

        wconn = open_db(db_path)
        try:
            while True:
                item = _clip_queue.get()
                if item is _clip_sentinel:
                    _clip_queue.task_done()
                    break

                batch = [item]
                saw_sentinel = False
                while len(batch) < _clip_batch_size:
                    try:
                        next_item = _clip_queue.get(timeout=0.02)
                    except queue.Empty:
                        break
                    if next_item is _clip_sentinel:
                        _clip_queue.task_done()
                        saw_sentinel = True
                        break
                    batch.append(next_item)

                try:
                    asset_ids = [entry[0] for entry in batch]  # type: ignore[index]
                    relpaths = [entry[1] for entry in batch]  # type: ignore[index]
                    images = [entry[2] for entry in batch]  # type: ignore[index]
                    _log.debug("CLIP batch inference start: %d image(s)", len(images))
                    tensors = [preprocess(img) for img in images]
                    batch_tensor = torch.stack(tensors).to(device)
                    with torch.no_grad():
                        embeddings = model.encode_image(batch_tensor)
                        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                        embeddings_np = embeddings.cpu().float().numpy()
                    rows = [
                        (asset_id, struct.pack(f"{emb.shape[0]}f", *emb), now)
                        for asset_id, emb in zip(asset_ids, embeddings_np, strict=True)
                    ]
                    wconn.executemany(
                        "INSERT OR REPLACE INTO clip_embeddings"
                        " (asset_id, embedding, computed_at) VALUES (?, ?, ?)",
                        rows,
                    )
                    wconn.commit()
                    for relpath in relpaths:
                        _log.debug("CLIP inference done for %r", relpath)
                except Exception:  # noqa: BLE001
                    _log.warning("CLIP batch embedding failed", exc_info=True)
                finally:
                    for _ in batch:
                        _clip_queue.task_done()

                if saw_sentinel:
                    break
        finally:
            wconn.close()

    _clip_thread: threading.Thread | None = None
    if _clip_warmup_ok.is_set():
        _clip_thread = threading.Thread(
            target=_clip_batch_worker,
            daemon=True,
            name="clip-batch-embed",
        )
        _clip_thread.start()

    def _enqueue_clip(asset_id: int, relpath: str, img_rgb: object) -> None:
        if _clip_thread is None:
            return
        while True:
            if not _clip_thread.is_alive():
                return
            if cancel_check is not None and cancel_check():
                return
            try:
                _clip_queue.put((asset_id, relpath, img_rgb), timeout=0.25)
                return
            except queue.Full:
                continue

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
                    from takeout_rater.db.queries import (
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

        # Always regenerate the thumbnail for new assets, even if a file
        # already exists at the expected path.  When the user deletes the
        # database but keeps the thumbs directory, the old thumbnail files
        # remain on disk.  Because auto-increment IDs restart from 1, a
        # freshly assigned asset_id can collide with an ID that previously
        # belonged to a completely different photo, causing the stale
        # thumbnail to be served for the wrong asset.  Unconditionally
        # overwriting for new assets is cheap (one extra write) and
        # guarantees correctness after a database reset.
        if is_new or not thumb.exists():
            if file_bytes:
                try:
                    import io

                    from PIL import Image

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

        # Compute phash + CLIP for assets that are missing these records.
        # For new assets both are always absent.  For assets already in the DB
        # (is_new=False), use the pre-fetched sets built before the thread pool
        # started so that a previously-aborted indexing run — where the asset
        # row was committed but phash/CLIP were not saved — is corrected on the
        # next run rather than silently skipped.
        needs_phash = is_new or asset_id not in _ids_with_phash
        needs_clip = is_new or asset_id not in _ids_with_clip

        if needs_phash or needs_clip:
            if thumb_img is None and thumb.exists():
                try:
                    import io

                    from PIL import Image

                    thumb_img = Image.open(io.BytesIO(thumb.read_bytes()))
                except ImportError:
                    pass
                except Exception:
                    _log.debug(
                        "Could not load thumbnail for phash/CLIP %r",
                        relpath,
                        exc_info=True,
                    )

            if thumb_img is not None:
                if needs_phash:
                    # Compute phash from thumbnail.
                    try:
                        from takeout_rater.db.queries import (
                            upsert_phash,
                        )

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

                if needs_clip and _clip_warmup_ok.is_set():
                    # Queue CLIP embedding from thumbnail.  A dedicated worker
                    # batches these queued images for efficient GPU inference.
                    try:
                        img_rgb = thumb_img.convert("RGB").copy()
                        _enqueue_clip(asset_id, relpath, img_rgb)
                    except Exception:
                        _log.warning("CLIP queueing failed for %r", relpath, exc_info=True)

    # Submit all workers in parallel; _process_one swallows every exception
    # internally (logging it) so future.result() never re-raises and the
    # executor always shuts down cleanly.
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures: list[Future] = [executor.submit(_process_one, af) for af in assets]
        for future in as_completed(futures):
            future.result()

    if _clip_thread is not None:
        _clip_queue.put(_clip_sentinel)
        if _clip_thread.is_alive():
            _clip_queue.join()
            _clip_thread.join(timeout=30)

    progress.running = False
    progress.done = True
    if cancel_check is not None and cancel_check():
        progress.cancelled = True
    return progress
