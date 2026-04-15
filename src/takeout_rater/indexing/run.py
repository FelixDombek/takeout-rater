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
   * After claiming ownership: computes phash, CLIP embedding, and thumbnail
     all in parallel, no further locking required

The mutex guards only the minimal critical section required to prevent two
workers from both claiming the same sha256-identified asset.  Once a worker
has determined whether an asset is brand-new or a known hash (and recorded
its alias if needed), it proceeds with all the compute-heavy work (phash,
embedding, thumbnail) entirely in parallel with other workers.

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
        phase: Current phase — ``"scanning"`` while :func:`scan_takeout` is
            running; ``"processing"`` for the parallel per-file worker phase.
        total_dirs: Total number of directories to scan (filled during scan).
        dirs_scanned: Number of directories fully processed so far.
        current_dir: Name of the directory most recently processed.
    """

    running: bool = False
    done: bool = False
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
    library_root: Path,
    conn: sqlite3.Connection,
    on_progress: Callable[[IndexProgress], None] | None = None,
) -> IndexProgress:
    """Scan *library_root* and upsert discovered assets into *conn*.

    Processing happens in two phases (scan → processing) for maximum
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
    import threading  # noqa: PLC0415

    from takeout_rater.db.connection import library_state_dir, open_library_db  # noqa: PLC0415
    from takeout_rater.db.queries import lookup_sha256, upsert_asset  # noqa: PLC0415
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
    from takeout_rater.scoring.phash import DHASH_ALGO, compute_dhash_from_image  # noqa: PLC0415

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

    # ── Phase 1: Scan + concurrent model pre-load ─────────────────────────────
    # The filesystem walk is pure I/O — no file reads, no DB access.
    # We kick off a background thread to download/warm the CLIP backbone at
    # the same time so that model loading (which can take several minutes on a
    # first run) overlaps with the scan rather than blocking the processing
    # phase.
    #
    # _clip_warmup_ok is set to True ONLY after get_clip_model() returns
    # successfully.  Workers check this flag before attempting CLIP inference,
    # so they never compete for the model-loading lock.  If the warm-up times
    # out (e.g. because the model host is unreachable and the request hangs),
    # workers skip CLIP entirely rather than blocking indefinitely — which
    # would also prevent Ctrl-C from working (Python's ThreadPoolExecutor
    # atexit handler waits for all workers to finish).
    _clip_warmup_ok: list[bool] = [False]

    def _warmup_clip() -> None:
        try:
            from takeout_rater.scorers.adapters.clip_backbone import (  # noqa: PLC0415
                get_clip_model,
                is_available,
            )

            if is_available():
                get_clip_model()
                _clip_warmup_ok[0] = True  # only set after successful return
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
        _warmup_thread.join(timeout=300)
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

    thumbs_dir = library_state_dir(library_root) / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    now = int(time.time())
    num_workers = os.cpu_count() or 1

    # Single mutex guards only the critical section: lookup_sha256 + upsert_asset.
    # This prevents two workers from both claiming the same hash as "new".
    _claim_lock = threading.Lock()
    _progress_lock = threading.Lock()

    def _process_one(asset_file: object) -> None:
        """Process a single asset: hash, sidecar, claim, then phash+embed+thumb."""
        # Step 1: Read file bytes + compute sha256 (parallel, no locking)
        sha256: str | None = None
        file_bytes: bytes | None = None
        try:
            with open(asset_file.abspath, "rb") as f:  # type: ignore[union-attr]
                file_bytes = f.read()
            sha256 = hashlib.sha256(file_bytes).hexdigest()
        except OSError:
            pass

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

        # Step 3: Critical section – check hash + upsert (mutex-guarded)
        # Each worker opens its own DB connection inside the lock to avoid
        # sharing connections across threads.
        with _claim_lock:
            wconn = open_library_db(library_root)
            existing = lookup_sha256(wconn, sha256) if sha256 else None
            is_new = existing is None

            row: dict = {
                "relpath": asset_file.relpath,  # type: ignore[union-attr]
                "filename": Path(asset_file.relpath).name,  # type: ignore[union-attr]
                "ext": Path(asset_file.relpath).suffix.lower(),  # type: ignore[union-attr]
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
            wconn.close()

        # Update progress (guarded by separate lock)
        with _progress_lock:
            progress.indexed += 1
            if on_progress:
                on_progress(progress)

        # Step 4: If brand-new, compute phash + CLIP embedding + thumbnail
        # (all in parallel, no locking). If already known, just ensure
        # thumbnail exists.
        if is_new and file_bytes:
            # Compute phash from the in-memory image
            img = None
            try:
                import io  # noqa: PLC0415

                from PIL import Image  # noqa: PLC0415

                from takeout_rater.db.queries import upsert_phash  # noqa: PLC0415

                img = Image.open(io.BytesIO(file_bytes))
                dhash_hex = compute_dhash_from_image(img)
                # Store in DB (each worker gets its own connection)
                wconn2 = open_library_db(library_root)
                upsert_phash(wconn2, asset_id, dhash_hex, DHASH_ALGO)
                wconn2.close()
            except (ImportError, OSError):
                pass

            # Compute CLIP embedding if warm-up confirmed the model loaded.
            # Checking _clip_warmup_ok here ensures workers never contend for
            # the model-loading lock; get_clip_model() returns the cached
            # singleton immediately once the flag is True.
            if img is not None and _clip_warmup_ok[0]:
                try:
                    import struct  # noqa: PLC0415

                    import torch  # noqa: PLC0415

                    from takeout_rater.scorers.adapters.clip_backbone import (
                        get_clip_model,  # noqa: PLC0415
                    )

                    model, preprocess, _tokenizer, device = get_clip_model()
                    img_rgb = img.convert("RGB")
                    img_tensor = preprocess(img_rgb).unsqueeze(0).to(device)
                    with torch.no_grad():
                        embedding = model.encode_image(img_tensor)
                        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                        embedding = embedding.cpu().float().numpy()[0]
                    blob = struct.pack(f"{embedding.shape[0]}f", *embedding)
                    wconn3 = open_library_db(library_root)
                    wconn3.execute(
                        "INSERT OR REPLACE INTO clip_embeddings (asset_id, embedding) VALUES (?, ?)",
                        (asset_id, blob),
                    )
                    wconn3.commit()
                    wconn3.close()
                except (ImportError, OSError, RuntimeError):
                    pass

        # Always ensure thumbnail exists (for both new and known assets)
        thumb = thumb_path_for_id(thumbs_dir, asset_id)
        if not thumb.exists():
            with contextlib.suppress(ImportError, OSError):
                generate_thumbnail(asset_file.abspath, thumb)  # type: ignore[union-attr]

    # Submit all workers in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures: list[Future] = [executor.submit(_process_one, af) for af in assets]
        for future in as_completed(futures):
            # Just wait for completion; errors are suppressed inside _process_one
            future.result()

    progress.running = False
    progress.done = True
    return progress
