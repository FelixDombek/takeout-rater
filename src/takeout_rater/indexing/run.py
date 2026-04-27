"""Reusable indexing function callable from both the CLI and the web API.

This module provides :func:`run_index` which processes a Google Photos
Takeout directory in two distinct phases:

1. **Scan phase** – pure filesystem walk; collects file-stat metadata into an
   in-memory list without opening any image file or reading any sidecar JSON.
   This is extremely fast and produces no database I/O.

2. **Processing phase** – fully parallel per-file workers.  Each worker:

   * Reads file bytes and computes SHA-256 in parallel
   * Parses the sidecar JSON in parallel (if present)
   * Sends asset claims to one dedicated SQLite writer thread, which serialises
     and commits each hash lookup/upsert/alias decision atomically.
   * After claiming ownership: computes thumbnail and pHash in parallel, then
     queues the thumbnail for a dedicated batched CLIP embedding worker

The writer keeps the hash-claim decision atomic without opening a new SQLite
connection for every asset.  pHash writes and CLIP inference are batched on
dedicated workers; CLIP batching avoids the PyTorch thread-pool deadlock that
occurs when multiple Python threads call ``encode_image()`` concurrently while
still giving the GPU real image batches.  All other per-asset work (SHA-256,
sidecar parsing, phash, thumbnails) proceeds fully in parallel.

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
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
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
    diagnostics: dict[str, int | float | str] = field(default_factory=dict)

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
    clip_batch_size: int | None = None,
    clip_accelerator: str | None = None,
    clip_fp16: bool | None = None,
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
    from takeout_rater.indexing.scanner import scan_photos_tree
    from takeout_rater.indexing.sidecar import parse_sidecar
    from takeout_rater.indexing.thumbnailer import (
        generate_thumbnail_from_image_timed,
        generate_thumbnail_from_jpeg_bytes_nvjpeg_timed,
        generate_thumbnail_from_path_timed,
        nvjpeg_enabled,
        thumb_path_for_id,
    )

    if db_root is None:
        raise ValueError("db_root is required")

    progress = IndexProgress(running=True)
    _diagnostics_lock = threading.Lock()
    _processing_start: float | None = None
    _diagnostics: dict[str, int | float | str] = {
        "assets_found": 0,
        "assets_indexed": 0,
        "assets_new": 0,
        "assets_existing": 0,
        "index_workers": 0,
        "thumbnails_generated": 0,
        "phashes_computed": 0,
        "clip_embeddings_computed": 0,
        "clip_batches": 0,
        "clip_batch_items": 0,
        "clip_batch_size": 0,
        "clip_batch_last_size": 0,
        "clip_batch_max_size": 0,
        "clip_batch_fill_wait_seconds": 0.0,
        "clip_batch_inference_last_seconds": 0.0,
        "clip_batch_inference_max_seconds": 0.0,
        "clip_accelerator": "",
        "clip_status": "",
        "clip_providers": "",
        "clip_preprocess_workers": 0,
        "clip_image_queue_size": 0,
        "clip_image_queue_max": 0,
        "clip_queue_wait_seconds": 0.0,
        "clip_tensor_queue_size": 0,
        "clip_tensor_queue_max": 0,
        "clip_inference_active_seconds": 0.0,
        "clip_first_inference_seconds": 0.0,
        "scan_seconds": 0.0,
        "clip_warmup_wait_seconds": 0.0,
        "processing_seconds": 0.0,
        "sha_seconds": 0.0,
        "sidecar_seconds": 0.0,
        "db_claim_seconds": 0.0,
        "db_claim_enqueue_wait_seconds": 0.0,
        "db_claim_writer_seconds": 0.0,
        "db_claim_commit_seconds": 0.0,
        "db_claim_future_wait_seconds": 0.0,
        "db_claims_committed": 0,
        "phash_write_seconds": 0.0,
        "phash_commits": 0,
        "thumbnail_seconds": 0.0,
        "thumbnail_decode_seconds": 0.0,
        "thumbnail_decode_bytes_seconds": 0.0,
        "thumbnail_decode_nvjpeg_seconds": 0.0,
        "thumbnail_nvjpeg_decoder_seconds": 0.0,
        "thumbnail_nvjpeg_decoder_init_seconds": 0.0,
        "thumbnail_nvjpeg_decoder_inits": 0,
        "thumbnail_decode_path_seconds": 0.0,
        "thumbnail_resize_seconds": 0.0,
        "thumbnail_write_seconds": 0.0,
        "thumbnails_from_bytes": 0,
        "thumbnails_from_nvjpeg": 0,
        "thumbnail_nvjpeg_fallbacks": 0,
        "thumbnails_from_path": 0,
        "phash_seconds": 0.0,
        "clip_preprocess_seconds": 0.0,
        "clip_tensor_queue_wait_seconds": 0.0,
        "clip_onnx_input_seconds": 0.0,
        "clip_onnx_run_seconds": 0.0,
        "clip_onnx_slow_fallbacks": 0,
        "clip_inference_seconds": 0.0,
        "clip_write_seconds": 0.0,
    }

    def _diag_snapshot() -> dict[str, int | float | str]:
        with _diagnostics_lock:
            snapshot = dict(_diagnostics)
        if _processing_start is not None:
            snapshot["processing_seconds"] = time.perf_counter() - _processing_start
        assets_indexed = int(snapshot.get("assets_indexed", 0))
        thumbs = int(snapshot.get("thumbnails_generated", 0))
        phashes = int(snapshot.get("phashes_computed", 0))
        clips = int(snapshot.get("clip_embeddings_computed", 0))
        clip_batches = int(snapshot.get("clip_batches", 0))
        infer_started = float(snapshot.get("clip_inference_started_at", 0.0) or 0.0)
        if infer_started:
            snapshot["clip_inference_active_seconds"] = time.perf_counter() - infer_started
        if assets_indexed:
            snapshot["sha_ms_per_asset"] = (
                float(snapshot.get("sha_seconds", 0.0)) * 1000.0 / assets_indexed
            )
            snapshot["sidecar_ms_per_asset"] = (
                float(snapshot.get("sidecar_seconds", 0.0)) * 1000.0 / assets_indexed
            )
            snapshot["db_claim_ms_per_asset"] = (
                float(snapshot.get("db_claim_seconds", 0.0)) * 1000.0 / assets_indexed
            )
            snapshot["db_claim_writer_ms_per_asset"] = (
                float(snapshot.get("db_claim_writer_seconds", 0.0)) * 1000.0 / assets_indexed
            )
            snapshot["db_claim_commit_ms_per_asset"] = (
                float(snapshot.get("db_claim_commit_seconds", 0.0)) * 1000.0 / assets_indexed
            )
        if thumbs:
            snapshot["thumbnail_ms_per_thumb"] = (
                float(snapshot.get("thumbnail_seconds", 0.0)) * 1000.0 / thumbs
            )
        if phashes:
            snapshot["phash_ms_per_hash"] = (
                float(snapshot.get("phash_seconds", 0.0)) * 1000.0 / phashes
            )
        if clips:
            snapshot["clip_inference_ms_per_embedding"] = (
                float(snapshot.get("clip_inference_seconds", 0.0)) * 1000.0 / clips
            )
        if clip_batches:
            snapshot["clip_batch_avg_size"] = (
                float(snapshot.get("clip_batch_items", 0.0)) / clip_batches
            )
            snapshot["clip_inference_ms_per_batch"] = (
                float(snapshot.get("clip_inference_seconds", 0.0)) * 1000.0 / clip_batches
            )
        return snapshot

    def _diag_set(key: str, value: int | float | str) -> None:
        with _diagnostics_lock:
            _diagnostics[key] = value
            progress.diagnostics = dict(_diagnostics)

    def _diag_add(key: str, seconds: float) -> None:
        with _diagnostics_lock:
            _diagnostics[key] = float(_diagnostics.get(key, 0.0)) + seconds
            progress.diagnostics = dict(_diagnostics)

    def _diag_inc(key: str, amount: int = 1) -> None:
        with _diagnostics_lock:
            _diagnostics[key] = int(_diagnostics.get(key, 0)) + amount
            progress.diagnostics = dict(_diagnostics)

    def _diag_max(key: str, value: int | float) -> None:
        with _diagnostics_lock:
            current = _diagnostics.get(key, 0)
            if not isinstance(current, (int, float)) or value > current:
                _diagnostics[key] = value
            progress.diagnostics = dict(_diagnostics)

    progress.diagnostics = _diag_snapshot()

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
        progress.diagnostics = _diag_snapshot()
        if on_progress:
            on_progress(progress)

    _scan_start = time.perf_counter()
    assets = scan_photos_tree(photos_root, on_dir_scanned=_on_dir_scanned)
    _diag_add("scan_seconds", time.perf_counter() - _scan_start)
    _diag_set("assets_found", len(assets))
    progress.found = len(assets)

    if not assets:
        # No assets to index; the warmup thread is a daemon and will be cleaned
        # up by the process, but attempt a very short join in case it already
        # finished to avoid leaving unnecessary background work.
        _warmup_thread.join(timeout=0.5)
        progress.running = False
        progress.done = True
        progress.diagnostics = _diag_snapshot()
        return progress

    # ── Phase 1b: Wait for model warm-up if still loading ─────────────────────
    # If the scan finished before the model loaded, block here (up to 5 min)
    # so that workers skip CLIP rather than block.  The UI shows
    # "Loading CLIP model…" during this window.
    if _warmup_thread.is_alive():
        progress.phase = "loading_models"
        progress.diagnostics = _diag_snapshot()
        if on_progress:
            on_progress(progress)
        _warmup_wait_start = time.perf_counter()
        _warmup_thread.join(timeout=300)
        _diag_add("clip_warmup_wait_seconds", time.perf_counter() - _warmup_wait_start)

    # ── Phase 2: Processing ───────────────────────────────────────────────────
    progress.phase = "processing"
    progress.diagnostics = _diag_snapshot()
    if on_progress:
        on_progress(progress)
    _processing_start = time.perf_counter()

    thumbs_dir = library_state_dir(db_root) / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    db_path = library_db_path(db_root)
    now = int(time.time())
    num_workers = os.cpu_count() or 1
    _diag_set("index_workers", num_workers)

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

    _progress_lock = threading.Lock()
    # CLIP image preprocessing runs on a small CPU worker pool and feeds one
    # dedicated GPU/DB writer thread that forms inference batches. Calling
    # encode_image() from many Python workers can deadlock PyTorch's thread
    # pool, but a single inference thread with preprocessed tensors keeps the
    # GPU path busy without putting CPU transforms on its critical path.
    _clip_batch_size = clip_batch_size or int(
        os.environ.get("TAKEOUT_RATER_CLIP_BATCH_SIZE", "128")
    )
    _clip_accelerator = (
        clip_accelerator or os.environ.get("TAKEOUT_RATER_CLIP_ACCELERATOR", "torch")
    ).lower()
    if _clip_accelerator not in {"auto", "tensorrt", "onnx", "cuda", "torch"}:
        _clip_accelerator = "torch"
    _clip_preprocess_workers = max(1, min(4, num_workers // 2))
    _clip_batch_fill_wait_seconds = max(
        0.02,
        float(os.environ.get("TAKEOUT_RATER_CLIP_BATCH_FILL_WAIT_SECONDS", "0.35")),
    )
    _clip_image_queue_max = max(_clip_batch_size, _clip_preprocess_workers * 8)
    _clip_tensor_queue_max = max(_clip_batch_size * 2, _clip_preprocess_workers * 8)
    _clip_image_queue: queue.Queue[object] = queue.Queue(maxsize=_clip_image_queue_max)
    _clip_tensor_queue: queue.Queue[object] = queue.Queue(maxsize=_clip_tensor_queue_max)
    _clip_image_sentinel = object()
    _clip_tensor_sentinel = object()
    _diag_set("clip_batch_size", _clip_batch_size)
    _diag_set("clip_preprocess_workers", _clip_preprocess_workers)
    _diag_set("clip_image_queue_max", _clip_image_queue_max)
    _diag_set("clip_tensor_queue_max", _clip_tensor_queue_max)

    def _diag_clip_queue_sizes() -> None:
        _diag_set("clip_image_queue_size", _clip_image_queue.qsize())
        _diag_set("clip_tensor_queue_size", _clip_tensor_queue.qsize())

    def _clip_batch_worker() -> None:
        import struct

        try:
            import torch

            from takeout_rater.scoring.scorers.clip_backbone import (
                CLIP_IMAGE_INPUT_NAME,
                CLIP_IMAGE_OUTPUT_NAME,
                get_clip_model,
                get_clip_onnx_session,
                preprocess_clip_image_fast,
            )
        except ImportError:
            return

        onnx_session = None
        model = None
        device = None
        try:
            if _clip_accelerator != "torch":
                _diag_set("clip_status", "initializing ONNX/TensorRT session")
                try:
                    onnx_bundle = get_clip_onnx_session(
                        library_state_dir(db_root) / "onnxruntime-clip",
                        accelerator=_clip_accelerator,
                    )
                except Exception as exc:  # noqa: BLE001
                    _log.warning(
                        "CLIP ONNX/TensorRT setup failed; falling back to Torch "
                        "(%s: %s). Select Torch in setup to skip ONNX export attempts.",
                        type(exc).__name__,
                        exc,
                        exc_info=_log.isEnabledFor(logging.DEBUG),
                    )
                    onnx_bundle = None
                if onnx_bundle is not None:
                    onnx_session, _preprocess, onnx_providers = onnx_bundle
                    _diag_set("clip_providers", ", ".join(onnx_providers))
                    if any(p == "TensorrtExecutionProvider" for p in onnx_providers):
                        _diag_set("clip_accelerator", "tensorrt")
                        _diag_set(
                            "clip_status", "TensorRT session ready; first batch may build engine"
                        )
                    elif any(p == "CUDAExecutionProvider" for p in onnx_providers):
                        _diag_set("clip_accelerator", "onnx-cuda")
                        _diag_set("clip_status", "ONNX Runtime CUDA session ready")
                    else:
                        onnx_session = None
            if onnx_session is None:
                _diag_set("clip_status", "loading Torch CLIP model")
                model, _preprocess, _tokenizer, device = get_clip_model()
                _diag_set(
                    "clip_accelerator", "torch-cuda" if str(device).startswith("cuda") else "torch"
                )
                _diag_set("clip_status", "Torch CLIP model ready")
        except Exception:  # noqa: BLE001
            _log.warning("CLIP batch worker failed to load model", exc_info=True)
            return

        def _is_cuda_oom(exc: BaseException) -> bool:
            return "out of memory" in str(exc).lower()

        def _encode_tensors(tensors: list[object]) -> object:
            import numpy as np

            nonlocal device, model, onnx_session

            if onnx_session is not None:
                try:
                    _onnx_input_start = time.perf_counter()
                    batch_tensor = torch.stack(tensors).contiguous()
                    embeddings_np = None
                    onnx_providers = onnx_session.get_providers()
                    is_tensorrt_session = "TensorrtExecutionProvider" in onnx_providers
                    slow_onnx_cuda_batch_seconds = max(
                        1.0,
                        float(
                            os.environ.get(
                                "TAKEOUT_RATER_ONNX_CUDA_SLOW_BATCH_SECONDS",
                                "10",
                            )
                        ),
                    )
                    use_cuda_binding = (
                        torch.cuda.is_available()
                        and "CPUExecutionProvider" not in onnx_providers[:1]
                    )
                    if use_cuda_binding:
                        try:
                            cuda_tensor = batch_tensor.pin_memory().to(
                                "cuda",
                                non_blocking=True,
                            )
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            _diag_add(
                                "clip_onnx_input_seconds",
                                time.perf_counter() - _onnx_input_start,
                            )
                            first_onnx_batch = int(_diagnostics.get("clip_batches", 0)) == 0
                            if first_onnx_batch:
                                _diag_set("clip_status", "TensorRT first inference / engine build")
                            else:
                                _diag_set("clip_status", "ONNX/TensorRT inference")
                            _clip_infer_start = time.perf_counter()
                            _diag_set("clip_inference_started_at", _clip_infer_start)
                            io_binding = onnx_session.io_binding()
                            io_binding.bind_input(
                                name=CLIP_IMAGE_INPUT_NAME,
                                device_type="cuda",
                                device_id=0,
                                element_type=np.float32,
                                shape=tuple(cuda_tensor.shape),
                                buffer_ptr=cuda_tensor.data_ptr(),
                            )
                            io_binding.bind_output(CLIP_IMAGE_OUTPUT_NAME, "cpu")
                            _onnx_run_start = time.perf_counter()
                            onnx_session.run_with_iobinding(io_binding)
                            _diag_add(
                                "clip_onnx_run_seconds",
                                time.perf_counter() - _onnx_run_start,
                            )
                            embeddings_np = io_binding.copy_outputs_to_cpu()[0]
                        except Exception:  # noqa: BLE001
                            _log.warning(
                                "CLIP ONNX CUDA I/O binding failed; falling back to CPU input",
                                exc_info=True,
                            )
                    if embeddings_np is None:
                        batch_np = batch_tensor.cpu().float().numpy()
                        _diag_add(
                            "clip_onnx_input_seconds",
                            time.perf_counter() - _onnx_input_start,
                        )
                        first_onnx_batch = int(_diagnostics.get("clip_batches", 0)) == 0
                        if first_onnx_batch:
                            _diag_set("clip_status", "TensorRT first inference / engine build")
                        else:
                            _diag_set("clip_status", "ONNX/TensorRT inference")
                        _clip_infer_start = time.perf_counter()
                        _diag_set("clip_inference_started_at", _clip_infer_start)
                        _onnx_run_start = time.perf_counter()
                        embeddings_np = onnx_session.run(
                            None,
                            {CLIP_IMAGE_INPUT_NAME: batch_np},
                        )[0]
                        _diag_add(
                            "clip_onnx_run_seconds",
                            time.perf_counter() - _onnx_run_start,
                        )
                    _diag_set("clip_inference_started_at", 0.0)
                    if first_onnx_batch:
                        _diag_set(
                            "clip_first_inference_seconds",
                            time.perf_counter() - _clip_infer_start,
                        )
                        _diag_set("clip_status", "ONNX/TensorRT inference ready")
                    norms = np.linalg.norm(embeddings_np, axis=-1, keepdims=True)
                    embeddings_np = embeddings_np / np.maximum(norms, 1e-12)
                    _onnx_infer_seconds = time.perf_counter() - _clip_infer_start
                    _diag_add("clip_inference_seconds", _onnx_infer_seconds)
                    _diag_inc("clip_batches")
                    if (
                        not is_tensorrt_session
                        and _onnx_infer_seconds >= slow_onnx_cuda_batch_seconds
                    ):
                        _log.warning(
                            "CLIP ONNX CUDA batch took %.1fs for %d image(s); "
                            "disabling ONNX CUDA for the rest of this indexing run "
                            "and falling back to Torch",
                            _onnx_infer_seconds,
                            len(tensors),
                        )
                        _diag_inc("clip_onnx_slow_fallbacks")
                        _diag_set("clip_status", "ONNX CUDA slow; falling back to Torch")
                        onnx_session = None
                    return embeddings_np.astype("float32", copy=False)
                except Exception:  # noqa: BLE001
                    _log.warning(
                        "CLIP ONNX inference failed; disabling ONNX for this indexing run "
                        "and falling back to Torch",
                        exc_info=_log.isEnabledFor(logging.DEBUG),
                    )
                    _diag_set("clip_inference_started_at", 0.0)
                    _diag_set("clip_status", "ONNX/TensorRT failed; falling back to Torch")
                    onnx_session = None

            try:
                if model is None or device is None:
                    _diag_set("clip_status", "loading Torch CLIP model")
                    model, _preprocess, _tokenizer, device = get_clip_model()
                    _diag_set(
                        "clip_accelerator",
                        "torch-cuda" if str(device).startswith("cuda") else "torch",
                    )
                    _diag_set("clip_status", "Torch CLIP model ready")
                batch_tensor = torch.stack(tensors)
                if str(device).startswith("cuda"):
                    batch_tensor = batch_tensor.pin_memory().to(device, non_blocking=True)
                else:
                    batch_tensor = batch_tensor.to(device)
                _clip_infer_start = time.perf_counter()
                _diag_set("clip_status", "Torch CLIP inference")
                _diag_set("clip_inference_started_at", _clip_infer_start)
                with torch.inference_mode():
                    use_fp16 = str(device).startswith("cuda") and (
                        clip_fp16
                        if clip_fp16 is not None
                        else os.environ.get("TAKEOUT_RATER_CLIP_FP16", "1") != "0"
                    )
                    if use_fp16:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            embeddings = model.encode_image(batch_tensor)
                    else:
                        embeddings = model.encode_image(batch_tensor)
                    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                    embeddings_np = embeddings.cpu().float().numpy()
                _diag_set("clip_inference_started_at", 0.0)
                _diag_set("clip_status", "Torch CLIP inference ready")
                _diag_add("clip_inference_seconds", time.perf_counter() - _clip_infer_start)
                _diag_inc("clip_batches")
                return embeddings_np
            except RuntimeError as exc:
                if len(tensors) <= 1 or not _is_cuda_oom(exc):
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                mid = len(tensors) // 2
                left = _encode_tensors(tensors[:mid])
                right = _encode_tensors(tensors[mid:])
                return np.concatenate([left, right], axis=0)

        def _clip_preprocess_worker() -> None:
            while True:
                item = _clip_image_queue.get()
                _diag_clip_queue_sizes()
                if item is _clip_image_sentinel:
                    _clip_image_queue.task_done()
                    _clip_tensor_queue.put(_clip_tensor_sentinel)
                    _diag_clip_queue_sizes()
                    break
                asset_id, relpath, img = item  # type: ignore[misc]
                try:
                    _clip_preprocess_start = time.perf_counter()
                    tensor = preprocess_clip_image_fast(img)
                    _diag_add(
                        "clip_preprocess_seconds",
                        time.perf_counter() - _clip_preprocess_start,
                    )
                    _tensor_queue_wait_start = time.perf_counter()
                    _clip_tensor_queue.put((asset_id, relpath, tensor))
                    _diag_clip_queue_sizes()
                    _diag_add(
                        "clip_tensor_queue_wait_seconds",
                        time.perf_counter() - _tensor_queue_wait_start,
                    )
                except Exception:  # noqa: BLE001
                    _log.warning("CLIP preprocessing failed for %r", relpath, exc_info=True)
                finally:
                    _clip_image_queue.task_done()

        preprocess_threads = [
            threading.Thread(
                target=_clip_preprocess_worker,
                daemon=True,
                name=f"clip-preprocess-{idx + 1}",
            )
            for idx in range(_clip_preprocess_workers)
        ]
        for thread in preprocess_threads:
            thread.start()

        wconn = open_db(db_path)
        try:
            preprocess_sentinels_seen = 0
            while True:
                item = _clip_tensor_queue.get()
                _diag_clip_queue_sizes()
                if item is _clip_tensor_sentinel:
                    _clip_tensor_queue.task_done()
                    preprocess_sentinels_seen += 1
                    if preprocess_sentinels_seen >= _clip_preprocess_workers:
                        break
                    continue

                batch = [item]
                _batch_fill_start = time.perf_counter()
                _batch_fill_deadline = _batch_fill_start + _clip_batch_fill_wait_seconds
                while len(batch) < _clip_batch_size:
                    if preprocess_sentinels_seen >= _clip_preprocess_workers:
                        break
                    remaining = _batch_fill_deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    try:
                        next_item = _clip_tensor_queue.get(timeout=min(0.05, remaining))
                        _diag_clip_queue_sizes()
                    except queue.Empty:
                        continue
                    if next_item is _clip_tensor_sentinel:
                        _clip_tensor_queue.task_done()
                        preprocess_sentinels_seen += 1
                        continue
                    batch.append(next_item)

                if not batch:
                    if preprocess_sentinels_seen >= _clip_preprocess_workers:
                        break
                    continue

                try:
                    asset_ids = [entry[0] for entry in batch]  # type: ignore[index]
                    relpaths = [entry[1] for entry in batch]  # type: ignore[index]
                    tensors = [entry[2] for entry in batch]  # type: ignore[index]
                    _log.debug("CLIP batch inference start: %d tensor(s)", len(tensors))
                    _diag_add(
                        "clip_batch_fill_wait_seconds",
                        time.perf_counter() - _batch_fill_start,
                    )
                    _batch_infer_start = time.perf_counter()
                    embeddings_np = _encode_tensors(tensors)
                    _batch_infer_seconds = time.perf_counter() - _batch_infer_start
                    _diag_set("clip_batch_inference_last_seconds", _batch_infer_seconds)
                    _diag_max("clip_batch_inference_max_seconds", _batch_infer_seconds)
                    rows = [
                        (asset_id, struct.pack(f"{emb.shape[0]}f", *emb), now)
                        for asset_id, emb in zip(asset_ids, embeddings_np, strict=True)
                    ]
                    _clip_write_start = time.perf_counter()
                    wconn.executemany(
                        "INSERT OR REPLACE INTO clip_embeddings"
                        " (asset_id, embedding, computed_at) VALUES (?, ?, ?)",
                        rows,
                    )
                    wconn.commit()
                    _diag_add("clip_write_seconds", time.perf_counter() - _clip_write_start)
                    _diag_inc("clip_embeddings_computed", len(rows))
                    _diag_inc("clip_batch_items", len(rows))
                    _diag_set("clip_batch_last_size", len(rows))
                    _diag_max("clip_batch_max_size", len(rows))
                    for relpath in relpaths:
                        _log.debug("CLIP inference done for %r", relpath)
                except Exception:  # noqa: BLE001
                    _log.warning("CLIP batch embedding failed", exc_info=True)
                finally:
                    for _ in batch:
                        _clip_tensor_queue.task_done()

                if preprocess_sentinels_seen >= _clip_preprocess_workers:
                    break
        finally:
            for thread in preprocess_threads:
                thread.join(timeout=5)
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
        _queue_wait_start = time.perf_counter()
        while True:
            if not _clip_thread.is_alive():
                _diag_add("clip_queue_wait_seconds", time.perf_counter() - _queue_wait_start)
                return
            if cancel_check is not None and cancel_check():
                _diag_add("clip_queue_wait_seconds", time.perf_counter() - _queue_wait_start)
                return
            try:
                _clip_image_queue.put((asset_id, relpath, img_rgb), timeout=0.25)
                _diag_clip_queue_sizes()
                _diag_add("clip_queue_wait_seconds", time.perf_counter() - _queue_wait_start)
                return
            except queue.Full:
                _diag_clip_queue_sizes()
                continue

    # SQLite writes are serialized through one writer thread. Asset claims are
    # still decided and committed one by one on that connection to preserve the
    # hash lookup/upsert atomicity; pHash writes are batched behind them.
    _phash_write_batch_size = 256
    _phash_write_batch_wait = 0.005
    _db_queue: queue.Queue[object] = queue.Queue(maxsize=max(256, num_workers * 8))
    _db_sentinel = object()

    def _upsert_asset_claim_no_commit(wconn: sqlite3.Connection, asset: dict) -> tuple[int, bool]:
        asset = dict(asset)
        sha256 = asset.get("sha256")
        relpath = asset["relpath"]
        indexed_at = asset["indexed_at"]
        existing_relpath = wconn.execute(
            "SELECT id FROM assets WHERE relpath = ? LIMIT 1", (relpath,)
        ).fetchone()
        is_new_row = existing_relpath is None

        if sha256:
            existing = wconn.execute(
                "SELECT id, relpath FROM assets WHERE sha256 = ? ORDER BY id LIMIT 1",
                (sha256,),
            ).fetchone()
            if existing:
                canonical_id = int(existing[0])
                canonical_relpath = str(existing[1])
                if relpath != canonical_relpath:
                    wconn.execute(
                        "INSERT INTO asset_paths (asset_id, relpath, indexed_at)"
                        " VALUES (?, ?, ?)"
                        " ON CONFLICT(relpath) DO UPDATE"
                        "   SET asset_id = excluded.asset_id, indexed_at = excluded.indexed_at",
                        (canonical_id, relpath, indexed_at),
                    )
                    return canonical_id, False
                is_new_row = False

        wconn.execute("DELETE FROM asset_paths WHERE relpath = ?", (relpath,))
        insert_cols = [k for k in asset if k != "id"]
        columns = ", ".join(insert_cols)
        placeholders = ", ".join("?" for _ in insert_cols)
        update_pairs = ", ".join(f"{k} = excluded.{k}" for k in insert_cols if k != "relpath")
        sql = (
            f"INSERT INTO assets ({columns}) VALUES ({placeholders})"  # noqa: S608
            f" ON CONFLICT(relpath) DO UPDATE SET {update_pairs}"
            " RETURNING id"
        )
        row = wconn.execute(sql, [asset[k] for k in insert_cols]).fetchone()
        return int(row[0]), is_new_row

    def _upsert_album_no_commit(wconn: sqlite3.Connection, name: str, relpath: str) -> int:
        row = wconn.execute(
            "INSERT INTO albums (name, relpath, indexed_at)"
            " VALUES (?, ?, ?)"
            " ON CONFLICT(relpath) DO UPDATE SET name = excluded.name, indexed_at = excluded.indexed_at"
            " RETURNING id",
            (name, relpath, now),
        ).fetchone()
        return int(row[0])

    def _link_asset_to_album_no_commit(
        wconn: sqlite3.Connection, album_id: int, asset_id: int
    ) -> None:
        wconn.execute(
            "INSERT OR IGNORE INTO album_assets (album_id, asset_id) VALUES (?, ?)",
            (album_id, asset_id),
        )

    def _upsert_phash_no_commit(wconn: sqlite3.Connection, asset_id: int, dhash_hex: str) -> None:
        wconn.execute(
            "INSERT OR REPLACE INTO phash (asset_id, phash_hex, algo, computed_at)"
            " VALUES (?, ?, ?, ?)",
            (asset_id, dhash_hex, DHASH_ALGO, int(time.time())),
        )

    def _handle_claim_item(wconn: sqlite3.Connection, item: object) -> None:
        future = item["future"]  # type: ignore[index]
        try:
            writer_start = time.perf_counter()
            asset_id, is_new = _upsert_asset_claim_no_commit(wconn, item["asset"])  # type: ignore[index]
            for relpath_for_album in item["album_relpaths"]:  # type: ignore[index]
                parts = Path(relpath_for_album).parts
                if len(parts) > 1:
                    album_name = parts[0]
                    album_id = _upsert_album_no_commit(wconn, album_name, album_name)
                    _link_asset_to_album_no_commit(wconn, album_id, asset_id)
            _diag_add("db_claim_writer_seconds", time.perf_counter() - writer_start)
            commit_start = time.perf_counter()
            wconn.commit()
            _diag_add("db_claim_commit_seconds", time.perf_counter() - commit_start)
            _diag_inc("db_claims_committed")
            future.set_result((asset_id, is_new))
        except Exception as exc:  # noqa: BLE001
            wconn.rollback()
            future.set_exception(exc)

    def _handle_db_item(wconn: sqlite3.Connection, item: object) -> None:
        kind = item["kind"]  # type: ignore[index]
        if kind == "claim":
            _handle_claim_item(wconn, item)
        elif kind == "phash":
            _upsert_phash_no_commit(wconn, item["asset_id"], item["dhash_hex"])  # type: ignore[index]

    def _db_writer() -> None:
        wconn = open_db(db_path)
        try:
            while True:
                item = _db_queue.get()
                if item is _db_sentinel:
                    _db_queue.task_done()
                    break
                if item["kind"] == "claim":  # type: ignore[index]
                    try:
                        _handle_claim_item(wconn, item)
                    except Exception:  # noqa: BLE001
                        _log.warning("Index DB writer claim failed", exc_info=True)
                    finally:
                        _db_queue.task_done()
                    continue

                batch = [item]
                saw_sentinel = False
                while len(batch) < _phash_write_batch_size:
                    try:
                        next_item = _db_queue.get(timeout=_phash_write_batch_wait)
                    except queue.Empty:
                        break
                    if next_item is _db_sentinel:
                        _db_queue.task_done()
                        saw_sentinel = True
                        break
                    if next_item["kind"] == "claim":  # type: ignore[index]
                        _handle_claim_item(wconn, next_item)
                        _db_queue.task_done()
                        continue
                    batch.append(next_item)

                try:
                    writer_start = time.perf_counter()
                    for batch_item in batch:
                        _handle_db_item(wconn, batch_item)
                    _diag_add("phash_write_seconds", time.perf_counter() - writer_start)
                    commit_start = time.perf_counter()
                    wconn.commit()
                    _diag_add("phash_write_seconds", time.perf_counter() - commit_start)
                    _diag_inc("phash_commits")
                except Exception:  # noqa: BLE001
                    wconn.rollback()
                    _log.warning("Index DB writer batch failed", exc_info=True)
                finally:
                    for _ in batch:
                        _db_queue.task_done()

                if saw_sentinel:
                    break
        finally:
            wconn.close()

    _db_thread = threading.Thread(target=_db_writer, daemon=True, name="index-db-writer")
    _db_thread.start()
    _use_nvjpeg = nvjpeg_enabled()

    def _claim_asset(asset_row: dict, album_relpaths: list[str]) -> tuple[int, bool]:
        future: Future = Future()
        enqueue_start = time.perf_counter()
        _db_queue.put(
            {
                "kind": "claim",
                "asset": asset_row,
                "album_relpaths": album_relpaths,
                "future": future,
            }
        )
        _diag_add("db_claim_enqueue_wait_seconds", time.perf_counter() - enqueue_start)
        future_start = time.perf_counter()
        result = future.result()
        _diag_add("db_claim_future_wait_seconds", time.perf_counter() - future_start)
        return result

    def _enqueue_phash(asset_id: int, dhash_hex: str) -> None:
        _db_queue.put({"kind": "phash", "asset_id": asset_id, "dhash_hex": dhash_hex})

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
        _sha_start = time.perf_counter()
        try:
            with open(asset_file.abspath, "rb") as f:  # type: ignore[union-attr]
                file_bytes = f.read()
            sha256 = hashlib.sha256(file_bytes).hexdigest()
        except OSError:
            _log.debug("Could not read file %r – skipping SHA-256", relpath)
        finally:
            _diag_add("sha_seconds", time.perf_counter() - _sha_start)

        # Step 2: Parse sidecar if present (parallel, no locking)
        sidecar_updates: dict = {}
        _sidecar_start = time.perf_counter()
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
        _diag_add("sidecar_seconds", time.perf_counter() - _sidecar_start)

        # Step 3: Critical section – check hash + upsert (mutex-guarded).
        # Use open_db (no migrations) since the DB is already initialised.
        # Each iteration opens and immediately closes a short-lived connection
        # so that no connection is shared across threads.
        is_new: bool = False
        asset_id: int = 0
        _log.debug("Claiming asset %r (sha256=%s)", relpath, sha256 and sha256[:8])
        _claim_start = time.perf_counter()
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
        asset_id, is_new = _claim_asset(row, [relpath])
        _diag_add("db_claim_seconds", time.perf_counter() - _claim_start)
        _diag_inc("assets_new" if is_new else "assets_existing")
        _log.debug("Claimed asset %r → id=%d is_new=%s", relpath, asset_id, is_new)

        # Update progress (guarded by separate lock)
        with _progress_lock:
            progress.indexed += 1
            _diag_set("assets_indexed", progress.indexed)
            progress.diagnostics = _diag_snapshot()
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
            _thumb_start = time.perf_counter()
            generated_thumb = False
            if file_bytes:
                is_jpeg = (
                    Path(relpath).suffix.lower() in {".jpg", ".jpeg"}
                    or asset_file.mime == "image/jpeg"  # type: ignore[union-attr]
                )
                if _use_nvjpeg and is_jpeg:
                    try:
                        thumb_img, thumb_timings = generate_thumbnail_from_jpeg_bytes_nvjpeg_timed(
                            file_bytes,
                            thumb,
                        )
                        _diag_add("thumbnail_decode_seconds", thumb_timings["decode"])
                        _diag_add("thumbnail_decode_bytes_seconds", thumb_timings["decode"])
                        _diag_add("thumbnail_decode_nvjpeg_seconds", thumb_timings["decode"])
                        _diag_add("thumbnail_nvjpeg_decoder_seconds", thumb_timings["decoder"])
                        _diag_add(
                            "thumbnail_nvjpeg_decoder_init_seconds",
                            thumb_timings["decoder_init"],
                        )
                        if thumb_timings["decoder_init"] > 0.0:
                            _diag_inc("thumbnail_nvjpeg_decoder_inits")
                        _diag_add("thumbnail_resize_seconds", thumb_timings["resize"])
                        _diag_add("thumbnail_write_seconds", thumb_timings["write"])
                        _diag_inc("thumbnails_generated")
                        _diag_inc("thumbnails_from_bytes")
                        _diag_inc("thumbnails_from_nvjpeg")
                        generated_thumb = True
                    except ImportError:
                        _diag_inc("thumbnail_nvjpeg_fallbacks")
                    except Exception:
                        _diag_inc("thumbnail_nvjpeg_fallbacks")
                        _log.debug(
                            "nvJPEG thumbnail generation failed for %r; falling back to Pillow",
                            relpath,
                            exc_info=True,
                        )
                        with contextlib.suppress(OSError):
                            thumb.unlink(missing_ok=True)
                try:
                    if not generated_thumb:
                        import io

                        from PIL import Image

                        with Image.open(io.BytesIO(file_bytes)) as full_img:
                            thumb_img, thumb_timings = generate_thumbnail_from_image_timed(
                                full_img, thumb
                            )
                        _diag_add("thumbnail_decode_seconds", thumb_timings["decode"])
                        _diag_add("thumbnail_decode_bytes_seconds", thumb_timings["decode"])
                        _diag_add("thumbnail_resize_seconds", thumb_timings["resize"])
                        _diag_add("thumbnail_write_seconds", thumb_timings["write"])
                        _diag_inc("thumbnails_generated")
                        _diag_inc("thumbnails_from_bytes")
                        generated_thumb = True
                except ImportError:
                    pass  # Pillow not available
                except Exception:
                    _log.debug("Thumbnail generation failed for %r", relpath, exc_info=True)
                    with contextlib.suppress(OSError):
                        thumb.unlink(missing_ok=True)

            if not generated_thumb:
                try:
                    thumb_img, thumb_timings = generate_thumbnail_from_path_timed(
                        asset_file.abspath,
                        thumb,  # type: ignore[union-attr]
                    )
                    _diag_add("thumbnail_decode_seconds", thumb_timings["decode"])
                    _diag_add("thumbnail_decode_path_seconds", thumb_timings["decode"])
                    _diag_add("thumbnail_resize_seconds", thumb_timings["resize"])
                    _diag_add("thumbnail_write_seconds", thumb_timings["write"])
                    _diag_inc("thumbnails_generated")
                    _diag_inc("thumbnails_from_path")
                except ImportError:
                    pass
                except Exception:
                    _log.debug("Path thumbnail generation failed for %r", relpath, exc_info=True)
                    with contextlib.suppress(OSError):
                        thumb.unlink(missing_ok=True)
            _diag_add("thumbnail_seconds", time.perf_counter() - _thumb_start)

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
                    _phash_start = time.perf_counter()
                    try:
                        dhash_hex = compute_dhash_from_image(thumb_img)
                        _enqueue_phash(asset_id, dhash_hex)
                        _diag_inc("phashes_computed")
                    except ImportError:
                        pass
                    except Exception:
                        _log.warning("phash failed for %r", relpath, exc_info=True)
                    finally:
                        _diag_add("phash_seconds", time.perf_counter() - _phash_start)

                if needs_clip and _clip_warmup_ok.is_set():
                    # Queue CLIP embedding from thumbnail.  A dedicated worker
                    # batches these queued images for efficient GPU inference.
                    try:
                        _enqueue_clip(asset_id, relpath, thumb_img)
                    except Exception:
                        _log.warning("CLIP queueing failed for %r", relpath, exc_info=True)

    # Keep only a bounded number of futures in flight. Large takeouts can
    # contain hundreds of thousands of files; submitting every asset at once
    # wastes memory and weakens queue backpressure.
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        asset_iter = iter(assets)
        max_in_flight = max(num_workers * 4, num_workers)
        pending: set[Future] = set()

        def _submit_more() -> None:
            while len(pending) < max_in_flight:
                try:
                    asset_file = next(asset_iter)
                except StopIteration:
                    break
                pending.add(executor.submit(_process_one, asset_file))

        _submit_more()
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
            _submit_more()

    _db_queue.put(_db_sentinel)
    _db_queue.join()
    _db_thread.join(timeout=30)

    if _clip_thread is not None:
        for _ in range(_clip_preprocess_workers):
            _clip_image_queue.put(_clip_image_sentinel)
        if _clip_thread.is_alive():
            _clip_image_queue.join()
            _clip_tensor_queue.join()
            _clip_thread.join(timeout=30)

    _diag_set("processing_seconds", time.perf_counter() - _processing_start)
    progress.running = False
    progress.done = True
    if cancel_check is not None and cancel_check():
        progress.cancelled = True
    progress.diagnostics = _diag_snapshot()
    return progress
