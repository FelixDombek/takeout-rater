"""Scoring pipeline: runs a scorer over indexed assets and writes results to the DB.

Usage example::

    from takeout_rater.scoring.pipeline import run_scorer
    from takeout_rater.scorers.heuristics.blur import BlurScorer

    scorer = BlurScorer.create()
    run_id = run_scorer(conn, scorer, thumbs_dir)

The function creates a ``scorer_runs`` record, iterates over assets in
configurable batches, calls ``scorer.score_batch()``, and writes each result
to ``asset_scores``.  When finished it sets ``scorer_runs.finished_at`` and
returns the run ID.  ``finish_scorer_run()`` is always called via a
``try/finally`` block, even if an error occurs mid-run.

For running multiple scorers together, :func:`run_scorers_parallel` provides a
more efficient alternative: it loads each thumbnail once and fans out scoring
tasks across a thread pool, so I/O cost is paid only once per asset regardless
of how many scorers are requested.
"""

from __future__ import annotations

import logging
import os
import queue
import sqlite3
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from takeout_rater.db.queries import (
    bulk_insert_asset_scores,
    finish_scorer_run,
    insert_scorer_run,
    list_asset_ids_without_score,
)
from takeout_rater.indexing.thumbnailer import thumb_path_for_id
from takeout_rater.scorers.base import BaseScorer

_logger = logging.getLogger(__name__)

# Number of in-flight result batches the queue can hold ahead of the DB writer.
# Larger values allow workers to proceed further without blocking, at the cost
# of buffering more (asset_id, metric_key, value) tuples in memory.
_QUEUE_SIZE_PER_WORKER = 4


def _score_batch_with_context(
    scorer: BaseScorer,
    paths: list[Path],
    scorer_id: str,
    variant_id: str,
) -> list[dict[str, float]]:
    """Call ``scorer.score_batch(paths)`` and re-raise with scorer/asset context.

    If ``score_batch`` raises any exception, the error is logged at ERROR level
    with the scorer ID, variant, and the paths of the failing batch, then
    re-raised as a :exc:`RuntimeError` whose message includes the same context.
    This ensures that the job-panel error message identifies both the scorer and
    the affected assets rather than showing a bare library-level traceback.

    Args:
        scorer: Instantiated scorer.
        paths: Batch of thumbnail paths to score.
        scorer_id: Scorer identifier used in error messages.
        variant_id: Variant identifier used in error messages.

    Returns:
        List of score dicts as returned by ``scorer.score_batch``.

    Raises:
        RuntimeError: When ``score_batch`` raises, wrapping the original error
            with scorer and asset path context.
    """
    try:
        return scorer.score_batch(paths)
    except Exception as exc:  # noqa: BLE001
        # Build a compact asset list for the error message.
        _MAX_SHOWN = 3
        shown = [str(p) for p in paths[:_MAX_SHOWN]]
        suffix = f", … (+{len(paths) - _MAX_SHOWN} more)" if len(paths) > _MAX_SHOWN else ""
        asset_summary = ", ".join(shown) + suffix
        _logger.error(
            "Scorer %r (variant %r) failed on %d asset(s) [%s]: %s",
            scorer_id,
            variant_id,
            len(paths),
            asset_summary,
            exc,
        )
        raise RuntimeError(
            f"Scorer {scorer_id!r} (variant {variant_id!r}) failed on "
            f"{len(paths)} asset(s) [{asset_summary}]: {exc}"
        ) from exc


def run_scorer(
    conn: sqlite3.Connection,
    scorer: BaseScorer,
    thumbs_dir: Path,
    *,
    asset_ids: list[int] | None = None,
    batch_size: int = 32,
    skip_existing: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> int:
    """Run *scorer* over assets and persist the results to the library DB.

    A new ``scorer_runs`` record is created before processing begins and its
    ``finished_at`` timestamp is always set when the function returns (even if
    an error occurs mid-run), via a ``try/finally`` block.

    Args:
        conn: Open library database connection.
        scorer: Instantiated scorer (must have been created via
            :meth:`~takeout_rater.scorers.base.BaseScorer.create`).
        thumbs_dir: Directory containing pre-generated thumbnail files.
            Thumbnails are used instead of originals for speed.
        asset_ids: Explicit list of asset IDs to score.  When ``None``
            (default), all assets that lack a score for this scorer/metric
            are scored.
        batch_size: Number of images per ``score_batch()`` call (default 32).
        skip_existing: When ``True`` (default) and ``asset_ids`` is ``None``,
            skip assets that already have a score for the *first* metric in
            the scorer spec.  For single-metric scorers this is equivalent to
            skipping fully-scored assets; for multi-metric scorers it uses the
            first metric as the primary "scored" indicator.
        on_progress: Optional callback invoked after each batch with
            ``(scored_so_far, total)`` integers.
        cancel_check: Optional callable that returns ``True`` when the run
            should be aborted.  Checked after each batch; when it returns
            ``True`` the loop exits early and the scorer run is still finalised
            in the DB.

    Returns:
        The integer primary key of the new ``scorer_runs`` row.
    """
    spec = scorer.spec()
    scorer_id = spec.scorer_id
    variant_id = scorer.variant_id

    # Determine which assets to score
    stream_all_assets = False
    if asset_ids is None:
        if skip_existing and spec.metrics:
            first_metric = spec.metrics[0].key
            asset_ids = list_asset_ids_without_score(conn, scorer_id, variant_id, first_metric)
        else:
            # Score all assets by streaming IDs directly from the DB to avoid
            # materializing full AssetRow objects or all IDs in memory.
            stream_all_assets = True

    # Create scorer run record
    run_id = insert_scorer_run(conn, scorer_id, variant_id, scorer_version=spec.version)

    try:
        if stream_all_assets:
            cur = conn.execute("SELECT COUNT(*) FROM assets")
            row = cur.fetchone()
            total = int(row[0]) if row is not None and row[0] is not None else 0
            scored = 0

            id_cur = conn.execute("SELECT id FROM assets ORDER BY id")
            while True:
                if cancel_check is not None and cancel_check():
                    break
                id_rows = id_cur.fetchmany(batch_size)
                if not id_rows:
                    break

                batch_ids = [int(r[0]) for r in id_rows]

                valid_pairs: list[tuple[int, Path]] = []
                for aid in batch_ids:
                    thumb = thumb_path_for_id(thumbs_dir, aid)
                    if thumb.exists():
                        valid_pairs.append((aid, thumb))

                if valid_pairs:
                    paths = [p for _, p in valid_pairs]
                    score_dicts = _score_batch_with_context(scorer, paths, scorer_id, variant_id)

                    rows: list[tuple[int, str, float]] = []
                    for (aid, _), score_dict in zip(valid_pairs, score_dicts, strict=True):
                        for metric_key, value in score_dict.items():
                            rows.append((aid, metric_key, value))

                    if rows:
                        bulk_insert_asset_scores(conn, run_id, rows)

                scored += len(batch_ids)
                if on_progress is not None:
                    on_progress(scored, total)

        else:
            assert asset_ids is not None
            total = len(asset_ids)
            scored = 0

            for batch_start in range(0, total, batch_size):
                if cancel_check is not None and cancel_check():
                    break
                batch_ids = asset_ids[batch_start : batch_start + batch_size]

                # Build (asset_id, thumb_path) pairs, skipping missing thumbnails
                valid_pairs = []
                for aid in batch_ids:
                    thumb = thumb_path_for_id(thumbs_dir, aid)
                    if thumb.exists():
                        valid_pairs.append((aid, thumb))

                if valid_pairs:
                    paths = [p for _, p in valid_pairs]
                    score_dicts = _score_batch_with_context(scorer, paths, scorer_id, variant_id)

                    rows = []
                    for (aid, _), score_dict in zip(valid_pairs, score_dicts, strict=True):
                        for metric_key, value in score_dict.items():
                            rows.append((aid, metric_key, value))

                    if rows:
                        bulk_insert_asset_scores(conn, run_id, rows)

                scored += len(batch_ids)
                if on_progress is not None:
                    on_progress(scored, total)

    finally:
        finish_scorer_run(conn, run_id)

    return run_id


def run_scorer_by_id(
    conn: sqlite3.Connection,
    scorer_id: str,
    thumbs_dir: Path,
    *,
    variant_id: str | None = None,
    asset_ids: list[int] | None = None,
    batch_size: int = 32,
    skip_existing: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> int:
    """Look up *scorer_id* in the registry and call :func:`run_scorer`.

    Args:
        conn: Open library database connection.
        scorer_id: ID matching a registered scorer's ``spec().scorer_id``.
        thumbs_dir: Thumbnail directory (see :func:`run_scorer`).
        variant_id: Variant to instantiate; defaults to the spec default.
        asset_ids: Optional explicit asset list (see :func:`run_scorer`).
        batch_size: Batch size (see :func:`run_scorer`).
        skip_existing: Skip already-scored assets (see :func:`run_scorer`).
        on_progress: Progress callback (see :func:`run_scorer`).
        cancel_check: Optional cancel callback (see :func:`run_scorer`).

    Returns:
        The ``scorer_runs`` run ID.

    Raises:
        KeyError: If *scorer_id* is not found in the registry.
        RuntimeError: If the scorer is not available (missing optional deps).
    """
    from takeout_rater.scorers.registry import list_scorers  # noqa: PLC0415

    cls_map = {cls.spec().scorer_id: cls for cls in list_scorers()}
    if scorer_id not in cls_map:
        raise KeyError(f"Unknown scorer id: {scorer_id!r}")
    cls = cls_map[scorer_id]
    if not cls.is_available():
        raise RuntimeError(
            f"Scorer {scorer_id!r} is not available. "
            f"Install the required extras: {cls.spec().requires_extras}"
        )
    scorer = cls.create(variant_id=variant_id)
    return run_scorer(
        conn,
        scorer,
        thumbs_dir,
        asset_ids=asset_ids,
        batch_size=batch_size,
        skip_existing=skip_existing,
        on_progress=on_progress,
        cancel_check=cancel_check,
    )


# ---------------------------------------------------------------------------
# Sentinel used to signal the DB-writer thread to stop.
# ---------------------------------------------------------------------------

_STOP = object()


def run_scorers_parallel(
    conn: sqlite3.Connection,
    scorers: list[BaseScorer],
    thumbs_dir: Path,
    *,
    skip_existing: bool = True,
    max_workers: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[tuple[str, str], int]:
    """Run multiple scorers over assets in parallel, loading each thumbnail once.

    This is a more efficient alternative to calling :func:`run_scorer`
    sequentially for each scorer.  The pipeline has three stages:

    1. **Planning** — determine which assets each scorer still needs to score
       (respecting *skip_existing*) and build a per-asset work list.
    2. **Execution** — a :class:`~concurrent.futures.ThreadPoolExecutor` with
       one thread per CPU core picks up assets from the work list.  Each worker
       loads the thumbnail with PIL once, then calls
       :meth:`~takeout_rater.scorers.base.BaseScorer.score_image` for every
       scorer that needs that asset, protected by a per-scorer
       :class:`threading.Lock` to keep non-thread-safe ML models safe.
    3. **DB writing** — a dedicated thread drains a results queue and calls
       :func:`~takeout_rater.db.queries.bulk_insert_asset_scores`, so all
       SQLite writes are serialised.

    Each scorer gets its own ``scorer_runs`` record created before the pool
    starts; all records are finalised (``finished_at`` set) in a ``finally``
    block even if an error occurs.

    Args:
        conn: Open library database connection.  Used only by the DB-writer
            thread; the worker threads do not touch it.
        scorers: Instantiated scorer instances to run.  Each must have been
            created via :meth:`~takeout_rater.scorers.base.BaseScorer.create`.
            Scorers with the same ``scorer_id``/``variant_id`` should not
            appear more than once (the second would have no work to do).
        thumbs_dir: Directory containing pre-generated thumbnail files.
        skip_existing: When ``True`` (default), skip assets that already have a
            score for the first metric declared by each scorer.
        max_workers: Size of the thread pool.  Defaults to
            ``os.cpu_count()`` (minimum 1).
        on_progress: Optional callback invoked after each asset is fully
            processed (all scorers done for that asset), called with
            ``(assets_done, total_assets)``.
        cancel_check: Optional callable that returns ``True`` when the run
            should be aborted.  Workers check this before each asset; when it
            returns ``True`` they stop submitting new tasks and let in-flight
            tasks finish.

    Returns:
        Dict mapping ``(scorer_id, variant_id)`` → ``scorer_run_id`` for every
        scorer that was run.
    """
    from PIL import Image  # noqa: PLC0415

    if not scorers:
        return {}

    n_workers = max(1, max_workers or os.cpu_count() or 1)

    # ── Phase 1: Planning ────────────────────────────────────────────────────
    # For each scorer, determine which asset IDs need to be scored.
    # Build: asset_id → list of BaseScorer instances needed for that asset.

    # Collect the union of all asset IDs across scorers.
    all_asset_ids_set: set[int] = set()
    scorer_needed_ids: dict[int, set[int]] = {}  # scorer object id → set of asset IDs

    for scorer in scorers:
        spec = scorer.spec()
        if skip_existing and spec.metrics:
            first_metric = spec.metrics[0].key
            needed = set(
                list_asset_ids_without_score(conn, spec.scorer_id, scorer.variant_id, first_metric)
            )
        else:
            cur = conn.execute("SELECT id FROM assets ORDER BY id")
            needed = {int(row[0]) for row in cur.fetchall()}
        scorer_needed_ids[id(scorer)] = needed
        all_asset_ids_set |= needed

    if not all_asset_ids_set:
        # Nothing to do — still create + immediately finish scorer runs so that
        # caller gets valid run IDs.
        run_ids: dict[tuple[str, str], int] = {}
        for scorer in scorers:
            spec = scorer.spec()
            run_id = insert_scorer_run(
                conn, spec.scorer_id, scorer.variant_id, scorer_version=spec.version
            )
            finish_scorer_run(conn, run_id)
            run_ids[(spec.scorer_id, scorer.variant_id)] = run_id
        return run_ids

    # Sort for deterministic order.
    all_asset_ids = sorted(all_asset_ids_set)

    # asset_id → list of (scorer, run_id) pairs needed for that asset.
    asset_scorer_map: dict[int, list[tuple[BaseScorer, int]]] = {aid: [] for aid in all_asset_ids}

    # Create scorer_runs records and populate asset_scorer_map.
    run_ids_map: dict[tuple[str, str], int] = {}
    for scorer in scorers:
        spec = scorer.spec()
        run_id = insert_scorer_run(
            conn, spec.scorer_id, scorer.variant_id, scorer_version=spec.version
        )
        run_ids_map[(spec.scorer_id, scorer.variant_id)] = run_id
        for aid in scorer_needed_ids[id(scorer)]:
            if aid in asset_scorer_map:
                asset_scorer_map[aid].append((scorer, run_id))

    # Remove assets that ended up with no scorers to run.
    asset_scorer_map = {aid: pairs for aid, pairs in asset_scorer_map.items() if pairs}
    effective_total = len(asset_scorer_map)
    effective_asset_ids = sorted(asset_scorer_map)

    # Per-scorer locks to serialise calls into potentially non-thread-safe ML models.
    scorer_locks: dict[int, threading.Lock] = {id(s): threading.Lock() for s in scorers}

    # ── Phase 2 + 3: Execution and DB writing ────────────────────────────────
    # results_queue carries (run_id, rows) tuples or _STOP.
    results_queue: queue.Queue[object] = queue.Queue(maxsize=n_workers * _QUEUE_SIZE_PER_WORKER)
    db_errors: list[Exception] = []

    def _db_writer() -> None:
        """Drain results_queue and write to DB (single-threaded)."""
        while True:
            item = results_queue.get()
            if item is _STOP:
                break
            run_id_w, rows_w = item  # type: ignore[misc]
            try:
                bulk_insert_asset_scores(conn, run_id_w, rows_w)
            except Exception as exc:  # noqa: BLE001
                _logger.error("DB writer error: %s", exc)
                db_errors.append(exc)

    db_thread = threading.Thread(target=_db_writer, daemon=True, name="tr-scorer-db-writer")
    db_thread.start()

    assets_done = 0
    lock_for_progress = threading.Lock()

    def _process_asset(asset_id: int, scorer_run_pairs: list[tuple[BaseScorer, int]]) -> None:
        """Load thumbnail once; run all required scorers; enqueue results."""
        thumb = thumb_path_for_id(thumbs_dir, asset_id)
        if not thumb.exists():
            return

        try:
            img = Image.open(thumb)
            img.load()  # Ensure file is fully decoded before closing.
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Could not load thumbnail %s: %s", thumb, exc)
            return

        for scorer, run_id in scorer_run_pairs:
            spec = scorer.spec()
            try:
                with scorer_locks[id(scorer)]:
                    score_dict = scorer.score_image(img)
            except Exception as exc:  # noqa: BLE001
                _logger.error(
                    "Scorer %r (variant %r) failed on asset %d [%s]: %s",
                    spec.scorer_id,
                    scorer.variant_id,
                    asset_id,
                    thumb,
                    exc,
                )
                continue

            rows: list[tuple[int, str, float]] = [
                (asset_id, metric_key, value) for metric_key, value in score_dict.items()
            ]
            if rows:
                results_queue.put((run_id, rows))

    try:
        with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="tr-scorer") as pool:
            futures = {}
            for asset_id in effective_asset_ids:
                if cancel_check is not None and cancel_check():
                    break
                pairs = asset_scorer_map[asset_id]
                future = pool.submit(_process_asset, asset_id, pairs)
                futures[future] = asset_id

            for future in as_completed(futures):
                exc = future.exception()
                if exc is not None:
                    aid = futures[future]
                    _logger.error("Unexpected error processing asset %d: %s", aid, exc)
                with lock_for_progress:
                    assets_done += 1
                    done_snapshot = assets_done
                if on_progress is not None:
                    on_progress(done_snapshot, effective_total)
    finally:
        # Signal DB writer to stop and wait for it to flush.
        results_queue.put(_STOP)
        db_thread.join()

        # Finalise all scorer run records.
        for run_id in run_ids_map.values():
            finish_scorer_run(conn, run_id)

    if db_errors:
        raise RuntimeError(
            f"DB writer encountered {len(db_errors)} error(s): {db_errors[0]}"
        ) from db_errors[0]

    return run_ids_map
