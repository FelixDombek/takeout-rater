"""Scoring pipeline: runs a scorer over indexed assets and writes results to the DB.

Usage example::

    from takeout_rater.scoring.scorers.simple import SimpleScorer

    scorer = SimpleScorer.create(variant_id="blur")
    run_scorer(conn, scorer, thumbs_dir)

The function iterates over assets in configurable batches, calls
``scorer.score_batch()``, and writes each result directly to
``asset_scores`` via ``INSERT OR REPLACE``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from takeout_rater.db.queries import (
    list_asset_ids_without_score,
    upsert_asset_scores,
)
from takeout_rater.indexing.thumbnailer import thumb_path_for_id
from takeout_rater.scoring.scorers.base import BaseScorer

_logger = logging.getLogger(__name__)

#: Default number of assets handed to one ``score_batch()`` call.  Scorers split
#: this into their own GPU-sized chunks and decode chunk N+1 while chunk N runs
#: on the GPU (see ``_run_pipelined_batches``); that overlap only happens when a
#: call spans several chunks, so this is deliberately much larger than the
#: per-model batch sizes.  It also sets how often progress is reported and
#: scores are committed.
DEFAULT_SCORE_BATCH_SIZE = 256

#: CLIP ViT-L/14 embedding size and the byte length of one stored float32 row.
_CLIP_DIM = 768
_CLIP_BLOB_BYTES = _CLIP_DIM * 4


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


def _score_pairs(
    conn: sqlite3.Connection,
    scorer: BaseScorer,
    pairs: list[tuple[int, Path]],
    scorer_id: str,
    variant_id: str,
    embeddings: dict[int, bytes] | None = None,
) -> tuple[list[dict[str, float]], int]:
    """Score ``(asset_id, thumb_path)`` pairs, using stored CLIP embeddings when possible.

    For scorers with :attr:`~BaseScorer.consumes_clip_embeddings`, assets
    that have a row in ``clip_embeddings`` are scored from that vector (no
    image decode, no CLIP forward pass); the rest go through
    :meth:`~BaseScorer.score_batch` on their thumbnails as before.

    Args:
        embeddings: Pre-fetched ``{asset_id: blob}``; looked up when ``None``.

    Returns:
        ``(score_dicts, n_from_embeddings)`` with one dict per pair, in order.
    """
    if not scorer.consumes_clip_embeddings:
        paths = [p for _, p in pairs]
        return _score_batch_with_context(scorer, paths, scorer_id, variant_id), 0

    from takeout_rater.db.queries import blobs_to_matrix, get_clip_embeddings_for_assets

    if embeddings is None:
        embeddings = get_clip_embeddings_for_assets(conn, [aid for aid, _ in pairs])
    emb_idx = [
        i for i, (aid, _) in enumerate(pairs) if len(embeddings.get(aid, b"")) == _CLIP_BLOB_BYTES
    ]
    img_idx = sorted(set(range(len(pairs))) - set(emb_idx))
    results: list[dict[str, float]] = [{} for _ in pairs]

    if emb_idx:
        matrix = blobs_to_matrix([embeddings[pairs[i][0]] for i in emb_idx], dim=_CLIP_DIM)
        try:
            emb_scores = scorer.score_clip_embeddings(matrix)
        except Exception as exc:  # noqa: BLE001
            _logger.error(
                "Scorer %r (variant %r) failed on %d stored CLIP embedding(s): %s",
                scorer_id,
                variant_id,
                len(emb_idx),
                exc,
            )
            raise RuntimeError(
                f"Scorer {scorer_id!r} (variant {variant_id!r}) failed on "
                f"{len(emb_idx)} stored CLIP embedding(s): {exc}"
            ) from exc
        for i, score in zip(emb_idx, emb_scores, strict=True):
            results[i] = score
    if img_idx:
        img_scores = _score_batch_with_context(
            scorer, [pairs[i][1] for i in img_idx], scorer_id, variant_id
        )
        for i, score in zip(img_idx, img_scores, strict=True):
            results[i] = score
    return results, len(emb_idx)


def run_scorer(
    conn: sqlite3.Connection,
    scorer: BaseScorer,
    thumbs_dir: Path,
    *,
    asset_ids: list[int] | None = None,
    batch_size: int = DEFAULT_SCORE_BATCH_SIZE,
    skip_existing: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
    on_diagnostics: Callable[[dict[str, object]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> int:
    """Run *scorer* over assets and persist the results to the library DB.

    Scores are written directly to ``asset_scores`` via ``INSERT OR REPLACE``.
    There is no separate "run" record; each score row is self-contained with
    its ``scorer_id``, ``variant_id``, ``metric_key``, ``scorer_version``, and
    ``scored_at`` timestamp.

    Args:
        conn: Open library database connection.
        scorer: Instantiated scorer (must have been created via
            :meth:`~takeout_rater.scoring.scorers.base.BaseScorer.create`).
        thumbs_dir: Directory containing pre-generated thumbnail files.
            Thumbnails are used instead of originals for speed.
        asset_ids: Explicit list of asset IDs to score.  When ``None``
            (default), all assets that lack a score for this scorer/metric
            are scored.
        batch_size: Number of images per ``score_batch()`` call (default
            :data:`DEFAULT_SCORE_BATCH_SIZE`).
        skip_existing: When ``True`` (default) and ``asset_ids`` is ``None``,
            skip assets that already have a score for the *first* metric in
            the variant spec. For single-metric variants this is equivalent to
            skipping fully-scored assets; for multi-metric variants it uses the
            first metric as the primary "scored" indicator.
        on_progress: Optional callback invoked after each batch with
            ``(scored_so_far, total)`` integers.
        on_diagnostics: Optional callback invoked with cumulative counters and
            timings after target selection and each processed batch.
        cancel_check: Optional callable that returns ``True`` when the run
            should be aborted.  Checked after each batch; when it returns
            ``True`` the loop exits early.

    Returns:
        The number of assets scored.
    """
    spec = scorer.spec()
    scorer_id = spec.scorer_id
    variant_id = scorer.variant_id
    scorer_version = spec.version
    started_at = time.perf_counter()
    diagnostics: dict[str, object] = {
        "scorer_id": scorer_id,
        "variant_id": variant_id,
        "scorer_version": scorer_version,
        "score_batch_size": batch_size,
        "score_assets_total": 0,
        "score_assets_seen": 0,
        "score_assets_with_thumbnails": 0,
        "score_thumbnails_missing": 0,
        "score_batches": 0,
        "score_from_clip_embeddings": 0,
        "score_rows_written": 0,
        "score_metrics_written": 0,
        "score_target_select_seconds": 0.0,
        "score_path_build_seconds": 0.0,
        "score_thumbnail_exists_seconds": 0.0,
        "score_batch_seconds": 0.0,
        "score_db_write_seconds": 0.0,
        "score_total_seconds": 0.0,
    }

    def _refresh_cuda_diagnostics() -> None:
        torch_mod = sys.modules.get("torch")
        if torch_mod is None:
            diagnostics["torch_cuda_available"] = "not loaded"
            return
        try:
            cuda_available = bool(torch_mod.cuda.is_available())
            diagnostics["torch_cuda_available"] = cuda_available
            diagnostics["torch_cuda_device_count"] = (
                int(torch_mod.cuda.device_count()) if cuda_available else 0
            )
            if cuda_available:
                diagnostics["torch_cuda_device"] = torch_mod.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            diagnostics["torch_cuda_available"] = "unknown"

    def _emit_diagnostics() -> None:
        _refresh_cuda_diagnostics()
        diagnostics["score_total_seconds"] = time.perf_counter() - started_at
        assets_seen = int(diagnostics.get("score_assets_seen", 0))
        rows_written = int(diagnostics.get("score_rows_written", 0))
        batches = int(diagnostics.get("score_batches", 0))
        if assets_seen:
            diagnostics["score_batch_ms_per_asset"] = (
                float(diagnostics.get("score_batch_seconds", 0.0)) * 1000.0 / assets_seen
            )
        if rows_written:
            diagnostics["score_db_write_ms_per_row"] = (
                float(diagnostics.get("score_db_write_seconds", 0.0)) * 1000.0 / rows_written
            )
        if batches:
            diagnostics["score_batch_seconds_per_batch"] = (
                float(diagnostics.get("score_batch_seconds", 0.0)) / batches
            )
        if on_diagnostics is not None:
            on_diagnostics(dict(diagnostics))

    def _diag_add(key: str, seconds: float) -> None:
        diagnostics[key] = float(diagnostics.get(key, 0.0)) + seconds

    def _diag_inc(key: str, amount: int = 1) -> None:
        diagnostics[key] = int(diagnostics.get(key, 0)) + amount

    # Determine which assets to score
    stream_all_assets = False
    select_start = time.perf_counter()
    if asset_ids is None:
        variant_metrics = spec.metrics_for_variant(variant_id)
        if skip_existing and variant_metrics:
            variant_spec = next((v for v in spec.variants if v.variant_id == variant_id), None)
            first_metric = (
                variant_spec.primary_metric_key
                if variant_spec is not None and variant_spec.primary_metric_key is not None
                else variant_metrics[0].key
            )
            asset_ids = list_asset_ids_without_score(
                conn, scorer_id, variant_id, first_metric, scorer_version=scorer_version
            )
        else:
            stream_all_assets = True
    _diag_add("score_target_select_seconds", time.perf_counter() - select_start)

    # Nothing to score — return early.
    if asset_ids is not None and not asset_ids:
        _emit_diagnostics()
        return 0

    scored = 0

    if stream_all_assets:
        count_start = time.perf_counter()
        cur = conn.execute("SELECT COUNT(*) FROM assets")
        row = cur.fetchone()
        total = int(row[0]) if row is not None and row[0] is not None else 0
        _diag_add("score_target_select_seconds", time.perf_counter() - count_start)
        diagnostics["score_assets_total"] = total
        _emit_diagnostics()

        id_cur = conn.execute("SELECT id FROM assets ORDER BY id")
        while True:
            if cancel_check is not None and cancel_check():
                break
            id_rows = id_cur.fetchmany(batch_size)
            if not id_rows:
                break

            batch_ids = [int(r[0]) for r in id_rows]

            valid_pairs: list[tuple[int, Path]] = []
            path_start = time.perf_counter()
            for aid in batch_ids:
                thumb = thumb_path_for_id(thumbs_dir, aid)
                exists_start = time.perf_counter()
                if thumb.exists():
                    _diag_add("score_thumbnail_exists_seconds", time.perf_counter() - exists_start)
                    valid_pairs.append((aid, thumb))
                else:
                    _diag_add("score_thumbnail_exists_seconds", time.perf_counter() - exists_start)
                    _diag_inc("score_thumbnails_missing")
            _diag_add("score_path_build_seconds", time.perf_counter() - path_start)
            _diag_inc("score_assets_seen", len(batch_ids))
            _diag_inc("score_assets_with_thumbnails", len(valid_pairs))

            if valid_pairs:
                batch_start = time.perf_counter()
                score_dicts, from_embeddings = _score_pairs(
                    conn, scorer, valid_pairs, scorer_id, variant_id
                )
                _diag_add("score_batch_seconds", time.perf_counter() - batch_start)
                _diag_inc("score_from_clip_embeddings", from_embeddings)
                _diag_inc("score_batches")

                rows: list[tuple[int, str, float]] = []
                for (aid, _), score_dict in zip(valid_pairs, score_dicts, strict=True):
                    for metric_key, value in score_dict.items():
                        rows.append((aid, metric_key, value))

                if rows:
                    write_start = time.perf_counter()
                    upsert_asset_scores(
                        conn,
                        scorer_id,
                        variant_id,
                        rows,
                        scorer_version=scorer_version,
                    )
                    _diag_add("score_db_write_seconds", time.perf_counter() - write_start)
                    _diag_inc("score_rows_written", len(rows))
                    _diag_inc("score_metrics_written", sum(len(score) for score in score_dicts))

            scored += len(batch_ids)
            _emit_diagnostics()
            if on_progress is not None:
                on_progress(scored, total)

    else:
        assert asset_ids is not None
        total = len(asset_ids)
        diagnostics["score_assets_total"] = total
        _emit_diagnostics()

        for batch_start in range(0, total, batch_size):
            if cancel_check is not None and cancel_check():
                break
            batch_ids = asset_ids[batch_start : batch_start + batch_size]

            # Build (asset_id, thumb_path) pairs, skipping missing thumbnails
            valid_pairs = []
            path_start = time.perf_counter()
            for aid in batch_ids:
                thumb = thumb_path_for_id(thumbs_dir, aid)
                exists_start = time.perf_counter()
                if thumb.exists():
                    _diag_add("score_thumbnail_exists_seconds", time.perf_counter() - exists_start)
                    valid_pairs.append((aid, thumb))
                else:
                    _diag_add("score_thumbnail_exists_seconds", time.perf_counter() - exists_start)
                    _diag_inc("score_thumbnails_missing")
            _diag_add("score_path_build_seconds", time.perf_counter() - path_start)
            _diag_inc("score_assets_seen", len(batch_ids))
            _diag_inc("score_assets_with_thumbnails", len(valid_pairs))

            if valid_pairs:
                score_start = time.perf_counter()
                score_dicts, from_embeddings = _score_pairs(
                    conn, scorer, valid_pairs, scorer_id, variant_id
                )
                _diag_add("score_batch_seconds", time.perf_counter() - score_start)
                _diag_inc("score_from_clip_embeddings", from_embeddings)
                _diag_inc("score_batches")

                rows = []
                for (aid, _), score_dict in zip(valid_pairs, score_dicts, strict=True):
                    for metric_key, value in score_dict.items():
                        rows.append((aid, metric_key, value))

                if rows:
                    write_start = time.perf_counter()
                    upsert_asset_scores(
                        conn,
                        scorer_id,
                        variant_id,
                        rows,
                        scorer_version=scorer_version,
                    )
                    _diag_add("score_db_write_seconds", time.perf_counter() - write_start)
                    _diag_inc("score_rows_written", len(rows))
                    _diag_inc("score_metrics_written", sum(len(score) for score in score_dicts))

            scored += len(batch_ids)
            _emit_diagnostics()
            if on_progress is not None:
                on_progress(scored, total)

    return scored


def run_scorer_by_id(
    conn: sqlite3.Connection,
    scorer_id: str,
    thumbs_dir: Path,
    *,
    variant_id: str | None = None,
    asset_ids: list[int] | None = None,
    batch_size: int = DEFAULT_SCORE_BATCH_SIZE,
    skip_existing: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
    on_diagnostics: Callable[[dict[str, object]], None] | None = None,
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
        The number of assets scored.

    Raises:
        KeyError: If *scorer_id* is not found in the registry.
        RuntimeError: If the scorer is not available (missing optional deps).
    """
    from takeout_rater.scoring.scorers.registry import list_scorers

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
    env_batch_size = os.environ.get("TAKEOUT_RATER_SCORE_BATCH_SIZE")
    if env_batch_size and batch_size == DEFAULT_SCORE_BATCH_SIZE:
        with contextlib.suppress(ValueError):
            batch_size = max(1, int(env_batch_size))
    try:
        return run_scorer(
            conn,
            scorer,
            thumbs_dir,
            asset_ids=asset_ids,
            batch_size=batch_size,
            skip_existing=skip_existing,
            on_progress=on_progress,
            on_diagnostics=on_diagnostics,
            cancel_check=cancel_check,
        )
    finally:
        # Scorers that hold heavy resources (models in VRAM, thread pools)
        # may expose close(); release them as soon as the run ends.
        close = getattr(scorer, "close", None)
        if callable(close):
            close()


# ---------------------------------------------------------------------------
# Shared-decode passes: several scorer variants over one decode per thumbnail
# ---------------------------------------------------------------------------

#: Default GPU memory budget for the models of one shared pass when CUDA is
#: unavailable (models then live in system RAM).
_CPU_GROUP_BUDGET_MB = 6144
#: Fraction of total VRAM that the *weights* of one pass may occupy; the rest
#: is left for activations, CUDA contexts and ONNX Runtime's own arena.
_GPU_GROUP_BUDGET_FRACTION = 0.5


def default_group_budget_mb() -> int:
    """Return the memory budget (MiB) for the models loaded in one shared pass.

    ``TAKEOUT_RATER_SCORE_GROUP_MB`` overrides it; ``0`` turns shared passes
    off (every variant then runs on its own, as before).  Otherwise it is half
    of the first CUDA device's memory, or :data:`_CPU_GROUP_BUDGET_MB` on CPU.
    """
    env = os.environ.get("TAKEOUT_RATER_SCORE_GROUP_MB")
    if env is not None and env.strip():
        try:
            return max(0, int(env))
        except ValueError:
            _logger.warning("Ignoring invalid TAKEOUT_RATER_SCORE_GROUP_MB=%r", env)
    try:
        import torch

        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            return int(total / (1024 * 1024) * _GPU_GROUP_BUDGET_FRACTION)
    except Exception:  # noqa: BLE001
        pass
    return _CPU_GROUP_BUDGET_MB


def plan_scorer_groups[T](
    items: list[T],
    cost: Callable[[T], tuple[int, str | None]],
    budget_mb: int,
) -> list[list[T]]:
    """Pack scorer variants into passes whose models fit *budget_mb* together.

    First-fit decreasing on each item's ``(megabytes, share_key)``; items with
    the same ``share_key`` (e.g. LAION and CLIP-IQA on the CLIP singleton)
    count their model once per group.  CPU-only items (``0`` MB) always join
    the first group.  An item larger than the budget gets a group of its own,
    and ``budget_mb <= 0`` puts every item in its own group.  Groups keep the
    items' original order and are ordered by their first item.
    """
    if budget_mb <= 0:
        return [[item] for item in items]

    costs = [cost(item) for item in items]
    order = sorted(range(len(items)), key=lambda i: -costs[i][0])
    groups: list[list[int]] = []
    group_cost: list[int] = []
    group_keys: list[set[str]] = []
    for i in order:
        mb, key = costs[i]
        placed = False
        for g in range(len(groups)):
            extra = 0 if key is not None and key in group_keys[g] else mb
            if group_cost[g] + extra <= budget_mb:
                groups[g].append(i)
                group_cost[g] += extra
                if key is not None:
                    group_keys[g].add(key)
                placed = True
                break
        if not placed:
            groups.append([i])
            group_cost.append(mb)
            group_keys.append({key} if key is not None else set())
    ordered = sorted((sorted(g) for g in groups), key=lambda g: g[0])
    return [[items[i] for i in g] for g in ordered]


def _decode_or_none(path: Path) -> Any:
    from takeout_rater.scoring.shared_decode import decode_rgb

    try:
        return decode_rgb(path)
    except Exception:  # noqa: BLE001 - the scorer will retry and handle it itself
        return None


def _target_asset_ids(
    conn: sqlite3.Connection, scorer: BaseScorer, skip_existing: bool
) -> list[int]:
    """Return the asset IDs *scorer* still needs to score (all when not skipping)."""
    spec = scorer.spec()
    variant_id = scorer.variant_id
    variant_metrics = spec.metrics_for_variant(variant_id)
    if skip_existing and variant_metrics:
        variant_spec = next((v for v in spec.variants if v.variant_id == variant_id), None)
        first_metric = (
            variant_spec.primary_metric_key
            if variant_spec is not None and variant_spec.primary_metric_key is not None
            else variant_metrics[0].key
        )
        return list_asset_ids_without_score(
            conn, spec.scorer_id, variant_id, first_metric, scorer_version=spec.version
        )
    return [int(r[0]) for r in conn.execute("SELECT id FROM assets ORDER BY id").fetchall()]


def _decode_workers() -> int:
    env = os.environ.get("TAKEOUT_RATER_SCORE_DECODE_WORKERS")
    if env:
        with contextlib.suppress(ValueError):
            return max(1, int(env))
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def run_scorer_group(
    conn: sqlite3.Connection,
    scorers: list[BaseScorer],
    thumbs_dir: Path,
    *,
    batch_size: int = DEFAULT_SCORE_BATCH_SIZE,
    skip_existing: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
    on_diagnostics: Callable[[dict[str, object]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[tuple[str, str], int]:
    """Run several scorer variants in one pass, decoding each thumbnail once.

    Assets are walked in batches over the union of what every scorer still
    needs.  For each batch the thumbnails that any scorer will actually read
    are decoded once, on a thread pool and one batch ahead (so decoding
    overlaps with inference), and handed to every scorer through
    :func:`takeout_rater.scoring.shared_decode.shared_images`.  Scorers that
    consume CLIP embeddings read ``clip_embeddings`` instead and only need a
    decode for assets that have no stored embedding.

    Scores, skip logic and error handling per scorer are the same as
    :func:`run_scorer`; only the order of work changes.  All scorers stay
    loaded for the whole pass, so callers should size groups with
    :func:`plan_scorer_groups`.

    Returns:
        ``{(scorer_id, variant_id): assets_scored}``.
    """
    from concurrent.futures import Future, ThreadPoolExecutor

    from takeout_rater.db.queries import get_clip_embeddings_for_assets
    from takeout_rater.scoring.shared_decode import shared_images

    started_at = time.perf_counter()
    keys = [(s.spec().scorer_id, s.variant_id) for s in scorers]
    counts: dict[tuple[str, str], int] = dict.fromkeys(keys, 0)
    diagnostics: dict[str, object] = {
        "scorer_id": ", ".join(sorted({k[0] for k in keys})),
        "variant_id": ", ".join(f"{sid}/{vid}" for sid, vid in keys),
        "score_group_size": len(scorers),
        "score_batch_size": batch_size,
        "score_assets_total": 0,
        "score_assets_seen": 0,
        "score_assets_with_thumbnails": 0,
        "score_thumbnails_missing": 0,
        "score_batches": 0,
        "score_rows_written": 0,
        "score_metrics_written": 0,
        "score_shared_decodes": 0,
        "score_from_clip_embeddings": 0,
        "score_target_select_seconds": 0.0,
        "score_thumbnail_exists_seconds": 0.0,
        "score_decode_wait_seconds": 0.0,
        "score_batch_seconds": 0.0,
        "score_db_write_seconds": 0.0,
        "score_total_seconds": 0.0,
    }

    def _add(key: str, value: float) -> None:
        diagnostics[key] = diagnostics[key] + value  # type: ignore[operator]

    def _emit() -> None:
        diagnostics["score_total_seconds"] = time.perf_counter() - started_at
        seen = int(diagnostics["score_assets_seen"])  # type: ignore[arg-type]
        rows = int(diagnostics["score_rows_written"])  # type: ignore[arg-type]
        batches = int(diagnostics["score_batches"])  # type: ignore[arg-type]
        if seen:
            diagnostics["score_batch_ms_per_asset"] = (
                float(diagnostics["score_batch_seconds"]) * 1000.0 / seen  # type: ignore[arg-type]
            )
        if rows:
            diagnostics["score_db_write_ms_per_row"] = (
                float(diagnostics["score_db_write_seconds"]) * 1000.0 / rows  # type: ignore[arg-type]
            )
        if batches:
            diagnostics["score_batch_seconds_per_batch"] = (
                float(diagnostics["score_batch_seconds"]) / batches  # type: ignore[arg-type]
            )
        torch_mod = sys.modules.get("torch")
        if torch_mod is not None:
            with contextlib.suppress(Exception):
                diagnostics["torch_cuda_available"] = bool(torch_mod.cuda.is_available())
                if diagnostics["torch_cuda_available"]:
                    diagnostics["torch_cuda_device"] = torch_mod.cuda.get_device_name(0)
        if on_diagnostics is not None:
            on_diagnostics(dict(diagnostics))

    select_start = time.perf_counter()
    targets = [set(_target_asset_ids(conn, s, skip_existing)) for s in scorers]
    union = sorted(set().union(*targets)) if targets else []
    _add("score_target_select_seconds", time.perf_counter() - select_start)
    total = len(union)
    diagnostics["score_assets_total"] = total
    _emit()
    if not union:
        return counts

    any_embedding_scorer = any(s.consumes_clip_embeddings for s in scorers)
    several_gpu_models = sum(1 for s in scorers if s.gpu_memory_estimate_mb(s.variant_id) > 0) > 1

    def _release_cuda_cache() -> None:
        # Scorers import torch lazily, so look it up at call time.
        torch_mod = sys.modules.get("torch")
        if torch_mod is not None and torch_mod.cuda.is_available():
            torch_mod.cuda.empty_cache()

    def _plan(
        batch_ids: list[int], pool: ThreadPoolExecutor
    ) -> tuple[list[list[tuple[int, Path]]], dict[int, bytes], dict[str, Future[Any]], int]:
        exists_start = time.perf_counter()
        thumbs: dict[int, Path] = {}
        for aid in batch_ids:
            thumb = thumb_path_for_id(thumbs_dir, aid)
            if thumb.exists():
                thumbs[aid] = thumb
        _add("score_thumbnail_exists_seconds", time.perf_counter() - exists_start)
        _add("score_thumbnails_missing", len(batch_ids) - len(thumbs))
        per_scorer = [
            [(aid, thumbs[aid]) for aid in batch_ids if aid in thumbs and aid in target]
            for target in targets
        ]
        embeddings: dict[int, bytes] = {}
        if any_embedding_scorer:
            wanted = sorted(
                {
                    aid
                    for s, pairs in zip(scorers, per_scorer, strict=True)
                    if s.consumes_clip_embeddings
                    for aid, _ in pairs
                }
            )
            embeddings = get_clip_embeddings_for_assets(conn, wanted)
        to_decode: dict[str, Path] = {}
        for s, pairs in zip(scorers, per_scorer, strict=True):
            if not s.uses_shared_decode:
                continue
            for aid, path in pairs:
                if s.consumes_clip_embeddings and len(embeddings.get(aid, b"")) == _CLIP_BLOB_BYTES:
                    continue
                to_decode[str(path)] = path
        futures = {key: pool.submit(_decode_or_none, path) for key, path in to_decode.items()}
        return per_scorer, embeddings, futures, len(thumbs)

    batches = [union[i : i + batch_size] for i in range(0, total, batch_size)]
    processed = 0
    pool = ThreadPoolExecutor(max_workers=_decode_workers(), thread_name_prefix="score-decode")
    try:
        planned = _plan(batches[0], pool)
        for b_idx, batch_ids in enumerate(batches):
            if cancel_check is not None and cancel_check():
                break
            per_scorer, embeddings, futures, n_thumbs = planned
            # Queue the next batch's decodes before scoring this one.
            planned = _plan(batches[b_idx + 1], pool) if b_idx + 1 < len(batches) else planned

            wait_start = time.perf_counter()
            images = {key: fut.result() for key, fut in futures.items()}
            images = {key: img for key, img in images.items() if img is not None}
            _add("score_decode_wait_seconds", time.perf_counter() - wait_start)
            _add("score_shared_decodes", len(images))

            with shared_images(images):
                for scorer, key, pairs in zip(scorers, keys, per_scorer, strict=True):
                    if not pairs:
                        continue
                    sid, vid = key
                    score_start = time.perf_counter()
                    score_dicts, from_emb = _score_pairs(
                        conn, scorer, pairs, sid, vid, embeddings=embeddings
                    )
                    _add("score_batch_seconds", time.perf_counter() - score_start)
                    _add("score_from_clip_embeddings", from_emb)
                    if several_gpu_models:
                        # Hand cached activation memory back before the next
                        # model (ONNX Runtime cannot use torch's cache).
                        _release_cuda_cache()

                    rows = [
                        (aid, metric_key, value)
                        for (aid, _), score in zip(pairs, score_dicts, strict=True)
                        for metric_key, value in score.items()
                    ]
                    if rows:
                        write_start = time.perf_counter()
                        upsert_asset_scores(
                            conn, sid, vid, rows, scorer_version=scorer.spec().version
                        )
                        _add("score_db_write_seconds", time.perf_counter() - write_start)
                        _add("score_rows_written", len(rows))
                        _add("score_metrics_written", sum(len(d) for d in score_dicts))
                    counts[key] += len(pairs)
            del images

            processed += len(batch_ids)
            _add("score_batches", 1)
            _add("score_assets_seen", len(batch_ids))
            _add("score_assets_with_thumbnails", n_thumbs)
            _emit()
            if on_progress is not None:
                on_progress(processed, total)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return counts
