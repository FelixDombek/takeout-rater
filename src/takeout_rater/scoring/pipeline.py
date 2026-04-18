"""Scoring pipeline: runs a scorer over indexed assets and writes results to the DB.

Usage example::

    from takeout_rater.scorers.simple import SimpleScorer

    scorer = SimpleScorer.create(variant_id="blur")
    run_scorer(conn, scorer, thumbs_dir)

The function iterates over assets in configurable batches, calls
``scorer.score_batch()``, and writes each result directly to
``asset_scores`` via ``INSERT OR REPLACE``.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from pathlib import Path

from takeout_rater.db.queries import (
    list_asset_ids_without_score,
    upsert_asset_scores,
)
from takeout_rater.indexing.thumbnailer import thumb_path_for_id
from takeout_rater.scorers.base import BaseScorer

_logger = logging.getLogger(__name__)


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

    Scores are written directly to ``asset_scores`` via ``INSERT OR REPLACE``.
    There is no separate "run" record; each score row is self-contained with
    its ``scorer_id``, ``variant_id``, ``metric_key``, ``scorer_version``, and
    ``scored_at`` timestamp.

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
            the variant spec. For single-metric variants this is equivalent to
            skipping fully-scored assets; for multi-metric variants it uses the
            first metric as the primary "scored" indicator.
        on_progress: Optional callback invoked after each batch with
            ``(scored_so_far, total)`` integers.
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

    # Determine which assets to score
    stream_all_assets = False
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

    # Nothing to score — return early.
    if asset_ids is not None and not asset_ids:
        return 0

    scored = 0

    if stream_all_assets:
        cur = conn.execute("SELECT COUNT(*) FROM assets")
        row = cur.fetchone()
        total = int(row[0]) if row is not None and row[0] is not None else 0

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
                    upsert_asset_scores(
                        conn,
                        scorer_id,
                        variant_id,
                        rows,
                        scorer_version=scorer_version,
                    )

            scored += len(batch_ids)
            if on_progress is not None:
                on_progress(scored, total)

    else:
        assert asset_ids is not None
        total = len(asset_ids)

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
                    upsert_asset_scores(
                        conn,
                        scorer_id,
                        variant_id,
                        rows,
                        scorer_version=scorer_version,
                    )

            scored += len(batch_ids)
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
        The number of assets scored.

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
