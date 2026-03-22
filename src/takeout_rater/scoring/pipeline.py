"""Scoring pipeline: runs a scorer over indexed assets and writes results to the DB.

Usage example::

    from takeout_rater.scoring.pipeline import run_scorer
    from takeout_rater.scorers.heuristics.blur import BlurScorer

    scorer = BlurScorer.create()
    run_id = run_scorer(conn, scorer, thumbs_dir)

The function creates a ``scorer_runs`` record, iterates over assets in
configurable batches, calls ``scorer.score_batch()``, and writes each result
to ``asset_scores``.  When finished it sets ``scorer_runs.finished_at`` and
returns the run ID.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

from takeout_rater.db.queries import (
    bulk_insert_asset_scores,
    finish_scorer_run,
    insert_scorer_run,
    list_asset_ids_without_score,
)
from takeout_rater.indexing.thumbnailer import thumb_path_for_id
from takeout_rater.scorers.base import BaseScorer


def run_scorer(
    conn: sqlite3.Connection,
    scorer: BaseScorer,
    thumbs_dir: Path,
    *,
    asset_ids: list[int] | None = None,
    batch_size: int = 32,
    skip_existing: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Run *scorer* over assets and persist the results to the library DB.

    A new ``scorer_runs`` record is created before processing begins and its
    ``finished_at`` timestamp is set when the function returns (even on
    partial completion, to allow re-runs to fill gaps).

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
            skip assets that already have a score for this scorer.
        on_progress: Optional callback invoked after each batch with
            ``(scored_so_far, total)`` integers.

    Returns:
        The integer primary key of the new ``scorer_runs`` row.
    """
    spec = scorer.spec()
    scorer_id = spec.scorer_id
    variant_id = scorer.variant_id

    # Determine which assets to score
    if asset_ids is None:
        if skip_existing and spec.metrics:
            first_metric = spec.metrics[0].key
            asset_ids = list_asset_ids_without_score(conn, scorer_id, variant_id, first_metric)
        else:
            # Score all assets
            from takeout_rater.db.queries import list_assets  # noqa: PLC0415

            asset_ids = [a.id for a in list_assets(conn, limit=10_000_000)]

    # Create scorer run record
    run_id = insert_scorer_run(conn, scorer_id, variant_id)

    total = len(asset_ids)
    scored = 0

    for batch_start in range(0, total, batch_size):
        batch_ids = asset_ids[batch_start : batch_start + batch_size]

        # Build (asset_id, thumb_path) pairs, skipping missing thumbnails
        valid_pairs: list[tuple[int, Path]] = []
        for aid in batch_ids:
            thumb = thumb_path_for_id(thumbs_dir, aid)
            if thumb.exists():
                valid_pairs.append((aid, thumb))

        if valid_pairs:
            paths = [p for _, p in valid_pairs]
            score_dicts = scorer.score_batch(paths)

            rows: list[tuple[int, str, float]] = []
            for (aid, _), score_dict in zip(valid_pairs, score_dicts, strict=True):
                for metric_key, value in score_dict.items():
                    rows.append((aid, metric_key, value))

            if rows:
                bulk_insert_asset_scores(conn, run_id, rows)

        scored += len(batch_ids)
        if on_progress is not None:
            on_progress(scored, total)

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
    )
