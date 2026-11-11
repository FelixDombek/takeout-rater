"""Best-of-cluster export shared by the CLI and the background export job."""

from __future__ import annotations

import shutil
import sqlite3
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from takeout_rater.export.folders import (
    DEFAULT_MIN_CONFIDENCE,
    Category,
    assign_folders,
)


@dataclass
class ExportResult:
    copied: int = 0
    skipped: int = 0
    #: Number of exported files per relative folder (``""`` = export root).
    folders: Counter[str] = field(default_factory=Counter)


def select_best_assets(
    conn: sqlite3.Connection,
    scorer_id: str | None = None,
    metric_key: str | None = None,
    *,
    on_cluster: Callable[[int], None] | None = None,
) -> list[tuple[int, int]]:
    """Pick one asset per cluster: ``[(cluster_id, asset_id), ...]``.

    With a scorer the member with the highest ``scorer_id``/``metric_key``
    value wins; otherwise (or when no member has that score) the cluster
    representative does.  *on_cluster* is called with the running count.
    """
    from takeout_rater.db.queries import (
        get_asset_scores,
        get_cluster_members,
        list_clusters_with_representatives,
    )

    picks: list[tuple[int, int]] = []
    batch = 200
    offset = 0
    seen = 0
    while True:
        clusters = list_clusters_with_representatives(conn, limit=batch, offset=offset)
        if not clusters:
            break
        offset += len(clusters)
        for cluster_info in clusters:
            seen += 1
            if on_cluster is not None:
                on_cluster(seen)
            cluster_id = cluster_info["cluster_id"]
            members = get_cluster_members(conn, cluster_id)
            if not members:
                continue
            best_asset_id: int | None = None
            if scorer_id and metric_key:
                best_score = float("-inf")
                for asset, _dist, _is_rep in members:
                    for s in get_asset_scores(conn, asset.id):
                        if s["scorer_id"] == scorer_id and s["metric_key"] == metric_key:
                            if s["value"] > best_score:
                                best_score = s["value"]
                                best_asset_id = asset.id
                            break
            if best_asset_id is None:
                best_asset_id = next(
                    (a.id for a, _d, is_rep in members if is_rep), members[0][0].id
                )
            picks.append((cluster_id, best_asset_id))
    return picks


def export_picks(
    conn: sqlite3.Connection,
    picks: Sequence[tuple[int, int]],
    photos_root: Path,
    export_dir: Path,
    *,
    folder_mode: str = "flat",
    categories: Sequence[Category] | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    on_copied: Callable[[int, str], None] | None = None,
) -> ExportResult:
    """Copy the picked assets into *export_dir*, sorted into folders by *folder_mode*."""
    from takeout_rater.db.queries import get_asset_by_id

    folders = assign_folders(
        conn,
        [aid for _cid, aid in picks],
        folder_mode,
        categories=categories,
        min_confidence=min_confidence,
    )
    result = ExportResult()
    for cluster_id, asset_id in picks:
        asset = get_asset_by_id(conn, asset_id)
        if asset is None:
            continue
        src = photos_root / asset.relpath
        if not src.exists():
            result.skipped += 1
            continue
        folder = folders.get(asset_id, "")
        dest_dir = export_dir / folder if folder else export_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_dir / f"cluster{cluster_id:06d}_{asset.filename}")
        result.copied += 1
        result.folders[folder] += 1
        if on_copied is not None:
            on_copied(result.copied, folder)
    return result
