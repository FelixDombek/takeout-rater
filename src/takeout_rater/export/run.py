"""Best-of-cluster export shared by the CLI and the background export job."""

from __future__ import annotations

import contextlib
import json
import logging
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

_log = logging.getLogger(__name__)

#: Lists the files the last export wrote, so the next one can remove them.
MANIFEST_NAME = ".takeout-rater-export.json"


@dataclass
class ExportResult:
    copied: int = 0
    skipped: int = 0
    #: Number of exported files per relative folder (``""`` = export root).
    folders: Counter[str] = field(default_factory=Counter)
    #: Copies from the previous export that were removed before this one.
    removed: int = 0


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
    """Copy the picked assets into *export_dir*, sorted into folders by *folder_mode*.

    Files written by the previous export into *export_dir* (listed in its
    manifest) are removed first, so changing the folder layout, the category
    labels or a trip name does not leave stale copies behind.  Files the user
    put there are never touched.
    """
    from takeout_rater.db.queries import get_asset_by_id

    folders = assign_folders(
        conn,
        [aid for _cid, aid in picks],
        folder_mode,
        categories=categories,
        min_confidence=min_confidence,
    )
    result = ExportResult()
    result.removed = _remove_previous_export(export_dir)
    written: list[str] = []
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
        dest = dest_dir / f"cluster{cluster_id:06d}_{asset.filename}"
        shutil.copy2(src, dest)
        written.append(dest.relative_to(export_dir).as_posix())
        result.copied += 1
        result.folders[folder] += 1
        if on_copied is not None:
            on_copied(result.copied, folder)
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / MANIFEST_NAME).write_text(json.dumps({"files": written}), encoding="utf-8")
    return result


def _remove_previous_export(export_dir: Path) -> int:
    """Delete the files listed in *export_dir*'s manifest; prune emptied folders."""
    manifest = export_dir / MANIFEST_NAME
    try:
        files = json.loads(manifest.read_text(encoding="utf-8")).get("files", [])
    except (OSError, ValueError, AttributeError):
        return 0
    root = export_dir.resolve()
    removed = 0
    dirs: set[Path] = set()
    for rel in files:
        if not isinstance(rel, str):
            continue
        path = (export_dir / rel).resolve()
        if path == root or root not in path.parents:
            continue  # never delete outside the export folder
        try:
            path.unlink()
            removed += 1
        except FileNotFoundError:
            pass
        except OSError as exc:
            _log.warning("Could not remove previous export file %s: %s", path, exc)
            continue
        parent = path.parent
        while parent != root and root in parent.parents:
            dirs.add(parent)
            parent = parent.parent
    for d in sorted(dirs, key=lambda p: len(p.parts), reverse=True):
        with contextlib.suppress(OSError):
            d.rmdir()  # only succeeds when empty
    manifest.unlink(missing_ok=True)
    return removed
