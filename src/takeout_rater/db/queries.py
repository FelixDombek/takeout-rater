"""Query helpers for the takeout-rater SQLite database.

All functions accept a :class:`sqlite3.Connection` and return plain Python
types (``dict``, ``list``, ``int``, ``None``).

Rows fetched from the DB are :class:`sqlite3.Row` objects, which support
both index and column-name access.

Constants
---------
CURRENT_INDEXER_VERSION : int
    Monotonically-increasing integer tracking which version of the indexing
    pipeline last processed each asset.  Increment this constant whenever the
    indexing logic changes in a way that requires re-processing existing
    assets.  Assets with ``indexer_version IS NULL`` or
    ``indexer_version < CURRENT_INDEXER_VERSION`` are candidates for the
    "Rescan library" job.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Indexer versioning
# ---------------------------------------------------------------------------

#: Increment this constant whenever the indexing pipeline changes in a way
#: that requires existing assets to be re-processed.  Version 1 is the
#: baseline introduced with the "Rescan library" feature.  Version 2 adds
#: thumbnail regeneration to the rescan pipeline (HEIC/orientation fixes).
CURRENT_INDEXER_VERSION: int = 2


@dataclass
class AssetRow:
    """Lightweight view of one row from the ``assets`` table."""

    id: int
    relpath: str
    filename: str
    ext: str
    size_bytes: int | None
    sha256: str | None
    taken_at: int | None
    created_at_sidecar: int | None
    width: int | None
    height: int | None
    geo_lat: float | None
    geo_lon: float | None
    geo_alt: float | None
    geo_exif_lat: float | None
    geo_exif_lon: float | None
    geo_exif_alt: float | None
    title: str | None
    description: str | None
    image_views: int | None
    google_photos_url: str | None
    favorited: bool | None
    archived: bool | None
    trashed: bool | None
    origin_type: str | None
    origin_device_type: str | None
    origin_device_folder: str | None
    app_source_package: str | None
    sidecar_relpath: str | None
    mime: str | None
    indexed_at: int


def _row_to_asset(row: sqlite3.Row) -> AssetRow:
    """Convert a :class:`sqlite3.Row` from the ``assets`` table to an :class:`AssetRow`."""
    return AssetRow(
        id=row["id"],
        relpath=row["relpath"],
        filename=row["filename"],
        ext=row["ext"],
        size_bytes=row["size_bytes"],
        sha256=row["sha256"],
        taken_at=row["taken_at"],
        created_at_sidecar=row["created_at_sidecar"],
        width=row["width"],
        height=row["height"],
        geo_lat=row["geo_lat"],
        geo_lon=row["geo_lon"],
        geo_alt=row["geo_alt"],
        geo_exif_lat=row["geo_exif_lat"],
        geo_exif_lon=row["geo_exif_lon"],
        geo_exif_alt=row["geo_exif_alt"],
        title=row["title"],
        description=row["description"],
        image_views=row["image_views"],
        google_photos_url=row["google_photos_url"],
        favorited=bool(row["favorited"]) if row["favorited"] is not None else None,
        archived=bool(row["archived"]) if row["archived"] is not None else None,
        trashed=bool(row["trashed"]) if row["trashed"] is not None else None,
        origin_type=row["origin_type"],
        origin_device_type=row["origin_device_type"],
        origin_device_folder=row["origin_device_folder"],
        app_source_package=row["app_source_package"],
        sidecar_relpath=row["sidecar_relpath"],
        mime=row["mime"],
        indexed_at=row["indexed_at"],
    )


def upsert_asset(conn: sqlite3.Connection, asset: dict[str, Any]) -> int:
    """Insert or update an asset row and return the asset ID.

    When a SHA-256 hash is provided and an existing asset already has the same
    hash, the new path is recorded as an alias in ``asset_paths`` rather than
    creating a second ``assets`` row.  This ensures each unique binary file has
    exactly one ``assets`` entry regardless of how many locations it appears in
    within the Takeout archive.

    When no SHA-256 is available, falls back to ``INSERT … ON CONFLICT(relpath)
    DO UPDATE`` so re-indexing is still idempotent.

    Args:
        conn: Open database connection.
        asset: Dict with column names as keys.  ``indexed_at`` defaults to
            the current Unix timestamp if not provided.

    Returns:
        The row ID of the canonical ``assets`` row for this file.
    """
    asset = dict(asset)
    if "indexed_at" not in asset:
        asset["indexed_at"] = int(time.time())

    sha256 = asset.get("sha256")
    relpath = asset["relpath"]
    indexed_at = asset["indexed_at"]

    if sha256:
        # Check whether a canonical asset with this SHA-256 already exists.
        existing = conn.execute(
            "SELECT id, relpath FROM assets WHERE sha256 = ? ORDER BY id LIMIT 1",
            (sha256,),
        ).fetchone()
        if existing:
            canonical_id: int = existing[0]
            canonical_relpath: str = existing[1]
            if relpath != canonical_relpath:
                # This is a secondary (alias) path — record it and return the
                # canonical asset's id without touching the assets table.
                conn.execute(
                    "INSERT INTO asset_paths (asset_id, relpath, indexed_at)"
                    " VALUES (?, ?, ?)"
                    " ON CONFLICT(relpath) DO UPDATE"
                    "   SET asset_id = excluded.asset_id, indexed_at = excluded.indexed_at",
                    (canonical_id, relpath, indexed_at),
                )
                conn.commit()
                return canonical_id
            # Fall through: this is the canonical path being re-indexed — update
            # the assets row normally via the ON CONFLICT path below.

    # If this relpath was previously recorded as an alias in asset_paths
    # (e.g. the canonical was removed and this is now the only copy), clean it
    # up so it can become a proper assets row.
    conn.execute("DELETE FROM asset_paths WHERE relpath = ?", (relpath,))

    # Exclude 'id' from the insert columns to let SQLite assign it
    insert_cols = [k for k in asset if k != "id"]
    columns = ", ".join(insert_cols)
    placeholders = ", ".join("?" for _ in insert_cols)
    # On conflict, update all columns except relpath (the conflict key) and id
    update_pairs = ", ".join(f"{k} = excluded.{k}" for k in insert_cols if k != "relpath")
    sql = (
        f"INSERT INTO assets ({columns}) VALUES ({placeholders})"  # noqa: S608
        f" ON CONFLICT(relpath) DO UPDATE SET {update_pairs}"
        " RETURNING id"
    )
    row = conn.execute(sql, [asset[k] for k in insert_cols]).fetchone()
    conn.commit()
    return row[0]


def get_asset_by_id(conn: sqlite3.Connection, asset_id: int) -> AssetRow | None:
    """Fetch one asset by its primary key.

    Args:
        conn: Open database connection.
        asset_id: The integer ``id`` of the asset.

    Returns:
        An :class:`AssetRow`, or ``None`` if not found.
    """
    row = conn.execute("SELECT * FROM assets WHERE id = ?", (asset_id,)).fetchone()
    return _row_to_asset(row) if row else None


def get_asset_by_relpath(conn: sqlite3.Connection, relpath: str) -> AssetRow | None:
    """Fetch one asset by its relative path.

    Checks both the canonical ``assets.relpath`` column and the
    ``asset_paths`` alias table, returning the canonical asset in either case.

    Args:
        conn: Open database connection.
        relpath: Path relative to the takeout root.

    Returns:
        An :class:`AssetRow`, or ``None`` if not found.
    """
    row = conn.execute("SELECT * FROM assets WHERE relpath = ?", (relpath,)).fetchone()
    if row:
        return _row_to_asset(row)
    # Check alias paths — return the canonical asset for this alias.
    alias = conn.execute(
        "SELECT asset_id FROM asset_paths WHERE relpath = ?", (relpath,)
    ).fetchone()
    if alias:
        row = conn.execute("SELECT * FROM assets WHERE id = ?", (alias[0],)).fetchone()
        return _row_to_asset(row) if row else None
    return None


def lookup_sha256(conn: sqlite3.Connection, sha256: str) -> tuple[int, bool] | None:
    """Check if an asset with the given SHA-256 hash already exists.

    Args:
        conn: Open database connection.
        sha256: The hex-encoded SHA-256 content hash.

    Returns:
        A tuple ``(asset_id, has_sidecar)`` if an asset with this hash exists,
        or ``None`` if no match is found. ``has_sidecar`` is ``True`` if the
        canonical asset row has a non-NULL ``sidecar_relpath``.
    """
    row = conn.execute(
        "SELECT id, sidecar_relpath FROM assets WHERE sha256 = ? LIMIT 1", (sha256,)
    ).fetchone()
    if row:
        return (row[0], row[1] is not None)
    return None


def list_assets(
    conn: sqlite3.Connection,
    *,
    limit: int = 100,
    offset: int = 0,
    order_by: str = "taken_at DESC",
    favorited: bool | None = None,
    archived: bool | None = None,
    trashed: bool | None = None,
    ext: str | None = None,
) -> list[AssetRow]:
    """Return a paginated list of assets with optional filters.

    Args:
        conn: Open database connection.
        limit: Maximum number of rows to return.
        offset: Number of rows to skip (for pagination).
        order_by: SQL ``ORDER BY`` clause fragment.  Must be a trusted value
            (not user-supplied without validation).
        favorited: If not ``None``, filter to assets where ``favorited`` matches.
        archived: If not ``None``, filter to assets where ``archived`` matches.
        trashed: If not ``None``, filter to assets where ``trashed`` matches.
        ext: If not ``None``, filter to assets with this file extension (e.g. ``".jpg"``).

    Returns:
        List of :class:`AssetRow` objects.
    """
    conditions: list[str] = []
    params: list[Any] = []

    if favorited is not None:
        conditions.append("favorited = ?")
        params.append(1 if favorited else 0)
    if archived is not None:
        conditions.append("archived = ?")
        params.append(1 if archived else 0)
    if trashed is not None:
        conditions.append("trashed = ?")
        params.append(1 if trashed else 0)
    if ext is not None:
        conditions.append("ext = ?")
        params.append(ext.lower())

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    # order_by is a trusted, code-controlled value — not user input
    sql = f"SELECT * FROM assets {where} ORDER BY {order_by} LIMIT ? OFFSET ?"  # noqa: S608
    params.extend([limit, offset])

    rows = conn.execute(sql, params).fetchall()
    return [_row_to_asset(r) for r in rows]


def count_assets(
    conn: sqlite3.Connection,
    *,
    favorited: bool | None = None,
    archived: bool | None = None,
    trashed: bool | None = None,
    ext: str | None = None,
) -> int:
    """Return the total number of assets matching the given filters.

    Args:
        conn: Open database connection.
        favorited: Optional favorited filter.
        archived: Optional archived filter.
        trashed: Optional trashed filter.
        ext: Optional file-extension filter.

    Returns:
        Integer count.
    """
    conditions: list[str] = []
    params: list[Any] = []

    if favorited is not None:
        conditions.append("favorited = ?")
        params.append(1 if favorited else 0)
    if archived is not None:
        conditions.append("archived = ?")
        params.append(1 if archived else 0)
    if trashed is not None:
        conditions.append("trashed = ?")
        params.append(1 if trashed else 0)
    if ext is not None:
        conditions.append("ext = ?")
        params.append(ext.lower())

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    sql = f"SELECT COUNT(*) FROM assets {where}"  # noqa: S608
    return conn.execute(sql, params).fetchone()[0]


def asset_relpath_to_path(asset: AssetRow, takeout_root: Path) -> Path:
    """Return the absolute path to an asset given its database row and the takeout root."""
    return takeout_root / asset.relpath


# ---------------------------------------------------------------------------
# SHA-256 / content-deduplication helpers
# ---------------------------------------------------------------------------


def list_asset_ids_without_sha256(conn: sqlite3.Connection) -> list[int]:
    """Return IDs of assets that do not yet have a stored SHA-256 hash.

    Args:
        conn: Open database connection.

    Returns:
        List of integer asset IDs ordered by ID.
    """
    rows = conn.execute("SELECT id FROM assets WHERE sha256 IS NULL ORDER BY id").fetchall()
    return [row[0] for row in rows]


def update_asset_sha256(conn: sqlite3.Connection, asset_id: int, sha256: str) -> None:
    """Store the SHA-256 content hash for a single asset.

    Args:
        conn: Open database connection.
        asset_id: Primary key of the asset to update.
        sha256: Hex-encoded SHA-256 digest of the image file.
    """
    conn.execute("UPDATE assets SET sha256 = ? WHERE id = ?", (sha256, asset_id))
    conn.commit()


def list_assets_deduped(
    conn: sqlite3.Connection,
    *,
    limit: int = 100,
    offset: int = 0,
    order_by: str = "taken_at DESC",
    favorited: bool | None = None,
    archived: bool | None = None,
    trashed: bool | None = None,
    ext: str | None = None,
) -> list[AssetRow]:
    """Return a deduplicated, paginated list of assets.

    For assets that share the same SHA-256 content hash, only the row with the
    lowest ``id`` (the first-indexed copy) is returned.  Assets that have no
    SHA-256 yet are each shown individually (no merging).

    Args:
        conn: Open database connection.
        limit: Maximum number of rows to return.
        offset: Rows to skip (for pagination).
        order_by: SQL ``ORDER BY`` clause fragment (trusted, code-controlled).
        favorited: Optional favorited filter.
        archived: Optional archived filter.
        trashed: Optional trashed filter.
        ext: Optional file-extension filter (e.g. ``".jpg"``).

    Returns:
        List of :class:`AssetRow` objects.
    """
    conditions: list[str] = []
    params: list[Any] = []

    if favorited is not None:
        conditions.append("favorited = ?")
        params.append(1 if favorited else 0)
    if archived is not None:
        conditions.append("archived = ?")
        params.append(1 if archived else 0)
    if trashed is not None:
        conditions.append("trashed = ?")
        params.append(1 if trashed else 0)
    if ext is not None:
        conditions.append("ext = ?")
        params.append(ext.lower())

    # Each sha256 group is represented by its minimum id; assets without sha256
    # are partitioned individually via COALESCE with a unique per-row key.
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    sql = (  # noqa: S608
        f"SELECT * FROM ("
        f"  SELECT *, ROW_NUMBER() OVER ("
        f"    PARTITION BY COALESCE(sha256, CAST(id AS TEXT))"
        f"    ORDER BY id"
        f"  ) AS _rn"
        f"  FROM assets {where}"
        f") WHERE _rn = 1"
        f" ORDER BY {order_by}"
        f" LIMIT ? OFFSET ?"
    )
    params.extend([limit, offset])
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_asset(r) for r in rows]


def count_assets_deduped(
    conn: sqlite3.Connection,
    *,
    favorited: bool | None = None,
    archived: bool | None = None,
    trashed: bool | None = None,
    ext: str | None = None,
) -> int:
    """Count unique images after SHA-256 deduplication.

    For assets that share a SHA-256 hash, only one is counted.  Assets without
    a SHA-256 are each counted individually.

    Args:
        conn: Open database connection.
        favorited: Optional favorited filter.
        archived: Optional archived filter.
        trashed: Optional trashed filter.
        ext: Optional file-extension filter.

    Returns:
        Integer count of unique images.
    """
    conditions: list[str] = []
    params: list[Any] = []

    if favorited is not None:
        conditions.append("favorited = ?")
        params.append(1 if favorited else 0)
    if archived is not None:
        conditions.append("archived = ?")
        params.append(1 if archived else 0)
    if trashed is not None:
        conditions.append("trashed = ?")
        params.append(1 if trashed else 0)
    if ext is not None:
        conditions.append("ext = ?")
        params.append(ext.lower())

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    # Count distinct sha256 groups (NULL-sha256 assets each count as 1 via COALESCE).
    # Uses COUNT(DISTINCT …) to avoid a subquery that can return NULL on some
    # SQLite/platform combinations, causing fetchone() to return None.
    sql = (  # noqa: S608
        f"SELECT COUNT(DISTINCT COALESCE(sha256, CAST(id AS TEXT))) FROM assets {where}"
    )
    return (conn.execute(sql, params).fetchone() or (0,))[0]


def get_duplicate_assets(conn: sqlite3.Connection, sha256: str) -> list[AssetRow]:
    """Return all assets that share the given SHA-256 content hash.

    This is used by the detail view to show all physical file locations for a
    deduplicated image.

    Args:
        conn: Open database connection.
        sha256: Hex-encoded SHA-256 digest to look up.

    Returns:
        List of :class:`AssetRow` objects ordered by ``id``.  Returns an empty
        list if *sha256* is not found in the database.
    """
    rows = conn.execute(
        "SELECT * FROM assets WHERE sha256 = ? ORDER BY id",
        (sha256,),
    ).fetchall()
    return [_row_to_asset(r) for r in rows]


def get_asset_alias_paths(conn: sqlite3.Connection, asset_id: int) -> list[str]:
    """Return all alias (secondary) paths stored for *asset_id*.

    When binary-duplicate files are indexed they are merged into a single
    ``assets`` row (the canonical one).  The other paths at which the same
    file appeared in the Takeout archive are stored in ``asset_paths``.
    This function returns those secondary paths in alphabetical order.

    Args:
        conn: Open database connection.
        asset_id: Primary key of the canonical asset.

    Returns:
        List of relative-path strings.  Empty when the asset has no known
        aliases (i.e. it appears in only one location).
    """
    rows = conn.execute(
        "SELECT relpath FROM asset_paths WHERE asset_id = ? ORDER BY relpath",
        (asset_id,),
    ).fetchall()
    return [row[0] for row in rows]


# ---------------------------------------------------------------------------
# Timeline helpers
# ---------------------------------------------------------------------------


def get_taken_at_range(
    conn: sqlite3.Connection,
) -> tuple[int | None, int | None]:
    """Return ``(min_taken_at, max_taken_at)`` for assets with a non-NULL ``taken_at``.

    Args:
        conn: Open database connection.

    Returns:
        A tuple of Unix timestamps, or ``(None, None)`` if no assets have
        ``taken_at`` data.
    """
    row = conn.execute(
        "SELECT MIN(taken_at), MAX(taken_at) FROM assets WHERE taken_at IS NOT NULL"
    ).fetchone()
    if row is None or row[0] is None:
        return None, None
    return row[0], row[1]


def count_assets_newer_than(
    conn: sqlite3.Connection,
    timestamp: int,
    *,
    favorited: bool | None = None,
    deduped: bool = False,
) -> int:
    """Count assets with ``taken_at`` strictly greater than *timestamp*.

    Used to compute the 1-based page number for a given date when assets are
    sorted by ``taken_at DESC`` (newest first).  The page a timestamp falls on
    is ``count_assets_newer_than(...) // page_size + 1``.

    Args:
        conn: Open database connection.
        timestamp: Unix timestamp threshold.  Assets taken *after* this time
            are counted.
        favorited: Optional favorited filter.
        deduped: If ``True``, count only the representative (lowest ``id``) of
            each SHA-256 group, matching the behaviour of
            :func:`list_assets_deduped`.

    Returns:
        Number of assets with ``taken_at > timestamp``.
    """
    conditions: list[str] = ["taken_at > ?"]
    params: list[Any] = [timestamp]

    if favorited is not None:
        conditions.append("favorited = ?")
        params.append(1 if favorited else 0)

    where = f"WHERE {' AND '.join(conditions)}"

    if deduped:
        sql = (  # noqa: S608
            f"SELECT COUNT(*) FROM ("
            f"  SELECT 1"
            f"  FROM assets {where}"
            f"  GROUP BY COALESCE(sha256, CAST(id AS TEXT))"
            f")"
        )
    else:
        sql = f"SELECT COUNT(*) FROM assets {where}"  # noqa: S608

    return conn.execute(sql, params).fetchone()[0]


def upsert_asset_scores(
    conn: sqlite3.Connection,
    scorer_id: str,
    variant_id: str,
    scores: list[tuple[int, str, float]],
    *,
    scorer_version: str | None = None,
) -> None:
    """Insert or replace ``asset_scores`` rows.

    Each ``(asset_id, scorer_id, variant_id, metric_key)`` combination is
    unique.  Re-scoring the same asset simply overwrites the previous value.

    Args:
        conn: Open database connection.
        scorer_id: Scorer identifier (e.g. ``"simple"``).
        variant_id: Variant identifier (e.g. ``"blur"``).
        scores: Iterable of ``(asset_id, metric_key, value)`` triples.
        scorer_version: Optional version string stored for auditability.
    """
    if not scores:
        return

    now = int(time.time())
    conn.executemany(
        "INSERT OR REPLACE INTO asset_scores"
        " (asset_id, scorer_id, variant_id, metric_key, value, scorer_version, scored_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (asset_id, scorer_id, variant_id, metric_key, value, scorer_version, now)
            for asset_id, metric_key, value in scores
        ],
    )
    conn.commit()


def get_asset_scores(
    conn: sqlite3.Connection,
    asset_id: int,
) -> list[dict[str, Any]]:
    """Return all scores for a single asset.

    Args:
        conn: Open database connection.
        asset_id: The asset to look up.

    Returns:
        List of dicts with keys ``scorer_id``, ``variant_id``, ``metric_key``,
        ``value``, and ``scored_at``.  Ordered alphabetically by
        ``scorer_id``, ``variant_id``, ``metric_key``.
    """
    rows = conn.execute(
        "SELECT scorer_id, variant_id, metric_key, value, scored_at"
        " FROM asset_scores"
        " WHERE asset_id = ?"
        " ORDER BY scorer_id, variant_id, metric_key",
        (asset_id,),
    ).fetchall()
    return [
        {
            "scorer_id": row["scorer_id"],
            "variant_id": row["variant_id"],
            "metric_key": row["metric_key"],
            "value": row["value"],
            "scored_at": row["scored_at"],
        }
        for row in rows
    ]


def has_scores_for(
    conn: sqlite3.Connection,
    scorer_id: str,
    variant_id: str,
) -> bool:
    """Return whether any scores exist for the given scorer+variant.

    Args:
        conn: Open database connection.
        scorer_id: Scorer identifier.
        variant_id: Variant identifier.

    Returns:
        ``True`` if at least one score row exists.
    """
    row = conn.execute(
        "SELECT 1 FROM asset_scores WHERE scorer_id = ? AND variant_id = ? LIMIT 1",
        (scorer_id, variant_id),
    ).fetchone()
    return row is not None


def list_available_score_metrics_with_variants(
    conn: sqlite3.Connection,
) -> set[tuple[str, str, str]]:
    """Return the set of (scorer_id, variant_id, metric_key) triples with scored results.

    Returns:
        A set of ``(scorer_id, variant_id, metric_key)`` tuples.
    """
    rows = conn.execute(
        "SELECT DISTINCT scorer_id, variant_id, metric_key FROM asset_scores"
    ).fetchall()
    return {(row[0], row[1], row[2]) for row in rows}


def list_assets_by_score(
    conn: sqlite3.Connection,
    scorer_id: str,
    metric_key: str,
    variant_id: str,
    *,
    limit: int = 100,
    offset: int = 0,
    descending: bool = True,
    favorited: bool | None = None,
    trashed: bool | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
) -> list[tuple[AssetRow, float]]:
    """Return assets that have been scored, sorted by score value.

    Assets without a score for the given ``scorer_id`` + ``variant_id`` +
    ``metric_key`` are excluded.

    Args:
        conn: Open database connection.
        scorer_id: Scorer whose scores to sort by.
        metric_key: Metric key to sort by.
        variant_id: Variant identifier.
        limit: Maximum number of rows.
        offset: Rows to skip (for pagination).
        descending: ``True`` (default) → highest score first.
        favorited: Optional favorited filter.
        trashed: Optional trashed filter.
        min_score: Optional inclusive lower bound on the score value.
        max_score: Optional inclusive upper bound on the score value.

    Returns:
        List of ``(AssetRow, score_value)`` tuples.
    """
    if not metric_key:
        raise ValueError("metric_key must be a non-empty string")

    order = "DESC" if descending else "ASC"
    conditions: list[str] = [
        "s.scorer_id = ?",
        "s.variant_id = ?",
        "s.metric_key = ?",
    ]
    params: list[Any] = [scorer_id, variant_id, metric_key]

    if favorited is not None:
        conditions.append("a.favorited = ?")
        params.append(1 if favorited else 0)
    if trashed is not None:
        conditions.append("a.trashed = ?")
        params.append(1 if trashed else 0)
    if min_score is not None:
        conditions.append("s.value >= ?")
        params.append(min_score)
    if max_score is not None:
        conditions.append("s.value <= ?")
        params.append(max_score)

    where = "WHERE " + " AND ".join(conditions)
    # order is a code-controlled literal, not user input  # noqa: S608
    sql = (
        f"SELECT a.*, s.value AS score_value"  # noqa: S608
        f" FROM assets a"
        f" JOIN asset_scores s ON s.asset_id = a.id"
        f" {where}"
        f" ORDER BY s.value {order}"
        f" LIMIT ? OFFSET ?"
    )
    params.extend([limit, offset])
    rows = conn.execute(sql, params).fetchall()
    return [(_row_to_asset(row), row["score_value"]) for row in rows]


def count_assets_with_score(
    conn: sqlite3.Connection,
    scorer_id: str,
    metric_key: str,
    variant_id: str,
    *,
    favorited: bool | None = None,
    trashed: bool | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
) -> int:
    """Count assets that have a score for a scorer+variant+metric.

    Args:
        conn: Open database connection.
        scorer_id: Scorer ID.
        metric_key: Metric key.
        variant_id: Variant identifier.
        favorited: Optional favorited filter.
        trashed: Optional trashed filter.
        min_score: Optional inclusive lower bound on the score value.
        max_score: Optional inclusive upper bound on the score value.

    Returns:
        Integer count.
    """
    conditions: list[str] = [
        "s.scorer_id = ?",
        "s.variant_id = ?",
        "s.metric_key = ?",
    ]
    params: list[Any] = [scorer_id, variant_id, metric_key]

    if favorited is not None:
        conditions.append("a.favorited = ?")
        params.append(1 if favorited else 0)
    if trashed is not None:
        conditions.append("a.trashed = ?")
        params.append(1 if trashed else 0)
    if min_score is not None:
        conditions.append("s.value >= ?")
        params.append(min_score)
    if max_score is not None:
        conditions.append("s.value <= ?")
        params.append(max_score)

    where = "WHERE " + " AND ".join(conditions)
    sql = f"SELECT COUNT(*) FROM assets a JOIN asset_scores s ON s.asset_id = a.id {where}"  # noqa: S608
    return conn.execute(sql, params).fetchone()[0]


# ---------------------------------------------------------------------------
# Multi-criteria sort helpers
# ---------------------------------------------------------------------------


@dataclass
class SortCriterion:
    """One level of a multi-key sort/filter specification."""

    scorer_id: str
    variant_id: str
    metric_key: str
    min_score: float | None = None
    max_score: float | None = None


def list_assets_multi_sort(
    conn: sqlite3.Connection,
    criteria: list[SortCriterion],
    *,
    limit: int = 100,
    offset: int = 0,
    favorited: bool | None = None,
    trashed: bool | None = None,
) -> list[tuple[AssetRow, float]]:
    """Return assets sorted by multiple score criteria.

    The *first* criterion is mandatory (INNER JOIN); subsequent ones use LEFT
    JOINs so that assets without a secondary score still appear (sorted last).

    Args:
        conn: Open database connection.
        criteria: Ordered list of sort levels.  Must contain at least one entry.
        limit: Maximum number of rows.
        offset: Rows to skip (for pagination).
        favorited: Optional favorited filter (applied at the asset level).
        trashed: Optional trashed filter.

    Returns:
        List of ``(AssetRow, primary_score_value)`` tuples.
    """
    if not criteria:
        raise ValueError("criteria must contain at least one SortCriterion")

    c0 = criteria[0]
    select_cols = ["a.*", "s1.value AS score_value"]
    joins = [
        "INNER JOIN asset_scores s1"
        " ON s1.asset_id = a.id"
        " AND s1.scorer_id = ?"
        " AND s1.variant_id = ?"
        " AND s1.metric_key = ?"
    ]
    join_params: list[Any] = [c0.scorer_id, c0.variant_id, c0.metric_key]
    order_cols = ["s1.value DESC"]

    for i, c in enumerate(criteria[1:], start=2):
        alias = f"s{i}"
        select_cols.append(f"{alias}.value AS score_value_{i}")
        joins.append(
            f"LEFT JOIN asset_scores {alias}"
            f" ON {alias}.asset_id = a.id"
            f" AND {alias}.scorer_id = ?"
            f" AND {alias}.variant_id = ?"
            f" AND {alias}.metric_key = ?"
        )
        join_params.extend([c.scorer_id, c.variant_id, c.metric_key])
        order_cols.append(f"{alias}.value DESC NULLS LAST")

    conditions: list[str] = []
    where_params: list[Any] = []

    if favorited is not None:
        conditions.append("a.favorited = ?")
        where_params.append(1 if favorited else 0)
    if trashed is not None:
        conditions.append("a.trashed = ?")
        where_params.append(1 if trashed else 0)

    # Range filters for all criteria (including primary).
    for i, c in enumerate(criteria, start=1):
        alias = f"s{i}"
        if c.min_score is not None:
            conditions.append(f"{alias}.value >= ?")
            where_params.append(c.min_score)
        if c.max_score is not None:
            conditions.append(f"{alias}.value <= ?")
            where_params.append(c.max_score)

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    join_sql = " ".join(joins)
    select_sql = ", ".join(select_cols)
    order_sql = ", ".join(order_cols)

    # All literals are code-controlled, not user input.  # noqa: S608
    sql = (
        f"SELECT {select_sql}"
        f" FROM assets a"
        f" {join_sql}"
        f" {where_clause}"
        f" ORDER BY {order_sql}"
        f" LIMIT ? OFFSET ?"
    )
    params = join_params + where_params + [limit, offset]
    rows = conn.execute(sql, params).fetchall()
    return [(_row_to_asset(row), row["score_value"]) for row in rows]


def count_assets_multi_sort(
    conn: sqlite3.Connection,
    criteria: list[SortCriterion],
    *,
    favorited: bool | None = None,
    trashed: bool | None = None,
) -> int:
    """Count assets that satisfy a multi-criteria sort/filter specification.

    Mirrors the filtering logic of :func:`list_assets_multi_sort`.

    Args:
        conn: Open database connection.
        criteria: Ordered list of sort levels.
        favorited: Optional favorited filter.
        trashed: Optional trashed filter.

    Returns:
        Integer count.
    """
    if not criteria:
        raise ValueError("criteria must contain at least one SortCriterion")

    c0 = criteria[0]
    joins = [
        "INNER JOIN asset_scores s1"
        " ON s1.asset_id = a.id"
        " AND s1.scorer_id = ?"
        " AND s1.variant_id = ?"
        " AND s1.metric_key = ?"
    ]
    join_params: list[Any] = [c0.scorer_id, c0.variant_id, c0.metric_key]

    for i, c in enumerate(criteria[1:], start=2):
        alias = f"s{i}"
        joins.append(
            f"LEFT JOIN asset_scores {alias}"
            f" ON {alias}.asset_id = a.id"
            f" AND {alias}.scorer_id = ?"
            f" AND {alias}.variant_id = ?"
            f" AND {alias}.metric_key = ?"
        )
        join_params.extend([c.scorer_id, c.variant_id, c.metric_key])

    conditions: list[str] = []
    where_params: list[Any] = []

    if favorited is not None:
        conditions.append("a.favorited = ?")
        where_params.append(1 if favorited else 0)
    if trashed is not None:
        conditions.append("a.trashed = ?")
        where_params.append(1 if trashed else 0)

    for i, c in enumerate(criteria, start=1):
        alias = f"s{i}"
        if c.min_score is not None:
            conditions.append(f"{alias}.value >= ?")
            where_params.append(c.min_score)
        if c.max_score is not None:
            conditions.append(f"{alias}.value <= ?")
            where_params.append(c.max_score)

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    join_sql = " ".join(joins)

    sql = (  # noqa: S608
        f"SELECT COUNT(*) FROM assets a {join_sql} {where_clause}"
    )
    params = join_params + where_params
    return conn.execute(sql, params).fetchone()[0]


# ---------------------------------------------------------------------------
# pHash helpers
# ---------------------------------------------------------------------------


def upsert_phash(
    conn: sqlite3.Connection,
    asset_id: int,
    phash_hex: str,
    algo: str = "dhash16",
) -> None:
    """Insert or update the perceptual hash for an asset.

    Args:
        conn: Open database connection.
        asset_id: Foreign key into ``assets``.
        phash_hex: Hex-encoded hash string.
        algo: Hash algorithm name (default ``"dhash16"`` for the 256-bit
            difference hash).
    """
    conn.execute(
        "INSERT INTO phash (asset_id, phash_hex, algo, computed_at)"
        " VALUES (?, ?, ?, ?)"
        " ON CONFLICT(asset_id) DO UPDATE SET phash_hex = excluded.phash_hex,"
        "   algo = excluded.algo, computed_at = excluded.computed_at",
        (asset_id, phash_hex, algo, int(time.time())),
    )
    conn.commit()


def get_phash(conn: sqlite3.Connection, asset_id: int) -> dict[str, Any] | None:
    """Return the stored perceptual hash for an asset, or ``None``.

    Args:
        conn: Open database connection.
        asset_id: The asset to look up.

    Returns:
        Dict with keys ``phash_hex``, ``algo``, ``computed_at``, or ``None``.
    """
    row = conn.execute(
        "SELECT phash_hex, algo, computed_at FROM phash WHERE asset_id = ?",
        (asset_id,),
    ).fetchone()
    if row is None:
        return None
    return {"phash_hex": row["phash_hex"], "algo": row["algo"], "computed_at": row["computed_at"]}


def list_asset_ids_without_phash(conn: sqlite3.Connection, algo: str | None = None) -> list[int]:
    """Return IDs of assets that do not yet have a stored pHash.

    When *algo* is given, assets whose existing hash was computed with a
    *different* algorithm are also included, so they will be re-hashed with
    the current algorithm.

    Args:
        conn: Open database connection.
        algo: If provided, also return assets whose stored ``phash.algo``
            does not match this value (i.e. hashes that need to be recomputed
            for the new algorithm).

    Returns:
        List of integer asset IDs.
    """
    if algo is None:
        rows = conn.execute(
            "SELECT a.id FROM assets a"
            " LEFT JOIN phash p ON p.asset_id = a.id"
            " WHERE p.asset_id IS NULL"
            " ORDER BY a.id",
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT a.id FROM assets a"
            " LEFT JOIN phash p ON p.asset_id = a.id"
            " WHERE p.asset_id IS NULL OR p.algo != ?"
            " ORDER BY a.id",
            (algo,),
        ).fetchall()
    return [row[0] for row in rows]


def list_all_asset_ids(conn: sqlite3.Connection) -> list[int]:
    """Return the IDs of all assets ordered by ID.

    This is a lightweight alternative to :func:`list_assets` when only the
    IDs are needed (e.g. in the scoring pipeline when re-scoring all assets).

    Args:
        conn: Open database connection.

    Returns:
        List of integer asset IDs.
    """
    rows = conn.execute("SELECT id FROM assets ORDER BY id").fetchall()
    return [row[0] for row in rows]


def list_all_phashes(conn: sqlite3.Connection, algo: str | None = None) -> list[tuple[int, str]]:
    """Return stored perceptual hashes as ``(asset_id, phash_hex)`` pairs.

    Args:
        conn: Open database connection.
        algo: When given, only return hashes whose ``algo`` column matches
            this value.  When ``None`` (default) all hashes are returned
            regardless of algorithm.

    Returns:
        List of ``(asset_id, phash_hex)`` tuples ordered by ``asset_id``.
    """
    if algo is None:
        rows = conn.execute("SELECT asset_id, phash_hex FROM phash ORDER BY asset_id").fetchall()
    else:
        rows = conn.execute(
            "SELECT asset_id, phash_hex FROM phash WHERE algo = ? ORDER BY asset_id",
            (algo,),
        ).fetchall()
    return [(row[0], row[1]) for row in rows]


def insert_clustering_run(
    conn: sqlite3.Connection,
    method: str,
    params_json: str | None,
) -> int:
    """Insert a new clustering_run row and return its ID.

    Args:
        conn: Open database connection.
        method: Algorithm identifier (e.g. ``"dhash_hamming"``).
        params_json: JSON-serialised clustering parameters, or ``None``.

    Returns:
        The integer primary key of the new ``clustering_runs`` row.
    """
    row = conn.execute(
        "INSERT INTO clustering_runs (method, params_json, created_at) VALUES (?, ?, ?)"
        " RETURNING id",
        (method, params_json, int(time.time())),
    ).fetchone()
    conn.commit()
    return row[0]


def list_clustering_runs(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return all clustering runs ordered by most-recent first.

    Each dict contains ``run_id``, ``method``, ``params_json``, ``created_at``,
    ``n_clusters``, and ``rep_asset_id`` (representative thumbnail from the
    first cluster of the run, or ``None``).
    """
    rows = conn.execute(
        "SELECT r.id AS run_id, r.method, r.params_json, r.created_at,"
        "   COUNT(DISTINCT c.id) AS n_clusters,"
        "   cm_rep.asset_id AS rep_asset_id"
        " FROM clustering_runs r"
        " LEFT JOIN clusters c ON c.run_id = r.id"
        " LEFT JOIN cluster_members cm_rep"
        "   ON cm_rep.cluster_id = (SELECT MIN(c2.id) FROM clusters c2 WHERE c2.run_id = r.id)"
        "   AND cm_rep.is_representative = 1"
        " GROUP BY r.id"
        " ORDER BY r.created_at DESC, r.id DESC",
    ).fetchall()
    return [
        {
            "run_id": row["run_id"],
            "method": row["method"],
            "params_json": row["params_json"],
            "created_at": row["created_at"],
            "n_clusters": row["n_clusters"],
            "rep_asset_id": row["rep_asset_id"],
        }
        for row in rows
    ]


def get_clustering_run(
    conn: sqlite3.Connection,
    run_id: int,
) -> dict[str, Any] | None:
    """Return metadata for a single clustering_run row, or ``None`` if not found."""
    row = conn.execute(
        "SELECT id, method, params_json, created_at FROM clustering_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "run_id": row["id"],
        "method": row["method"],
        "params_json": row["params_json"],
        "created_at": row["created_at"],
    }


def delete_clustering_run(conn: sqlite3.Connection, run_id: int) -> bool:
    """Delete a clustering run and all its clusters and members.

    Args:
        conn: Open database connection.
        run_id: The clustering run to delete.

    Returns:
        ``True`` if the run existed and was deleted; ``False`` if not found.
    """
    cluster_ids = [
        row[0]
        for row in conn.execute("SELECT id FROM clusters WHERE run_id = ?", (run_id,)).fetchall()
    ]
    if cluster_ids:
        conn.executemany(
            "DELETE FROM cluster_members WHERE cluster_id = ?",
            [(cid,) for cid in cluster_ids],
        )
        conn.execute("DELETE FROM clusters WHERE run_id = ?", (run_id,))
    result = conn.execute("DELETE FROM clustering_runs WHERE id = ?", (run_id,))
    conn.commit()
    return result.rowcount > 0


def list_clusters_for_run(
    conn: sqlite3.Connection,
    run_id: int,
) -> list[dict[str, Any]]:
    """Return all clusters for a clustering run, ordered by size (descending).

    Args:
        conn: Open database connection.
        run_id: Foreign key into ``clustering_runs``.

    Returns:
        List of dicts with keys: ``cluster_id``, ``member_count``,
        ``rep_asset_id``, ``rep_filename``, ``diameter``.
    """
    rows = conn.execute(
        "SELECT c.id AS cluster_id, c.diameter,"
        "   COUNT(cm.asset_id) AS member_count,"
        "   cm2.asset_id AS rep_asset_id,"
        "   a.filename AS rep_filename"
        " FROM clusters c"
        " JOIN cluster_members cm ON cm.cluster_id = c.id"
        " JOIN cluster_members cm2 ON cm2.cluster_id = c.id AND cm2.is_representative = 1"
        " JOIN assets a ON a.id = cm2.asset_id"
        " WHERE c.run_id = ?"
        " GROUP BY c.id"
        " ORDER BY member_count DESC, c.id",
        (run_id,),
    ).fetchall()
    return [
        {
            "cluster_id": row["cluster_id"],
            "member_count": row["member_count"],
            "rep_asset_id": row["rep_asset_id"],
            "rep_filename": row["rep_filename"],
            "diameter": row["diameter"],
        }
        for row in rows
    ]


def insert_cluster(
    conn: sqlite3.Connection,
    method: str,
    params_json: str | None,
    diameter: float | None = None,
    run_id: int | None = None,
) -> int:
    """Insert a new cluster row and return its ID.

    Args:
        conn: Open database connection.
        method: Algorithm identifier (e.g. ``"dhash_hamming"``).
        params_json: JSON-serialised clustering parameters.
        diameter: Intra-cluster diameter — the maximum pairwise Hamming
            distance among all cluster members.  ``None`` when unknown.
        run_id: Foreign key into ``clustering_runs``.  Must be provided for
            all new clusters (the column is ``NOT NULL`` in the schema).

    Returns:
        The integer primary key of the new ``clusters`` row.
    """
    row = conn.execute(
        "INSERT INTO clusters (method, params_json, created_at, diameter, run_id)"
        " VALUES (?, ?, ?, ?, ?) RETURNING id",
        (method, params_json, int(time.time()), diameter, run_id),
    ).fetchone()
    conn.commit()
    return row[0]


def bulk_insert_cluster_members(
    conn: sqlite3.Connection,
    cluster_id: int,
    members: list[tuple[int, float | None, int]],
) -> None:
    """Insert multiple ``cluster_members`` rows in a single transaction.

    Args:
        conn: Open database connection.
        cluster_id: Foreign key into ``clusters``.
        members: List of ``(asset_id, distance, is_representative)`` triples.
    """
    conn.executemany(
        "INSERT OR IGNORE INTO cluster_members (cluster_id, asset_id, distance, is_representative)"
        " VALUES (?, ?, ?, ?)",
        [(cluster_id, asset_id, distance, is_rep) for asset_id, distance, is_rep in members],
    )
    conn.commit()


def delete_clusters_by_method_params(
    conn: sqlite3.Connection,
    method: str,
    params_json: str | None,
) -> int:
    """Delete all clusters (and their members) for a given method + params.

    Args:
        conn: Open database connection.
        method: Algorithm identifier.
        params_json: JSON parameters string (exact match).

    Returns:
        Number of cluster rows deleted.
    """
    rows = conn.execute(
        "SELECT id FROM clusters WHERE method = ? AND params_json IS ?",
        (method, params_json),
    ).fetchall()
    cluster_ids = [row[0] for row in rows]
    if cluster_ids:
        conn.executemany(
            "DELETE FROM cluster_members WHERE cluster_id = ?",
            [(cid,) for cid in cluster_ids],
        )
        conn.executemany(
            "DELETE FROM clusters WHERE id = ?",
            [(cid,) for cid in cluster_ids],
        )
        conn.commit()
    return len(cluster_ids)


def delete_all_clusters(conn: sqlite3.Connection) -> int:
    """Delete all clustering runs, clusters, and their members from the database.

    Args:
        conn: Open database connection.

    Returns:
        Number of clustering run rows deleted.
    """
    with conn:
        count = conn.execute("SELECT COUNT(*) FROM clustering_runs").fetchone()[0]
        conn.execute("DELETE FROM cluster_members")
        conn.execute("DELETE FROM clusters")
        conn.execute("DELETE FROM clustering_runs")
    return count


def list_clusters_with_representatives(
    conn: sqlite3.Connection,
    *,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Return clusters with their representative asset info, ordered by size (descending).

    Args:
        conn: Open database connection.
        limit: Maximum number of clusters to return.
        offset: Number of clusters to skip (for pagination).

    Returns:
        List of dicts with keys: ``cluster_id``, ``method``, ``member_count``,
        ``rep_asset_id``, ``rep_filename``, ``created_at``, ``diameter``.
    """
    rows = conn.execute(
        "SELECT c.id AS cluster_id, c.method, c.created_at, c.diameter,"
        "   COUNT(cm.asset_id) AS member_count,"
        "   cm2.asset_id AS rep_asset_id,"
        "   a.filename AS rep_filename"
        " FROM clusters c"
        " JOIN cluster_members cm ON cm.cluster_id = c.id"
        " JOIN cluster_members cm2 ON cm2.cluster_id = c.id AND cm2.is_representative = 1"
        " JOIN assets a ON a.id = cm2.asset_id"
        " GROUP BY c.id"
        " ORDER BY member_count DESC, c.id"
        " LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    return [
        {
            "cluster_id": row["cluster_id"],
            "method": row["method"],
            "member_count": row["member_count"],
            "rep_asset_id": row["rep_asset_id"],
            "rep_filename": row["rep_filename"],
            "created_at": row["created_at"],
            "diameter": row["diameter"],
        }
        for row in rows
    ]


def count_clusters(conn: sqlite3.Connection) -> int:
    """Return the total number of clusters stored in the database.

    Args:
        conn: Open database connection.

    Returns:
        Integer count.
    """
    return conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]


def get_cluster_members(
    conn: sqlite3.Connection,
    cluster_id: int,
) -> list[tuple[Any, float | None, bool]]:
    """Return all members of a cluster as ``(AssetRow, distance, is_representative)`` triples.

    Args:
        conn: Open database connection.
        cluster_id: The cluster to look up.

    Returns:
        List of ``(AssetRow, distance, is_representative)`` triples.
        Representative is listed first.
    """
    rows = conn.execute(
        "SELECT a.*, cm.distance, cm.is_representative"
        " FROM cluster_members cm"
        " JOIN assets a ON a.id = cm.asset_id"
        " WHERE cm.cluster_id = ?"
        " ORDER BY cm.is_representative DESC, a.id",
        (cluster_id,),
    ).fetchall()
    return [(_row_to_asset(row), row["distance"], bool(row["is_representative"])) for row in rows]


def get_cluster_info(
    conn: sqlite3.Connection,
    cluster_id: int,
) -> dict[str, Any] | None:
    """Return metadata for a single cluster row.

    Args:
        conn: Open database connection.
        cluster_id: The cluster to look up.

    Returns:
        Dict with keys ``cluster_id``, ``method``, ``params_json``,
        ``diameter``, ``created_at``, ``run_id``, or ``None`` if not found.
    """
    row = conn.execute(
        "SELECT id, method, params_json, diameter, created_at, run_id FROM clusters WHERE id = ?",
        (cluster_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "cluster_id": row["id"],
        "method": row["method"],
        "params_json": row["params_json"],
        "diameter": row["diameter"],
        "created_at": row["created_at"],
        "run_id": row["run_id"],
    }


def get_cluster_member_hashes(
    conn: sqlite3.Connection,
    cluster_id: int,
) -> dict[int, str | None]:
    """Return a mapping of asset_id → phash_hex for all members of a cluster.

    Args:
        conn: Open database connection.
        cluster_id: The cluster to look up.

    Returns:
        Dict mapping each member's asset ID to its stored hex hash string,
        or to ``None`` when no hash has been computed for that asset.
    """
    rows = conn.execute(
        "SELECT cm.asset_id, p.phash_hex"
        " FROM cluster_members cm"
        " LEFT JOIN phash p ON p.asset_id = cm.asset_id"
        " WHERE cm.cluster_id = ?",
        (cluster_id,),
    ).fetchall()
    return {row["asset_id"]: row["phash_hex"] for row in rows}


def list_asset_ids_without_score(
    conn: sqlite3.Connection,
    scorer_id: str,
    variant_id: str,
    metric_key: str,
    scorer_version: str | None = None,
) -> list[int]:
    """Return IDs of assets that need scoring (no score, or score is stale).

    An asset is considered *needing a score* when it either has no score row
    for the given ``scorer_id`` + ``variant_id`` + ``metric_key``, **or** its
    existing score was produced by a different (older) ``scorer_version``.

    When *scorer_version* is ``None`` any existing score counts — backward
    compatible with callers that do not track versions.

    Args:
        conn: Open database connection.
        scorer_id: Scorer ID.
        variant_id: Variant ID.
        metric_key: Metric key.
        scorer_version: Current scorer version string.  When provided, only
            scores whose ``scorer_version`` exactly matches are considered
            up-to-date.  Assets scored by a different version are treated as
            un-scored.

    Returns:
        List of integer asset IDs ordered by ``assets.id``.
    """
    if scorer_version is not None:
        rows = conn.execute(
            "SELECT a.id FROM assets a"
            " WHERE a.id NOT IN ("
            "   SELECT s.asset_id FROM asset_scores s"
            "   WHERE s.scorer_id = ? AND s.variant_id = ? AND s.metric_key = ?"
            "     AND s.scorer_version = ?"
            " )"
            " ORDER BY a.id",
            (scorer_id, variant_id, metric_key, scorer_version),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT a.id FROM assets a"
            " WHERE a.id NOT IN ("
            "   SELECT s.asset_id FROM asset_scores s"
            "   WHERE s.scorer_id = ? AND s.variant_id = ? AND s.metric_key = ?"
            " )"
            " ORDER BY a.id",
            (scorer_id, variant_id, metric_key),
        ).fetchall()
    return [row[0] for row in rows]


# ---------------------------------------------------------------------------
# View preset helpers
# ---------------------------------------------------------------------------


@dataclass
class ViewPreset:
    """One saved view preset (filter + sort combination)."""

    id: int
    name: str
    sort_by: str | None
    favorited: int | None
    min_score: float | None
    max_score: float | None
    created_at: int
    updated_at: int


def _row_to_preset(row: sqlite3.Row) -> ViewPreset:
    """Convert a ``view_presets`` row to a :class:`ViewPreset`."""
    return ViewPreset(
        id=row["id"],
        name=row["name"],
        sort_by=row["sort_by"],
        favorited=row["favorited"],
        min_score=row["min_score"],
        max_score=row["max_score"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def list_view_presets(conn: sqlite3.Connection) -> list[ViewPreset]:
    """Return all view presets ordered by name.

    Args:
        conn: Open database connection.

    Returns:
        List of :class:`ViewPreset` objects.
    """
    rows = conn.execute(
        "SELECT id, name, sort_by, favorited, min_score, max_score, created_at, updated_at"
        " FROM view_presets ORDER BY name"
    ).fetchall()
    return [_row_to_preset(row) for row in rows]


def get_view_preset(conn: sqlite3.Connection, preset_id: int) -> ViewPreset | None:
    """Return the view preset with the given ID, or ``None`` if not found.

    Args:
        conn: Open database connection.
        preset_id: Primary key of the preset.

    Returns:
        :class:`ViewPreset` or ``None``.
    """
    row = conn.execute(
        "SELECT id, name, sort_by, favorited, min_score, max_score, created_at, updated_at"
        " FROM view_presets WHERE id = ?",
        (preset_id,),
    ).fetchone()
    return _row_to_preset(row) if row else None


def upsert_view_preset(
    conn: sqlite3.Connection,
    name: str,
    sort_by: str | None = None,
    favorited: int | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
) -> int:
    """Insert or update a view preset identified by *name*.

    If a preset with the given *name* already exists its fields are updated;
    otherwise a new row is inserted.

    Args:
        conn: Open database connection.
        name: Unique human-readable name for the preset.
        sort_by: ``"scorer_id:metric_key"`` or ``None`` (date order).
        favorited: ``1`` to filter favourites only, ``None`` for no filter.
        min_score: Inclusive lower bound on the sort-metric score, or ``None``.
        max_score: Inclusive upper bound on the sort-metric score, or ``None``.

    Returns:
        The integer primary key of the inserted or updated row.
    """
    now = int(time.time())
    row = conn.execute(
        "INSERT INTO view_presets (name, sort_by, favorited, min_score, max_score,"
        "   created_at, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(name) DO UPDATE SET"
        "   sort_by = excluded.sort_by,"
        "   favorited = excluded.favorited,"
        "   min_score = excluded.min_score,"
        "   max_score = excluded.max_score,"
        "   updated_at = excluded.updated_at"
        " RETURNING id",
        (name, sort_by, favorited, min_score, max_score, now, now),
    ).fetchone()
    conn.commit()
    return row[0]


def delete_view_preset(conn: sqlite3.Connection, preset_id: int) -> bool:
    """Delete the view preset with the given ID.

    Args:
        conn: Open database connection.
        preset_id: Primary key of the preset to delete.

    Returns:
        ``True`` if a row was deleted, ``False`` if the preset did not exist.
    """
    cursor = conn.execute("DELETE FROM view_presets WHERE id = ?", (preset_id,))
    conn.commit()
    return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# Rescan helpers
# ---------------------------------------------------------------------------


def list_asset_ids_needing_rescan(
    conn: sqlite3.Connection,
    full: bool = False,
) -> list[tuple[int, str, str | None]]:
    """Return assets that need reprocessing by the indexer.

    Args:
        conn: Open database connection.
        full: When ``True``, return *all* assets regardless of their current
            ``indexer_version``.  When ``False`` (default), return only assets
            where ``indexer_version IS NULL OR indexer_version < CURRENT_INDEXER_VERSION``.

    Returns:
        A list of ``(id, relpath, sidecar_relpath)`` tuples ordered by ``id``.
    """
    if full:
        sql = "SELECT id, relpath, sidecar_relpath FROM assets ORDER BY id"
        rows = conn.execute(sql).fetchall()
    else:
        sql = (
            "SELECT id, relpath, sidecar_relpath FROM assets "
            "WHERE indexer_version IS NULL OR indexer_version < ? "
            "ORDER BY id"
        )
        rows = conn.execute(sql, (CURRENT_INDEXER_VERSION,)).fetchall()
    return [(row[0], row[1], row[2]) for row in rows]


def count_assets_needing_rescan(conn: sqlite3.Connection) -> int:
    """Return the number of assets with a stale or missing ``indexer_version``.

    Useful for showing an "upgrade recommended" hint in the UI.

    Args:
        conn: Open database connection.

    Returns:
        Count of assets where
        ``indexer_version IS NULL OR indexer_version < CURRENT_INDEXER_VERSION``.
    """
    row = conn.execute(
        "SELECT COUNT(*) FROM assets WHERE indexer_version IS NULL OR indexer_version < ?",
        (CURRENT_INDEXER_VERSION,),
    ).fetchone()
    return row[0] if row else 0


# ---------------------------------------------------------------------------
# CLIP embeddings
# ---------------------------------------------------------------------------


def list_asset_ids_without_embedding(conn: sqlite3.Connection) -> list[int]:
    """Return IDs of assets that do not yet have a stored CLIP embedding.

    Used by the ``embed`` job to find assets that need their CLIP image
    embedding computed.

    Args:
        conn: Open database connection.

    Returns:
        List of integer asset IDs ordered by ID.
    """
    rows = conn.execute(
        "SELECT a.id FROM assets a"
        " LEFT JOIN clip_embeddings ce ON a.id = ce.asset_id"
        " WHERE ce.asset_id IS NULL"
        " ORDER BY a.id"
    ).fetchall()
    return [row[0] for row in rows]


def count_clip_embeddings(conn: sqlite3.Connection) -> int:
    """Return the number of stored CLIP embeddings.

    Args:
        conn: Open database connection.

    Returns:
        Count of rows in ``clip_embeddings``.
    """
    row = conn.execute("SELECT COUNT(*) FROM clip_embeddings").fetchone()
    return row[0] if row else 0


def upsert_clip_embedding(
    conn: sqlite3.Connection,
    asset_id: int,
    embedding_blob: bytes,
    model_id: str = "ViT-L-14/openai",
) -> None:
    """Insert or update the CLIP embedding for an asset.

    Args:
        conn: Open database connection.
        asset_id: Foreign key into ``assets``.
        embedding_blob: Raw bytes of the float32 embedding vector.
        model_id: Identifier for the CLIP backbone used.
    """
    conn.execute(
        "INSERT INTO clip_embeddings (asset_id, embedding, model_id, computed_at)"
        " VALUES (?, ?, ?, ?)"
        " ON CONFLICT(asset_id) DO UPDATE SET embedding = excluded.embedding,"
        "   model_id = excluded.model_id, computed_at = excluded.computed_at",
        (asset_id, embedding_blob, model_id, int(time.time())),
    )


def bulk_upsert_clip_embeddings(
    conn: sqlite3.Connection,
    rows: list[tuple[int, bytes]],
    model_id: str = "ViT-L-14/openai",
) -> None:
    """Insert or update CLIP embeddings for multiple assets in one transaction.

    Args:
        conn: Open database connection.
        rows: List of ``(asset_id, embedding_blob)`` tuples.
        model_id: Identifier for the CLIP backbone used.
    """
    now = int(time.time())
    conn.executemany(
        "INSERT INTO clip_embeddings (asset_id, embedding, model_id, computed_at)"
        " VALUES (?, ?, ?, ?)"
        " ON CONFLICT(asset_id) DO UPDATE SET embedding = excluded.embedding,"
        "   model_id = excluded.model_id, computed_at = excluded.computed_at",
        [(aid, blob, model_id, now) for aid, blob in rows],
    )
    conn.commit()


def load_all_clip_embeddings(conn: sqlite3.Connection) -> list[tuple[int, bytes]]:
    """Load all CLIP embeddings from the database.

    Returns a list of ``(asset_id, embedding_blob)`` tuples ordered by
    ``asset_id``.  Each ``embedding_blob`` is the raw bytes of a 768-element
    float32 vector (3072 bytes).

    Args:
        conn: Open database connection.

    Returns:
        List of ``(asset_id, embedding_blob)`` tuples.
    """
    rows = conn.execute(
        "SELECT asset_id, embedding FROM clip_embeddings ORDER BY asset_id"
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def get_clip_embedding_for_asset(conn: sqlite3.Connection, asset_id: int) -> bytes | None:
    """Return the stored CLIP embedding blob for a single asset, or ``None``.

    Args:
        conn: Open database connection.
        asset_id: The asset to look up.

    Returns:
        Raw bytes of the float32 embedding vector (3072 bytes for ViT-L/14),
        or ``None`` if no embedding has been computed for this asset yet.
    """
    row = conn.execute(
        "SELECT embedding FROM clip_embeddings WHERE asset_id = ?",
        (asset_id,),
    ).fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# CLIP user tags
# ---------------------------------------------------------------------------


def list_clip_user_tags(conn: sqlite3.Connection) -> list[str]:
    """Return all user-defined CLIP tagging terms, ordered by creation time.

    Args:
        conn: Open database connection.

    Returns:
        List of term strings.
    """
    rows = conn.execute("SELECT term FROM clip_user_tags ORDER BY created_at, id").fetchall()
    return [row[0] for row in rows]


def insert_clip_user_tag(conn: sqlite3.Connection, term: str) -> bool:
    """Insert a new user-defined CLIP tagging term.

    Args:
        conn: Open database connection.
        term: The tagging term to insert (must be non-empty).

    Returns:
        ``True`` if the term was inserted, ``False`` if it already exists.
    """
    try:
        conn.execute(
            "INSERT INTO clip_user_tags (term, created_at) VALUES (?, ?)",
            (term, int(time.time())),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def delete_clip_user_tag(conn: sqlite3.Connection, term: str) -> bool:
    """Delete a user-defined CLIP tagging term.

    Args:
        conn: Open database connection.
        term: The tagging term to delete.

    Returns:
        ``True`` if the term was deleted, ``False`` if it did not exist.
    """
    cur = conn.execute("DELETE FROM clip_user_tags WHERE term = ?", (term,))
    conn.commit()
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Face detection & clustering queries
# ---------------------------------------------------------------------------


def insert_face_detection_run(
    conn: sqlite3.Connection,
    model_id: str,
    params_json: str | None,
) -> int:
    """Insert a new face detection run and return its ID.

    Args:
        conn: Open database connection.
        model_id: InsightFace model pack name (e.g. ``"buffalo_l"``).
        params_json: JSON-serialised detection parameters.

    Returns:
        The integer primary key of the new row.
    """
    row = conn.execute(
        "INSERT INTO face_detection_runs (model_id, params_json, started_at)"
        " VALUES (?, ?, ?) RETURNING id",
        (model_id, params_json, int(time.time())),
    ).fetchone()
    conn.commit()
    return row[0]


def finish_face_detection_run(conn: sqlite3.Connection, run_id: int) -> None:
    """Mark a face detection run as finished."""
    conn.execute(
        "UPDATE face_detection_runs SET finished_at = ? WHERE id = ?",
        (int(time.time()), run_id),
    )
    conn.commit()


def bulk_insert_face_embeddings(
    conn: sqlite3.Connection,
    rows: list[tuple[int, int, int, float, float, float, float, float, bytes]],
) -> None:
    """Insert multiple face embedding rows in one transaction.

    Args:
        conn: Open database connection.
        rows: List of tuples:
            ``(asset_id, run_id, face_index, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
              det_score, embedding_blob)``.
    """
    now = int(time.time())
    conn.executemany(
        "INSERT INTO face_embeddings"
        " (asset_id, run_id, face_index, bbox_x1, bbox_y1, bbox_x2, bbox_y2,"
        "  det_score, embedding, computed_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(*r, now) for r in rows],
    )
    conn.commit()


def list_asset_ids_without_face_detection(
    conn: sqlite3.Connection,
    run_id: int | None = None,
) -> list[int]:
    """Return IDs of assets that have no face embeddings.

    When *run_id* is provided, only assets not covered by that specific
    detection run are returned.

    Args:
        conn: Open database connection.
        run_id: Optional detection run to check against.

    Returns:
        List of integer asset IDs ordered by ID.
    """
    if run_id is not None:
        rows = conn.execute(
            "SELECT a.id FROM assets a"
            " LEFT JOIN face_embeddings fe"
            "   ON a.id = fe.asset_id AND fe.run_id = ?"
            " WHERE fe.id IS NULL"
            " ORDER BY a.id",
            (run_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT a.id FROM assets a"
            " LEFT JOIN face_embeddings fe ON a.id = fe.asset_id"
            " WHERE fe.id IS NULL"
            " ORDER BY a.id",
        ).fetchall()
    return [row[0] for row in rows]


def count_face_embeddings(conn: sqlite3.Connection) -> int:
    """Return total number of face embedding rows."""
    row = conn.execute("SELECT COUNT(*) FROM face_embeddings").fetchone()
    return row[0] if row else 0


def count_faces_for_asset(conn: sqlite3.Connection, asset_id: int) -> int:
    """Return the number of detected faces for a specific asset."""
    row = conn.execute(
        "SELECT COUNT(*) FROM face_embeddings WHERE asset_id = ?",
        (asset_id,),
    ).fetchone()
    return row[0] if row else 0


def list_face_detection_runs(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return all face detection runs, most recent first."""
    rows = conn.execute(
        "SELECT r.id, r.model_id, r.params_json, r.started_at, r.finished_at,"
        "   COUNT(fe.id) AS n_faces"
        " FROM face_detection_runs r"
        " LEFT JOIN face_embeddings fe ON fe.run_id = r.id"
        " GROUP BY r.id"
        " ORDER BY r.started_at DESC",
    ).fetchall()
    return [
        {
            "run_id": row["id"],
            "model_id": row["model_id"],
            "params_json": row["params_json"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "n_faces": row["n_faces"],
        }
        for row in rows
    ]


def list_face_cluster_runs(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return all face cluster runs, most recent first."""
    rows = conn.execute(
        "SELECT r.id, r.method, r.params_json, r.detection_run_id, r.created_at,"
        "   COUNT(DISTINCT fc.id) AS n_clusters"
        " FROM face_cluster_runs r"
        " LEFT JOIN face_clusters fc ON fc.run_id = r.id"
        " GROUP BY r.id"
        " ORDER BY r.created_at DESC",
    ).fetchall()
    return [
        {
            "run_id": row["id"],
            "method": row["method"],
            "params_json": row["params_json"],
            "detection_run_id": row["detection_run_id"],
            "created_at": row["created_at"],
            "n_clusters": row["n_clusters"],
        }
        for row in rows
    ]


def list_face_clusters_for_run(
    conn: sqlite3.Connection,
    run_id: int,
) -> list[dict[str, Any]]:
    """Return all face clusters in a clustering run, ordered by size desc.

    Each dict includes ``cluster_id``, ``label``, ``n_faces``,
    ``n_assets`` (distinct assets), and ``rep_asset_id`` (asset of the
    representative face).
    """
    rows = conn.execute(
        "SELECT fc.id AS cluster_id, fc.label,"
        "   COUNT(fcm.face_id) AS n_faces,"
        "   COUNT(DISTINCT fe.asset_id) AS n_assets,"
        "   fe_rep.asset_id AS rep_asset_id"
        " FROM face_clusters fc"
        " JOIN face_cluster_members fcm ON fcm.cluster_id = fc.id"
        " JOIN face_embeddings fe ON fe.id = fcm.face_id"
        " LEFT JOIN face_cluster_members fcm_rep"
        "   ON fcm_rep.cluster_id = fc.id AND fcm_rep.is_representative = 1"
        " LEFT JOIN face_embeddings fe_rep ON fe_rep.id = fcm_rep.face_id"
        " WHERE fc.run_id = ?"
        " GROUP BY fc.id"
        " ORDER BY n_faces DESC, fc.id",
        (run_id,),
    ).fetchall()
    return [
        {
            "cluster_id": row["cluster_id"],
            "label": row["label"],
            "n_faces": row["n_faces"],
            "n_assets": row["n_assets"],
            "rep_asset_id": row["rep_asset_id"],
        }
        for row in rows
    ]


def get_face_cluster_assets(
    conn: sqlite3.Connection,
    cluster_id: int,
) -> list[dict[str, Any]]:
    """Return all assets in a face cluster with face details.

    Returns a list of dicts with ``asset_id``, ``face_id``, ``distance``,
    ``is_representative``, ``det_score``, ``bbox``, and ``filename``.
    """
    rows = conn.execute(
        "SELECT fe.asset_id, fcm.face_id, fcm.distance, fcm.is_representative,"
        "   fe.det_score, fe.bbox_x1, fe.bbox_y1, fe.bbox_x2, fe.bbox_y2,"
        "   a.filename"
        " FROM face_cluster_members fcm"
        " JOIN face_embeddings fe ON fe.id = fcm.face_id"
        " JOIN assets a ON a.id = fe.asset_id"
        " WHERE fcm.cluster_id = ?"
        " ORDER BY fcm.is_representative DESC, fcm.distance ASC",
        (cluster_id,),
    ).fetchall()
    return [
        {
            "asset_id": row["asset_id"],
            "face_id": row["face_id"],
            "distance": row["distance"],
            "is_representative": row["is_representative"],
            "det_score": row["det_score"],
            "bbox": (row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]),
            "filename": row["filename"],
        }
        for row in rows
    ]


def rename_face_cluster(
    conn: sqlite3.Connection,
    cluster_id: int,
    label: str,
) -> bool:
    """Set the user-friendly label for a face cluster.

    Args:
        conn: Open database connection.
        cluster_id: The face cluster to rename.
        label: New label (person name).

    Returns:
        ``True`` if the cluster existed and was updated.
    """
    cur = conn.execute(
        "UPDATE face_clusters SET label = ? WHERE id = ?",
        (label, cluster_id),
    )
    conn.commit()
    return cur.rowcount > 0


def delete_face_cluster_run(conn: sqlite3.Connection, run_id: int) -> bool:
    """Delete a face clustering run and all its clusters and members.

    Returns ``True`` if the run existed and was deleted.
    """
    cluster_ids = [
        row[0]
        for row in conn.execute(
            "SELECT id FROM face_clusters WHERE run_id = ?", (run_id,)
        ).fetchall()
    ]
    if cluster_ids:
        conn.executemany(
            "DELETE FROM face_cluster_members WHERE cluster_id = ?",
            [(cid,) for cid in cluster_ids],
        )
        conn.execute("DELETE FROM face_clusters WHERE run_id = ?", (run_id,))
    result = conn.execute("DELETE FROM face_cluster_runs WHERE id = ?", (run_id,))
    conn.commit()
    return result.rowcount > 0


def get_face_cluster_label(conn: sqlite3.Connection, cluster_id: int) -> str | None:
    """Return the label for a face cluster, or ``None``."""
    row = conn.execute("SELECT label FROM face_clusters WHERE id = ?", (cluster_id,)).fetchone()
    return row["label"] if row else None
