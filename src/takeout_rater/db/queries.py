"""Query helpers for the takeout-rater SQLite database.

All functions accept a :class:`sqlite3.Connection` and return plain Python
types (``dict``, ``list``, ``int``, ``None``).

Rows fetched from the DB are :class:`sqlite3.Row` objects, which support
both index and column-name access.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class AssetRow:
    """Lightweight view of one row from the ``assets`` table."""

    id: int
    relpath: str
    filename: str
    ext: str
    size_bytes: int | None
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

    Uses ``INSERT ... ON CONFLICT(relpath) DO UPDATE`` so re-indexing is
    idempotent and the row's primary key is preserved across updates.

    Args:
        conn: Open database connection.
        asset: Dict with column names as keys.  ``indexed_at`` defaults to
            the current Unix timestamp if not provided.

    Returns:
        The row ID of the inserted or updated row.
    """
    asset = dict(asset)
    if "indexed_at" not in asset:
        asset["indexed_at"] = int(time.time())

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
    """Fetch one asset by its unique relative path.

    Args:
        conn: Open database connection.
        relpath: Path relative to the takeout root.

    Returns:
        An :class:`AssetRow`, or ``None`` if not found.
    """
    row = conn.execute("SELECT * FROM assets WHERE relpath = ?", (relpath,)).fetchone()
    return _row_to_asset(row) if row else None


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
# Scorer-run helpers
# ---------------------------------------------------------------------------


def insert_scorer_run(
    conn: sqlite3.Connection,
    scorer_id: str,
    variant_id: str,
    *,
    scorer_version: str | None = None,
    params_json: str | None = None,
    params_hash: str | None = None,
) -> int:
    """Insert a new scorer-run record and return its ID.

    The ``started_at`` timestamp is set to the current Unix time.  Call
    :func:`finish_scorer_run` once scoring is complete.

    Args:
        conn: Open database connection.
        scorer_id: Stable scorer identifier (e.g. ``"blur"``).
        variant_id: Variant identifier (e.g. ``"default"``).
        scorer_version: Optional version string stored for auditability.
        params_json: Optional JSON-serialised scorer parameters.
        params_hash: Optional hash of ``params_json`` for deduplication.

    Returns:
        The integer primary key of the new ``scorer_runs`` row.
    """
    row = conn.execute(
        "INSERT INTO scorer_runs (scorer_id, variant_id, scorer_version, params_json, params_hash, started_at)"
        " VALUES (?, ?, ?, ?, ?, ?)"
        " RETURNING id",
        (scorer_id, variant_id, scorer_version, params_json, params_hash, int(time.time())),
    ).fetchone()
    conn.commit()
    return row[0]


def finish_scorer_run(conn: sqlite3.Connection, run_id: int) -> None:
    """Mark a scorer run as finished by setting its ``finished_at`` timestamp.

    Args:
        conn: Open database connection.
        run_id: The ``id`` of the ``scorer_runs`` row to update.
    """
    conn.execute(
        "UPDATE scorer_runs SET finished_at = ? WHERE id = ?",
        (int(time.time()), run_id),
    )
    conn.commit()


def bulk_insert_asset_scores(
    conn: sqlite3.Connection,
    run_id: int,
    scores: list[tuple[int, str, float]],
) -> None:
    """Insert multiple ``asset_scores`` rows in a single transaction.

    Rows with duplicate ``(asset_id, scorer_run_id, metric_key)`` are ignored
    (``INSERT OR IGNORE``).

    Args:
        conn: Open database connection.
        run_id: Foreign key into ``scorer_runs``.
        scores: Iterable of ``(asset_id, metric_key, value)`` triples.
    """
    conn.executemany(
        "INSERT OR IGNORE INTO asset_scores (asset_id, scorer_run_id, metric_key, value)"
        " VALUES (?, ?, ?, ?)",
        [(asset_id, run_id, metric_key, value) for asset_id, metric_key, value in scores],
    )
    conn.commit()


def get_asset_scores(
    conn: sqlite3.Connection,
    asset_id: int,
) -> list[dict[str, Any]]:
    """Return all scores for a single asset, across all scorer runs.

    Args:
        conn: Open database connection.
        asset_id: The asset to look up.

    Returns:
        List of dicts with keys ``scorer_id``, ``variant_id``, ``metric_key``,
        ``value``, and ``finished_at``.  Ordered by ``finished_at DESC``.
    """
    rows = conn.execute(
        "SELECT r.scorer_id, r.variant_id, s.metric_key, s.value, r.finished_at"
        " FROM asset_scores s"
        " JOIN scorer_runs r ON r.id = s.scorer_run_id"
        " WHERE s.asset_id = ? AND r.finished_at IS NOT NULL"
        " ORDER BY r.finished_at DESC",
        (asset_id,),
    ).fetchall()
    return [
        {
            "scorer_id": row["scorer_id"],
            "variant_id": row["variant_id"],
            "metric_key": row["metric_key"],
            "value": row["value"],
            "finished_at": row["finished_at"],
        }
        for row in rows
    ]


def get_latest_scorer_run_id(
    conn: sqlite3.Connection,
    scorer_id: str,
    variant_id: str,
) -> int | None:
    """Return the ID of the most recently *finished* run for a scorer+variant.

    Args:
        conn: Open database connection.
        scorer_id: Scorer identifier.
        variant_id: Variant identifier.

    Returns:
        Integer run ID, or ``None`` if no finished run exists.
    """
    row = conn.execute(
        "SELECT id FROM scorer_runs"
        " WHERE scorer_id = ? AND variant_id = ? AND finished_at IS NOT NULL"
        " ORDER BY finished_at DESC, id DESC LIMIT 1",
        (scorer_id, variant_id),
    ).fetchone()
    return row[0] if row else None


def list_assets_by_score(
    conn: sqlite3.Connection,
    scorer_id: str,
    metric_key: str,
    variant_id: str = "default",
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

    Only assets with a score from the *latest finished* run for the given
    ``scorer_id`` + ``variant_id`` are returned.  Assets without a score are
    excluded (per the design: "only scored" view when a primary sort metric is
    active).

    Args:
        conn: Open database connection.
        scorer_id: Scorer whose scores to sort by.
        metric_key: Metric key to sort by.
        variant_id: Variant ID (default ``"default"``).
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

    run_id = get_latest_scorer_run_id(conn, scorer_id, variant_id)
    if run_id is None:
        return []

    order = "DESC" if descending else "ASC"
    conditions: list[str] = ["s.scorer_run_id = ?", "s.metric_key = ?"]
    params: list[Any] = [run_id, metric_key]

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
    variant_id: str = "default",
    *,
    favorited: bool | None = None,
    trashed: bool | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
) -> int:
    """Count assets that have a score from the latest run for scorer+variant.

    Args:
        conn: Open database connection.
        scorer_id: Scorer ID.
        metric_key: Metric key.
        variant_id: Variant ID.
        favorited: Optional favorited filter.
        trashed: Optional trashed filter.
        min_score: Optional inclusive lower bound on the score value.
        max_score: Optional inclusive upper bound on the score value.

    Returns:
        Integer count.
    """
    run_id = get_latest_scorer_run_id(conn, scorer_id, variant_id)
    if run_id is None:
        return 0

    conditions: list[str] = ["s.scorer_run_id = ?", "s.metric_key = ?"]
    params: list[Any] = [run_id, metric_key]

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
# pHash helpers
# ---------------------------------------------------------------------------


def upsert_phash(
    conn: sqlite3.Connection,
    asset_id: int,
    phash_hex: str,
    algo: str = "dhash",
) -> None:
    """Insert or update the perceptual hash for an asset.

    Args:
        conn: Open database connection.
        asset_id: Foreign key into ``assets``.
        phash_hex: Hex-encoded hash string.
        algo: Hash algorithm name (default ``"dhash"``).
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


def list_asset_ids_without_phash(conn: sqlite3.Connection) -> list[int]:
    """Return IDs of assets that do not yet have a stored pHash.

    Args:
        conn: Open database connection.

    Returns:
        List of integer asset IDs.
    """
    rows = conn.execute(
        "SELECT a.id FROM assets a"
        " LEFT JOIN phash p ON p.asset_id = a.id"
        " WHERE p.asset_id IS NULL"
        " ORDER BY a.id",
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


def list_all_phashes(conn: sqlite3.Connection) -> list[tuple[int, str]]:
    """Return all stored perceptual hashes as ``(asset_id, phash_hex)`` pairs.

    Args:
        conn: Open database connection.

    Returns:
        List of ``(asset_id, phash_hex)`` tuples ordered by ``asset_id``.
    """
    rows = conn.execute("SELECT asset_id, phash_hex FROM phash ORDER BY asset_id").fetchall()
    return [(row[0], row[1]) for row in rows]


def insert_cluster(
    conn: sqlite3.Connection,
    method: str,
    params_json: str | None,
) -> int:
    """Insert a new cluster row and return its ID.

    Args:
        conn: Open database connection.
        method: Algorithm identifier (e.g. ``"dhash_hamming"``).
        params_json: JSON-serialised clustering parameters.

    Returns:
        The integer primary key of the new ``clusters`` row.
    """
    row = conn.execute(
        "INSERT INTO clusters (method, params_json, created_at) VALUES (?, ?, ?) RETURNING id",
        (method, params_json, int(time.time())),
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
        ``rep_asset_id``, ``rep_filename``, ``created_at``.
    """
    rows = conn.execute(
        "SELECT c.id AS cluster_id, c.method, c.created_at,"
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


def list_asset_ids_without_score(
    conn: sqlite3.Connection,
    scorer_id: str,
    variant_id: str,
    metric_key: str,
) -> list[int]:
    """Return IDs of assets that have no score for the given scorer run.

    More precisely: assets that have never appeared in *any* finished run for
    the given ``scorer_id`` + ``variant_id`` + ``metric_key``.

    Args:
        conn: Open database connection.
        scorer_id: Scorer ID.
        variant_id: Variant ID.
        metric_key: Metric key.

    Returns:
        List of integer asset IDs ordered by ``assets.id``.
    """
    rows = conn.execute(
        "SELECT a.id FROM assets a"
        " WHERE a.id NOT IN ("
        "   SELECT s.asset_id FROM asset_scores s"
        "   JOIN scorer_runs r ON r.id = s.scorer_run_id"
        "   WHERE r.scorer_id = ? AND r.variant_id = ? AND s.metric_key = ?"
        "     AND r.finished_at IS NOT NULL"
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
