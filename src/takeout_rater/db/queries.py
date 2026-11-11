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

import re
import sqlite3
import time
import unicodedata
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


@dataclass(frozen=True)
class PlaceFilter:
    """Restrict asset queries to a reverse-geocoded place (see ``asset_places``).

    Each field is optional; set fields are combined with AND.  ``country_code``
    is an ISO alpha-2 code, ``admin1`` and ``city`` are the stored names.
    """

    country_code: str | None = None
    admin1: str | None = None
    city: str | None = None

    def is_empty(self) -> bool:
        return not (self.country_code or self.admin1 or self.city)

    def sql_condition(self, id_column: str) -> tuple[str, list[Any]]:
        """Return ``("<id_column> IN (SELECT …)", params)`` for a WHERE clause.

        *id_column* is a trusted, code-controlled column reference such as
        ``"id"`` or ``"a.id"``.
        """
        conds: list[str] = []
        params: list[Any] = []
        for col, value in (
            ("country_code", self.country_code),
            ("admin1", self.admin1),
            ("city", self.city),
        ):
            if value:
                conds.append(f"{col} = ?")
                params.append(value)
        if not conds:
            return "1 = 1", []
        where = " AND ".join(conds)
        return f"{id_column} IN (SELECT asset_id FROM asset_places WHERE {where})", params  # noqa: S608


def list_assets(
    conn: sqlite3.Connection,
    *,
    limit: int = 100,
    offset: int = 0,
    order_by: str = "taken_at DESC",
    favorited: bool | None = None,
    place: PlaceFilter | None = None,
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
        place: Optional place-name filter (see :class:`PlaceFilter`).
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
    if place is not None:
        place_sql, place_params = place.sql_condition("id")
        conditions.append(place_sql)
        params.extend(place_params)
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
    place: PlaceFilter | None = None,
    archived: bool | None = None,
    trashed: bool | None = None,
    ext: str | None = None,
) -> int:
    """Return the total number of assets matching the given filters.

    Args:
        conn: Open database connection.
        favorited: Optional favorited filter.
        place: Optional place-name filter (see :class:`PlaceFilter`).
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
    if place is not None:
        place_sql, place_params = place.sql_condition("id")
        conditions.append(place_sql)
        params.extend(place_params)
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


def asset_relpath_to_path(asset: AssetRow, photos_root: Path) -> Path:
    """Return the absolute path to an asset given its database row and photos root."""
    return photos_root / asset.relpath


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


def _deduped_assets_sql(where: str) -> str:
    """Return a subquery selecting one row per SHA-256 group from ``assets``.

    *where* (a full ``WHERE …`` clause or ``""``) is applied **before** ranking,
    so each group's representative is its lowest-``id`` row that passes the
    filters.  Assets without a SHA-256 are each their own group.
    """
    return (
        f"(SELECT * FROM ("
        f"  SELECT *, ROW_NUMBER() OVER ("
        f"    PARTITION BY COALESCE(sha256, CAST(id AS TEXT)) ORDER BY id"
        f"  ) AS _rn FROM assets {where}"
        f") WHERE _rn = 1)"
    )


def list_assets_deduped(
    conn: sqlite3.Connection,
    *,
    limit: int = 100,
    offset: int = 0,
    order_by: str = "taken_at DESC",
    favorited: bool | None = None,
    place: PlaceFilter | None = None,
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
        place: Optional place-name filter (see :class:`PlaceFilter`).
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
    if place is not None:
        place_sql, place_params = place.sql_condition("id")
        conditions.append(place_sql)
        params.extend(place_params)
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
        f"SELECT * FROM {_deduped_assets_sql(where)} ORDER BY {order_by} LIMIT ? OFFSET ?"
    )
    params.extend([limit, offset])
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_asset(r) for r in rows]


def count_assets_deduped(
    conn: sqlite3.Connection,
    *,
    favorited: bool | None = None,
    place: PlaceFilter | None = None,
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
        place: Optional place-name filter (see :class:`PlaceFilter`).
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
    if place is not None:
        place_sql, place_params = place.sql_condition("id")
        conditions.append(place_sql)
        params.extend(place_params)
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


def load_dated_asset_ids(conn: sqlite3.Connection) -> tuple[Any, Any]:
    """Return ``(asset_ids, taken_at)`` for every dated, non-trashed photo.

    Only one asset per SHA-256 group is kept (as in :func:`list_assets_deduped`),
    so exact duplicates are not counted twice by time-based aggregations such
    as the mood timeline.

    Returns:
        Two ``int64`` numpy arrays of equal length, ordered by asset id.
    """
    import numpy as np

    source = _deduped_assets_sql("WHERE COALESCE(trashed, 0) = 0 AND taken_at IS NOT NULL")
    rows = conn.execute(f"SELECT id, taken_at FROM {source} ORDER BY id").fetchall()  # noqa: S608
    if not rows:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    arr = np.asarray(rows, dtype=np.int64)
    return arr[:, 0].copy(), arr[:, 1].copy()


def dated_assets_fingerprint(conn: sqlite3.Connection) -> tuple:
    """Return a cheap fingerprint of what :func:`load_dated_asset_ids` depends on.

    It changes when photos are added or removed, re-dated, (un)trashed or
    re-hashed, so callers can tell when a cached result is stale without
    reloading every row.
    """
    row = conn.execute(
        "SELECT COUNT(*), TOTAL(taken_at), TOTAL(COALESCE(trashed, 0) * id), COUNT(sha256)"
        " FROM assets"
    ).fetchone()
    return tuple(row)


def count_assets_newer_than(
    conn: sqlite3.Connection,
    timestamp: int,
    *,
    favorited: bool | None = None,
    place: PlaceFilter | None = None,
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
        place: Optional place-name filter (see :class:`PlaceFilter`).
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
    if place is not None:
        place_sql, place_params = place.sql_condition("id")
        conditions.append(place_sql)
        params.extend(place_params)

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


# ---------------------------------------------------------------------------
# Geolocation (map view)
# ---------------------------------------------------------------------------


_HAS_GEO = "(geo_lat IS NOT NULL AND geo_lon IS NOT NULL AND (geo_lat != 0 OR geo_lon != 0))"
_HAS_EXIF_GEO = (
    "(geo_exif_lat IS NOT NULL AND geo_exif_lon IS NOT NULL"
    " AND (geo_exif_lat != 0 OR geo_exif_lon != 0))"
)


def _map_filter_sql(favorited: bool | None, *, with_location: bool) -> tuple[str, list[Any]]:
    """Return the ``WHERE`` clause and params shared by the map queries."""
    conditions: list[str] = ["COALESCE(trashed, 0) = 0"]
    params: list[Any] = []
    if favorited is not None:
        conditions.append("COALESCE(favorited, 0) = ?")
        params.append(1 if favorited else 0)
    if with_location:
        conditions.append(f"({_HAS_GEO} OR {_HAS_EXIF_GEO})")
    return "WHERE " + " AND ".join(conditions), params


def count_map_candidates(
    conn: sqlite3.Connection,
    *,
    favorited: bool | None = None,
    deduped: bool = True,
) -> int:
    """Count the assets the map could show under these filters, with or without a location.

    Used for the "N of M photos have a location" figure, so M reflects the same
    favorite / duplicate filters as :func:`list_geotagged_points`.
    """
    where, params = _map_filter_sql(favorited, with_location=False)
    source = _deduped_assets_sql(where) if deduped else f"(SELECT * FROM assets {where})"
    return conn.execute(f"SELECT COUNT(*) FROM {source}", params).fetchone()[0]  # noqa: S608


def _valid_coords(lat: float | None, lon: float | None) -> bool:
    return lat is not None and lon is not None and (lat != 0 or lon != 0)


def asset_location(asset: Any) -> tuple[float, float] | None:
    """Return the ``(lat, lon)`` the map uses for *asset*, or ``None``.

    Same rule as :func:`list_geotagged_points`: ``geoData`` first, then
    ``geoDataExif``; ``0.0/0.0`` means "no location".
    """
    if _valid_coords(asset.geo_lat, asset.geo_lon):
        return asset.geo_lat, asset.geo_lon
    if _valid_coords(asset.geo_exif_lat, asset.geo_exif_lon):
        return asset.geo_exif_lat, asset.geo_exif_lon
    return None


def list_geotagged_points(
    conn: sqlite3.Connection,
    *,
    favorited: bool | None = None,
    deduped: bool = True,
) -> list[tuple[int, float, float, int | None, int]]:
    """Return a compact ``(id, lat, lon, taken_at, favorited)`` tuple per geotagged asset.

    The location comes from the sidecar ``geoData`` block and falls back to
    ``geoDataExif`` when ``geoData`` is missing or ``0.0/0.0`` (Google Takeout's
    encoding for "no location").  Assets without any usable location, and
    trashed assets, are omitted.  Rows are ordered by ``taken_at`` (NULLs first)
    so clients can binary-search by time.

    Args:
        conn: Open database connection.
        favorited: Optional favorited filter.
        deduped: If ``True`` (default), keep only the lowest-``id`` asset per
            SHA-256 group, matching :func:`list_assets_deduped`.

    Returns:
        List of ``(id, lat, lon, taken_at, favorited)`` tuples, where
        ``favorited`` is ``0`` or ``1``.
    """
    where, params = _map_filter_sql(favorited, with_location=True)
    # Filters run before picking a duplicate group's representative, so a group
    # keeps an eligible copy even if its lowest-id copy is filtered out.
    source = _deduped_assets_sql(where) if deduped else f"(SELECT * FROM assets {where})"
    sql = (  # noqa: S608
        f"SELECT id,"
        f" CASE WHEN {_HAS_GEO} THEN geo_lat ELSE geo_exif_lat END AS lat,"
        f" CASE WHEN {_HAS_GEO} THEN geo_lon ELSE geo_exif_lon END AS lon,"
        f" taken_at, COALESCE(favorited, 0) AS fav"
        f" FROM {source}"
        f" ORDER BY taken_at, id"
    )
    return [(r[0], r[1], r[2], r[3], r[4]) for r in conn.execute(sql, params)]


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
    place: PlaceFilter | None = None,
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
        place: Optional place-name filter (see :class:`PlaceFilter`).
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
    if place is not None:
        place_sql, place_params = place.sql_condition("a.id")
        conditions.append(place_sql)
        params.extend(place_params)
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
    place: PlaceFilter | None = None,
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
        place: Optional place-name filter (see :class:`PlaceFilter`).
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
    if place is not None:
        place_sql, place_params = place.sql_condition("a.id")
        conditions.append(place_sql)
        params.extend(place_params)
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
    descending: bool = True


def list_assets_multi_sort(
    conn: sqlite3.Connection,
    criteria: list[SortCriterion],
    *,
    limit: int = 100,
    offset: int = 0,
    favorited: bool | None = None,
    place: PlaceFilter | None = None,
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
        place: Optional place-name filter (see :class:`PlaceFilter`).
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
    order0 = "DESC" if c0.descending else "ASC"
    order_cols = [f"s1.value {order0}"]

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
        order_i = "DESC" if c.descending else "ASC"
        order_cols.append(f"{alias}.value {order_i} NULLS LAST")

    conditions: list[str] = []
    where_params: list[Any] = []

    if favorited is not None:
        conditions.append("a.favorited = ?")
        where_params.append(1 if favorited else 0)
    if place is not None:
        place_sql, place_params = place.sql_condition("a.id")
        conditions.append(place_sql)
        where_params.extend(place_params)
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
    place: PlaceFilter | None = None,
    trashed: bool | None = None,
) -> int:
    """Count assets that satisfy a multi-criteria sort/filter specification.

    Mirrors the filtering logic of :func:`list_assets_multi_sort`.

    Args:
        conn: Open database connection.
        criteria: Ordered list of sort levels.
        favorited: Optional favorited filter.
        place: Optional place-name filter (see :class:`PlaceFilter`).
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
    if place is not None:
        place_sql, place_params = place.sql_condition("a.id")
        conditions.append(place_sql)
        where_params.extend(place_params)
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
    n_skipped: int = 0,
) -> int:
    """Insert a new clustering_run row and return its ID.

    Args:
        conn: Open database connection.
        method: Algorithm identifier (e.g. ``"dhash_hamming"``).
        params_json: JSON-serialised clustering parameters, or ``None``.
        n_skipped: Number of components/clusters that were skipped (e.g. due
            to :class:`MemoryError` or a ``max_cluster_size`` filter).

    Returns:
        The integer primary key of the new ``clustering_runs`` row.
    """
    row = conn.execute(
        "INSERT INTO clustering_runs (method, params_json, created_at, n_skipped)"
        " VALUES (?, ?, ?, ?)"
        " RETURNING id",
        (method, params_json, int(time.time()), n_skipped),
    ).fetchone()
    conn.commit()
    return row[0]


def update_clustering_run_n_skipped(
    conn: sqlite3.Connection,
    run_id: int,
    n_skipped: int,
) -> None:
    """Update the *n_skipped* field on an existing clustering run.

    Args:
        conn: Open database connection.
        run_id: The clustering run to update.
        n_skipped: New total number of skipped components/clusters.
    """
    conn.execute(
        "UPDATE clustering_runs SET n_skipped = ? WHERE id = ?",
        (n_skipped, run_id),
    )
    conn.commit()


def list_clustering_runs(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return all clustering runs ordered by most-recent first.

    Each dict contains ``run_id``, ``method``, ``params_json``, ``created_at``,
    ``n_clusters``, ``n_skipped``, and ``rep_asset_id`` (representative thumbnail
    from the first cluster of the run, or ``None``).
    """
    rows = conn.execute(
        "SELECT r.id AS run_id, r.method, r.params_json, r.created_at,"
        "   r.n_skipped,"
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
            "n_skipped": row["n_skipped"],
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


def load_clip_embeddings_with_relpaths(
    conn: sqlite3.Connection,
) -> list[tuple[int, bytes, str]]:
    """Load all CLIP embeddings joined with asset relative paths.

    Returns a list of ``(asset_id, embedding_blob, relpath)`` tuples ordered
    by ``asset_id``.  Only assets with a stored embedding are included.

    Args:
        conn: Open database connection.

    Returns:
        List of ``(asset_id, embedding_blob, relpath)`` tuples.
    """
    rows = conn.execute(
        "SELECT ce.asset_id, ce.embedding, a.relpath"
        " FROM clip_embeddings ce"
        " JOIN assets a ON a.id = ce.asset_id"
        " ORDER BY ce.asset_id"
    ).fetchall()
    return [(row[0], row[1], row[2]) for row in rows]


# ---------------------------------------------------------------------------
# Model-keyed image embeddings (search backbones other than CLIP L/14)
# ---------------------------------------------------------------------------

#: Key under which the legacy ``clip_embeddings`` table (CLIP ViT-L/14,
#: OpenAI weights) is exposed by the model-keyed helpers below.
CLIP_L14_MODEL_KEY = "clip-vit-l-14"


def blobs_to_matrix(blobs: list[bytes], dim: int | None = None) -> Any:
    """Convert float32 embedding blobs into one ``(N, dim)`` numpy matrix.

    Preallocates the result and fills it row by row with
    :func:`numpy.frombuffer`, which is far faster than unpacking each row with
    :mod:`struct` and avoids an intermediate joined copy.  Raises
    :class:`ValueError` when a blob's length differs from the first row (or
    from ``dim``), since mixing dimensions would corrupt the matrix.

    Returns a writable, C-contiguous ``float32`` array.
    """
    import numpy as np

    if not blobs:
        return np.empty((0, dim or 0), dtype=np.float32)
    row_bytes = len(blobs[0]) if dim is None else dim * 4
    if row_bytes % 4:
        raise ValueError("Embedding blob length is not a multiple of 4 bytes")
    out = np.empty((len(blobs), row_bytes // 4), dtype=np.float32)
    for i, blob in enumerate(blobs):
        if len(blob) != row_bytes:
            raise ValueError("Embedding blobs have inconsistent lengths")
        out[i] = np.frombuffer(blob, dtype="<f4")
    return out


def load_embedding_matrix(conn: sqlite3.Connection, model_key: str) -> tuple[Any, Any]:
    """Load every stored embedding for *model_key* as numpy arrays.

    ``model_key`` :data:`CLIP_L14_MODEL_KEY` reads the ``clip_embeddings``
    table; any other key reads ``image_embeddings``.  Rows are streamed from
    the cursor into a preallocated matrix, so peak memory stays close to the
    size of the result.  Rows whose length differs from the first row's are
    skipped and logged, rather than failing the whole load.

    Returns:
        ``(asset_ids, matrix)``: an ``int64`` array of shape ``(N,)`` and a
        ``float32`` array of shape ``(N, dim)``, both ordered by asset id.
        Both are empty (``N == 0``) when nothing is stored.
    """
    import logging

    import numpy as np

    if model_key == CLIP_L14_MODEL_KEY:
        count_sql = "SELECT COUNT(*) FROM clip_embeddings"
        rows_sql = "SELECT asset_id, embedding FROM clip_embeddings ORDER BY asset_id"
        params: tuple = ()
    else:
        count_sql = "SELECT COUNT(*) FROM image_embeddings WHERE model_key = ?"
        rows_sql = (
            "SELECT asset_id, embedding FROM image_embeddings WHERE model_key = ? ORDER BY asset_id"
        )
        params = (model_key,)

    capacity = conn.execute(count_sql, params).fetchone()[0]
    asset_ids = np.empty(capacity, dtype=np.int64)
    matrix: Any = None
    n = 0
    skipped = 0
    for aid, blob in conn.execute(rows_sql, params):
        if n >= capacity:  # rows inserted between COUNT and SELECT
            break
        if matrix is None:
            if len(blob) == 0 or len(blob) % 4:
                skipped += 1
                continue
            matrix = np.empty((capacity, len(blob) // 4), dtype=np.float32)
        if len(blob) != matrix.shape[1] * 4:
            skipped += 1
            continue
        asset_ids[n] = aid
        matrix[n] = np.frombuffer(blob, dtype="<f4")
        n += 1
    if skipped:
        logging.getLogger(__name__).warning(
            "Skipped %d %s embedding(s) with an unexpected length", skipped, model_key
        )
    if matrix is None:
        return np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.float32)
    if n < capacity:
        return asset_ids[:n].copy(), matrix[:n].copy()
    return asset_ids, matrix


def count_image_embeddings(conn: sqlite3.Connection, model_key: str) -> int:
    """Return how many assets have an embedding stored for *model_key*."""
    if model_key == CLIP_L14_MODEL_KEY:
        return count_clip_embeddings(conn)
    row = conn.execute(
        "SELECT COUNT(*) FROM image_embeddings WHERE model_key = ?", (model_key,)
    ).fetchone()
    return row[0] if row else 0


def bulk_upsert_image_embeddings(
    conn: sqlite3.Connection,
    model_key: str,
    rows: list[tuple[int, bytes]],
) -> None:
    """Insert or replace embeddings for *model_key* in one transaction.

    Args:
        conn: Open database connection.
        model_key: Search backbone key.  Must not be :data:`CLIP_L14_MODEL_KEY`
            (use :func:`bulk_upsert_clip_embeddings` for that).
        rows: ``(asset_id, float32_blob)`` tuples; every blob must have the
            same length.
    """
    if model_key == CLIP_L14_MODEL_KEY:
        raise ValueError("CLIP L/14 embeddings belong in clip_embeddings")
    if not rows:
        return
    lengths = {len(blob) for _aid, blob in rows}
    if len(lengths) != 1 or next(iter(lengths)) % 4 or not next(iter(lengths)):
        raise ValueError("Embedding blobs must all have the same non-zero float32 length")
    existing = conn.execute(
        "SELECT dim FROM image_embeddings WHERE model_key = ? LIMIT 1", (model_key,)
    ).fetchone()
    if existing is not None and existing[0] * 4 not in lengths:
        raise ValueError(
            f"{model_key} embeddings are {existing[0]}-d; refusing to mix in other dimensions"
        )
    now = int(time.time())
    conn.executemany(
        "INSERT INTO image_embeddings (asset_id, model_key, dim, embedding, computed_at)"
        " VALUES (?, ?, ?, ?, ?)"
        " ON CONFLICT(model_key, asset_id) DO UPDATE SET embedding = excluded.embedding,"
        "   dim = excluded.dim, computed_at = excluded.computed_at",
        [(aid, model_key, len(blob) // 4, blob, now) for aid, blob in rows],
    )
    conn.commit()


def list_asset_ids_without_image_embedding(conn: sqlite3.Connection, model_key: str) -> list[int]:
    """Return IDs of assets with no stored embedding for *model_key*, by ID."""
    if model_key == CLIP_L14_MODEL_KEY:
        return list_asset_ids_without_embedding(conn)
    rows = conn.execute(
        "SELECT a.id FROM assets a"
        " WHERE NOT EXISTS ("
        "   SELECT 1 FROM image_embeddings ie"
        "   WHERE ie.model_key = ? AND ie.asset_id = a.id"
        " ) ORDER BY a.id",
        (model_key,),
    ).fetchall()
    return [r[0] for r in rows]


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


def clear_face_detection_data(conn: sqlite3.Connection) -> None:
    """Delete all face detection and face clustering data.

    A fresh face detection run should replace previous embeddings rather than
    accumulating rows from old or aborted runs.  Face clustering rows reference
    face embeddings and detection runs, so delete them first.
    """
    conn.execute("DELETE FROM face_cluster_members")
    conn.execute("DELETE FROM face_clusters")
    conn.execute("DELETE FROM face_cluster_runs")
    conn.execute("DELETE FROM face_embeddings")
    conn.execute("DELETE FROM face_detection_runs")
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


def count_face_embeddings(conn: sqlite3.Connection, run_id: int | None = None) -> int:
    """Return total number of face embedding rows."""
    if run_id is None:
        row = conn.execute("SELECT COUNT(*) FROM face_embeddings").fetchone()
    else:
        row = conn.execute(
            "SELECT COUNT(*) FROM face_embeddings WHERE run_id = ?",
            (run_id,),
        ).fetchone()
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


def get_latest_face_detection_run_id(conn: sqlite3.Connection) -> int | None:
    """Return the most recent finished face detection run ID, or latest run if none finished."""
    row = conn.execute(
        "SELECT id FROM face_detection_runs"
        " ORDER BY (finished_at IS NOT NULL) DESC,"
        "          COALESCE(finished_at, started_at) DESC,"
        "          id DESC"
        " LIMIT 1"
    ).fetchone()
    return row[0] if row else None


def delete_face_detection_run(conn: sqlite3.Connection, run_id: int) -> bool:
    """Delete a face detection run, its embeddings, and dependent cluster runs.

    Face cluster runs may reference the detection run directly or indirectly
    through face embeddings.  Delete those cluster runs first so the UI cannot
    show person groups whose source detections no longer exist.
    """
    exists = conn.execute("SELECT 1 FROM face_detection_runs WHERE id = ?", (run_id,)).fetchone()
    if not exists:
        return False

    cluster_ids = [
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT fc.id"
            " FROM face_clusters fc"
            " LEFT JOIN face_cluster_runs fcr ON fcr.id = fc.run_id"
            " LEFT JOIN face_cluster_members fcm ON fcm.cluster_id = fc.id"
            " LEFT JOIN face_embeddings fe ON fe.id = fcm.face_id"
            " WHERE fcr.detection_run_id = ? OR fe.run_id = ?",
            (run_id, run_id),
        ).fetchall()
    ]
    cluster_run_ids = [
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT fcr.id"
            " FROM face_cluster_runs fcr"
            " LEFT JOIN face_clusters fc ON fc.run_id = fcr.id"
            " LEFT JOIN face_cluster_members fcm ON fcm.cluster_id = fc.id"
            " LEFT JOIN face_embeddings fe ON fe.id = fcm.face_id"
            " WHERE fcr.detection_run_id = ? OR fe.run_id = ?",
            (run_id, run_id),
        ).fetchall()
    ]

    if cluster_ids:
        conn.executemany(
            "DELETE FROM face_cluster_members WHERE cluster_id = ?",
            [(cid,) for cid in cluster_ids],
        )
    if cluster_run_ids:
        conn.executemany(
            "DELETE FROM face_clusters WHERE run_id = ?",
            [(rid,) for rid in cluster_run_ids],
        )
        conn.executemany(
            "DELETE FROM face_cluster_runs WHERE id = ?",
            [(rid,) for rid in cluster_run_ids],
        )
    conn.execute("DELETE FROM face_embeddings WHERE run_id = ?", (run_id,))
    result = conn.execute("DELETE FROM face_detection_runs WHERE id = ?", (run_id,))
    conn.commit()
    return result.rowcount > 0


def list_face_cluster_runs(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return all face cluster runs, most recent first."""
    rows = conn.execute(
        "SELECT r.id, r.method, r.params_json, r.detection_run_id, r.created_at,"
        "   COUNT(DISTINCT fc.id) AS n_clusters,"
        "   COUNT(fcm.face_id) AS n_faces,"
        "   COUNT(DISTINCT fe.asset_id) AS n_assets,"
        "   MIN(fe_rep.asset_id) AS rep_asset_id"
        " FROM face_cluster_runs r"
        " LEFT JOIN face_clusters fc ON fc.run_id = r.id"
        " LEFT JOIN face_cluster_members fcm ON fcm.cluster_id = fc.id"
        " LEFT JOIN face_embeddings fe ON fe.id = fcm.face_id"
        " LEFT JOIN face_cluster_members fcm_rep"
        "   ON fcm_rep.cluster_id = fc.id AND fcm_rep.is_representative = 1"
        " LEFT JOIN face_embeddings fe_rep ON fe_rep.id = fcm_rep.face_id"
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
            "n_faces": row["n_faces"],
            "n_assets": row["n_assets"],
            "rep_asset_id": row["rep_asset_id"],
        }
        for row in rows
    ]


def get_face_cluster_run(conn: sqlite3.Connection, run_id: int) -> dict[str, Any] | None:
    """Return one face clustering run with aggregate counts, or ``None``."""
    row = conn.execute(
        "SELECT r.id, r.method, r.params_json, r.detection_run_id, r.created_at,"
        "   COUNT(DISTINCT fc.id) AS n_clusters,"
        "   COUNT(fcm.face_id) AS n_faces,"
        "   COUNT(DISTINCT fe.asset_id) AS n_assets"
        " FROM face_cluster_runs r"
        " LEFT JOIN face_clusters fc ON fc.run_id = r.id"
        " LEFT JOIN face_cluster_members fcm ON fcm.cluster_id = fc.id"
        " LEFT JOIN face_embeddings fe ON fe.id = fcm.face_id"
        " WHERE r.id = ?"
        " GROUP BY r.id",
        (run_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "run_id": row["id"],
        "method": row["method"],
        "params_json": row["params_json"],
        "detection_run_id": row["detection_run_id"],
        "created_at": row["created_at"],
        "n_clusters": row["n_clusters"],
        "n_faces": row["n_faces"],
        "n_assets": row["n_assets"],
    }


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


# ---------------------------------------------------------------------------
# Album helpers
# ---------------------------------------------------------------------------


def upsert_album(conn: sqlite3.Connection, name: str, relpath: str) -> int:
    """Insert or update an album row and return its ID.

    Uses ``INSERT … ON CONFLICT(relpath) DO UPDATE`` so re-indexing is
    idempotent — the same directory will always resolve to the same album row.

    Args:
        conn: Open database connection.
        name: Human-readable album name (typically the directory name).
        relpath: Path of the album directory relative to the takeout root.

    Returns:
        The integer primary key of the album row.
    """
    now = int(time.time())
    row = conn.execute(
        "INSERT INTO albums (name, relpath, indexed_at)"
        " VALUES (?, ?, ?)"
        " ON CONFLICT(relpath) DO UPDATE SET name = excluded.name, indexed_at = excluded.indexed_at"
        " RETURNING id",
        (name, relpath, now),
    ).fetchone()
    conn.commit()
    return row[0]


def link_asset_to_album(conn: sqlite3.Connection, album_id: int, asset_id: int) -> None:
    """Record that *asset_id* belongs to *album_id*.

    Uses ``INSERT OR IGNORE`` so calling this multiple times for the same pair
    is safe (idempotent).

    Args:
        conn: Open database connection.
        album_id: Primary key of the album.
        asset_id: Primary key of the asset.
    """
    conn.execute(
        "INSERT OR IGNORE INTO album_assets (album_id, asset_id) VALUES (?, ?)",
        (album_id, asset_id),
    )
    conn.commit()


def list_albums(conn: sqlite3.Connection) -> list[dict]:
    """Return all albums, each enriched with photo count and a cover asset ID.

    Albums are ordered by name.  The cover asset is the most-recently-taken
    photo in the album (by ``taken_at DESC``).

    Args:
        conn: Open database connection.

    Returns:
        List of dicts with keys ``id``, ``name``, ``relpath``, ``indexed_at``,
        ``photo_count``, and ``cover_asset_id`` (may be ``None``).
    """
    rows = conn.execute(
        """
        SELECT
            al.id,
            al.name,
            al.relpath,
            al.indexed_at,
            COUNT(aa.asset_id) AS photo_count,
            (
                SELECT aa2.asset_id
                FROM album_assets aa2
                JOIN assets a2 ON a2.id = aa2.asset_id
                WHERE aa2.album_id = al.id
                ORDER BY a2.taken_at DESC NULLS LAST
                LIMIT 1
            ) AS cover_asset_id
        FROM albums al
        LEFT JOIN album_assets aa ON aa.album_id = al.id
        GROUP BY al.id
        ORDER BY al.name COLLATE NOCASE
        """
    ).fetchall()
    return [dict(row) for row in rows]


def get_album(conn: sqlite3.Connection, album_id: int) -> dict | None:
    """Return album metadata for *album_id*, or ``None`` if not found.

    Args:
        conn: Open database connection.
        album_id: Primary key of the album.

    Returns:
        Dict with keys ``id``, ``name``, ``relpath``, ``indexed_at``, and
        ``photo_count``, or ``None``.
    """
    row = conn.execute(
        """
        SELECT
            al.id,
            al.name,
            al.relpath,
            al.indexed_at,
            COUNT(aa.asset_id) AS photo_count
        FROM albums al
        LEFT JOIN album_assets aa ON aa.album_id = al.id
        WHERE al.id = ?
        GROUP BY al.id
        """,
        (album_id,),
    ).fetchone()
    return dict(row) if row else None


def get_album_assets(
    conn: sqlite3.Connection,
    album_id: int,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> list[AssetRow]:
    """Return all assets belonging to *album_id*, ordered by taken_at DESC.

    Args:
        conn: Open database connection.
        album_id: Primary key of the album.
        limit: Maximum number of assets to return (``None`` = no limit).
        offset: Rows to skip for pagination.

    Returns:
        List of :class:`AssetRow` objects.
    """
    params: list[Any] = [album_id]
    limit_clause = ""
    if limit is not None:
        limit_clause = " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
    sql = (
        "SELECT a.* FROM assets a"
        " JOIN album_assets aa ON aa.asset_id = a.id"
        " WHERE aa.album_id = ?"
        " ORDER BY a.taken_at DESC NULLS LAST, a.filename"
        f"{limit_clause}"  # noqa: S608
    )
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_asset(row) for row in rows]


def count_album_assets(conn: sqlite3.Connection, album_id: int) -> int:
    """Return the number of assets in *album_id*.

    Args:
        conn: Open database connection.
        album_id: Primary key of the album.

    Returns:
        Integer count.
    """
    row = conn.execute(
        "SELECT COUNT(*) FROM album_assets WHERE album_id = ?", (album_id,)
    ).fetchone()
    return row[0]


# ---------------------------------------------------------------------------
# Reverse-geocoded place names (asset_places)
# ---------------------------------------------------------------------------


def list_assets_for_geocoding(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return candidate rows for the place-name lookup.

    One row per asset with some non-zero coordinates, with columns ``id``,
    ``geo_lat``, ``geo_lon``, ``geo_exif_lat``, ``geo_exif_lon`` and the stored
    ``asset_places`` values ``p_lat``, ``p_lon``, ``p_ver`` (``NULL`` when the
    asset has not been geocoded yet).
    """
    sql = (
        "SELECT a.id, a.geo_lat, a.geo_lon, a.geo_exif_lat, a.geo_exif_lon,"
        " p.lat AS p_lat, p.lon AS p_lon, p.geocoder_version AS p_ver"
        " FROM assets a LEFT JOIN asset_places p ON p.asset_id = a.id"
        " WHERE (a.geo_lat IS NOT NULL AND a.geo_lon IS NOT NULL"
        "        AND (a.geo_lat != 0 OR a.geo_lon != 0))"
        "    OR (a.geo_exif_lat IS NOT NULL AND a.geo_exif_lon IS NOT NULL"
        "        AND (a.geo_exif_lat != 0 OR a.geo_exif_lon != 0))"
        " ORDER BY a.id"
    )
    return conn.execute(sql).fetchall()


def upsert_asset_places(
    conn: sqlite3.Connection,
    rows: list[dict[str, Any]],
) -> None:
    """Insert or replace ``asset_places`` rows and commit.

    Each dict needs ``asset_id``, ``lat``, ``lon``, ``geocoder_version`` and may
    carry ``country_code``, ``country``, ``admin1``, ``admin2``, ``city``,
    ``distance_km`` (missing keys are stored as NULL).
    """
    if not rows:
        return
    now = int(time.time())
    conn.executemany(
        "INSERT OR REPLACE INTO asset_places"
        " (asset_id, lat, lon, country_code, country, admin1, admin2, city,"
        "  distance_km, geocoder_version, geocoded_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                r["asset_id"],
                r["lat"],
                r["lon"],
                r.get("country_code"),
                r.get("country"),
                r.get("admin1"),
                r.get("admin2"),
                r.get("city"),
                r.get("distance_km"),
                r["geocoder_version"],
                now,
            )
            for r in rows
        ],
    )
    conn.commit()


def delete_stale_asset_places(conn: sqlite3.Connection, keep_ids: set[int]) -> int:
    """Delete ``asset_places`` rows whose asset is not in *keep_ids*; return count.

    Used to drop stored places of assets whose GPS coordinates were cleared.
    """
    stale = [
        (r[0],)
        for r in conn.execute("SELECT asset_id FROM asset_places").fetchall()
        if r[0] not in keep_ids
    ]
    if stale:
        conn.executemany("DELETE FROM asset_places WHERE asset_id = ?", stale)
        conn.commit()
    return len(stale)


def get_asset_places(conn: sqlite3.Connection, asset_ids: list[int]) -> dict[int, Any]:
    """Return ``{asset_id: Place}`` for assets that have a named place.

    Assets without a row, or whose row has no country (coordinates far from
    any populated place), are omitted.
    """
    from takeout_rater.geo.places import Place

    out: dict[int, Any] = {}
    ids = list(dict.fromkeys(asset_ids))
    for start in range(0, len(ids), 500):
        chunk = ids[start : start + 500]
        marks = ",".join("?" * len(chunk))
        rows = conn.execute(
            "SELECT asset_id, country_code, country, admin1, admin2, city, distance_km"
            f" FROM asset_places WHERE asset_id IN ({marks})"  # noqa: S608
            " AND country_code IS NOT NULL",
            chunk,
        ).fetchall()
        for r in rows:
            out[r["asset_id"]] = Place(
                country_code=r["country_code"],
                country=r["country"] or r["country_code"],
                admin1=r["admin1"],
                admin2=r["admin2"],
                city=r["city"],
                distance_km=r["distance_km"],
            )
    return out


def get_asset_place(conn: sqlite3.Connection, asset_id: int) -> Any | None:
    """Return the :class:`~takeout_rater.geo.places.Place` of one asset, or ``None``."""
    return get_asset_places(conn, [asset_id]).get(asset_id)


def count_asset_places(conn: sqlite3.Connection) -> int:
    """Number of assets with a named place."""
    return conn.execute(
        "SELECT COUNT(*) FROM asset_places WHERE country_code IS NOT NULL"
    ).fetchone()[0]


def list_place_facets(
    conn: sqlite3.Connection,
    *,
    country_code: str | None = None,
    admin1: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return place values with photo counts for cascading browse filters.

    Always returns ``countries`` (``[{code, name, count}]``).  When
    *country_code* is given also returns ``regions`` (``[{name, count}]``) for
    that country, and ``cities`` (``[{name, count}]``) for the country, further
    narrowed to *admin1* when given.  Lists are ordered by descending count.
    """
    countries = [
        {"code": r[0], "name": r[1] or r[0], "count": r[2]}
        for r in conn.execute(
            "SELECT country_code, MAX(country), COUNT(*) AS n FROM asset_places"
            " WHERE country_code IS NOT NULL"
            " GROUP BY country_code ORDER BY n DESC, country_code"
        ).fetchall()
    ]
    result: dict[str, list[dict[str, Any]]] = {"countries": countries}
    if country_code:
        result["regions"] = [
            {"name": r[0], "count": r[1]}
            for r in conn.execute(
                "SELECT admin1, COUNT(*) AS n FROM asset_places"
                " WHERE country_code = ? AND admin1 IS NOT NULL"
                " GROUP BY admin1 ORDER BY n DESC, admin1",
                (country_code,),
            ).fetchall()
        ]
        params: list[Any] = [country_code]
        extra = ""
        if admin1:
            extra = " AND admin1 = ?"
            params.append(admin1)
        result["cities"] = [
            {"name": r[0], "count": r[1]}
            for r in conn.execute(
                "SELECT city, COUNT(*) AS n FROM asset_places"
                f" WHERE country_code = ? AND city IS NOT NULL{extra}"  # noqa: S608
                " GROUP BY city ORDER BY n DESC, city",
                params,
            ).fetchall()
        ]
    return result


# ---------------------------------------------------------------------------
# OCR text + FTS5 search
# ---------------------------------------------------------------------------

# Sentinel characters wrapped around matched terms by FTS5 ``snippet()``.  They
# never occur in OCR output, so the API layer can HTML-escape the snippet and
# then turn them into ``<mark>`` tags safely.
OCR_SNIPPET_START = "\x02"
OCR_SNIPPET_END = "\x03"

#: Shortest term the trigram FTS index can match; shorter terms use LIKE.
_FTS_MIN_TERM_CHARS = 3


def fold_for_search(text: str) -> str:
    """Strip diacritics character by character (``"Brötchen"`` → ``"Brotchen"``).

    Each character maps to exactly one character, so offsets in the folded
    string are valid offsets in the original.  Case is kept; the trigram index
    and the highlighter compare case-insensitively.
    """
    out = []
    for ch in text:
        if ch.isascii():
            out.append(ch)
            continue
        decomposed = unicodedata.normalize("NFD", ch)
        base, marks = decomposed[0], decomposed[1:]
        # Only strip when the rest is purely combining marks (é → e + ́).  Other
        # decompositions (e.g. Hangul syllables → jamo) carry meaning: keep them.
        if (
            marks
            and not unicodedata.combining(base)
            and all(unicodedata.combining(m) for m in marks)
        ):
            out.append(base)
        else:
            out.append(ch)
    return "".join(out)


def search_key(text: str) -> str:
    """Normalise text for matching: strip diacritics, then Unicode case-fold.

    Stored in ``ocr_text.text_folded`` and applied to query terms, so the FTS
    index and the ``LIKE`` path for short terms are case-insensitive for every
    script (not just ASCII) and ``casefold`` also maps ``ß`` → ``ss``
    ("strasse" finds "Straße").  Unlike :func:`fold_for_search` this may change
    the length; the highlighter works on the original text instead.
    """
    return fold_for_search(text).casefold()


def parse_ocr_query(user_query: str) -> list[str]:
    """Split free text typed by a user into search terms.

    Whitespace separates terms; text in double quotes is one term (a phrase).
    Every term is matched as a case-insensitive substring of the OCR text and
    all terms must match.
    """
    terms: list[str] = []
    for i, seg in enumerate(user_query.split('"')):
        if i % 2 == 1:
            phrase = " ".join(seg.split())
            if phrase:
                terms.append(phrase)
        else:
            terms.extend(seg.split())
    return terms


def build_ocr_fts_query(terms: list[str]) -> str:
    """Build an FTS5 MATCH expression for the terms the trigram index can match.

    Each term of at least three characters becomes a quoted string, which the
    trigram tokenizer matches as a substring.  Quoting neutralises FTS5
    operators (``OR``, ``NEAR``, ``-``, ``:``…), so arbitrary input never raises
    a syntax error.  Returns ``''`` when no term is long enough.
    """
    return " ".join(
        '"' + term.replace('"', '""') + '"' for term in terms if len(term) >= _FTS_MIN_TERM_CHARS
    )


def _like_pattern(term: str) -> str:
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _highlight(text: str, terms: list[str], context: int = 60) -> str:
    """Return an excerpt of *text* around the first match, with every match of
    *terms* (diacritic- and case-insensitive) wrapped in snippet sentinels."""
    folded = fold_for_search(text)
    pattern = re.compile(
        "|".join(
            # search_key() folds ß → ss, so let "ss" in a term highlight a ß too.
            re.sub("(?i)ss", "(?:ss|ß)", re.escape(fold_for_search(t)))
            for t in sorted(terms, key=len, reverse=True)
        ),
        re.IGNORECASE,
    )
    spans = [m.span() for m in pattern.finditer(folded)]
    first = spans[0][0] if spans else 0
    lo = max(0, first - context)
    hi = min(len(text), first + context)
    parts: list[str] = []
    pos = lo
    for start, end in spans:
        if end <= lo or start >= hi:
            continue
        start, end = max(start, lo), min(end, hi)
        parts.append(text[pos:start])
        parts.append(OCR_SNIPPET_START + text[start:end] + OCR_SNIPPET_END)
        pos = end
    parts.append(text[pos:hi])
    return ("…" if lo > 0 else "") + "".join(parts) + ("…" if hi < len(text) else "")


def upsert_ocr_texts(
    conn: sqlite3.Connection,
    rows: list[tuple[int, str, int, float | None, str, str]],
) -> None:
    """Insert or update OCR results in one transaction.

    Args:
        conn: Open database connection.
        rows: Tuples of ``(asset_id, text, n_lines, mean_score, source, engine)``.
            ``text`` is ``''`` for assets in which no text was found.
    """
    now = int(time.time())
    # ON CONFLICT DO UPDATE (not INSERT OR REPLACE) so the FTS update trigger fires.
    conn.executemany(
        "INSERT INTO ocr_text"
        " (asset_id, text, text_folded, n_lines, mean_score, source, engine, processed_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(asset_id) DO UPDATE SET"
        "  text = excluded.text, text_folded = excluded.text_folded,"
        "  n_lines = excluded.n_lines,"
        "  mean_score = excluded.mean_score, source = excluded.source,"
        "  engine = excluded.engine, processed_at = excluded.processed_at",
        [(r[0], r[1], search_key(r[1]), *r[2:], now) for r in rows],
    )
    conn.commit()


def clear_ocr_text(conn: sqlite3.Connection) -> None:
    """Delete all OCR results (the delete trigger removes their FTS entries)."""
    conn.execute("DELETE FROM ocr_text")
    conn.commit()


def list_asset_ids_without_ocr(conn: sqlite3.Connection) -> list[int]:
    """Return IDs of assets that have not been processed by the OCR job yet."""
    rows = conn.execute(
        "SELECT a.id FROM assets a"
        " LEFT JOIN ocr_text o ON o.asset_id = a.id"
        " WHERE o.asset_id IS NULL"
        " ORDER BY a.id"
    ).fetchall()
    return [row[0] for row in rows]


def get_ocr_stats(conn: sqlite3.Connection) -> dict[str, int]:
    """Return OCR coverage counts.

    Keys: ``total_assets``, ``processed`` (assets the OCR job has seen) and
    ``with_text`` (processed assets in which some text was found).
    """
    total = conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
    processed, with_text = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(CASE WHEN text != '' THEN 1 ELSE 0 END), 0) FROM ocr_text"
    ).fetchone()
    return {"total_assets": total, "processed": processed, "with_text": with_text}


def get_ocr_text(conn: sqlite3.Connection, asset_id: int) -> dict[str, Any] | None:
    """Return the stored OCR row for *asset_id*, or ``None`` if not processed."""
    row = conn.execute(
        "SELECT asset_id, text, n_lines, mean_score, source, engine, processed_at"
        " FROM ocr_text WHERE asset_id = ?",
        (asset_id,),
    ).fetchone()
    if row is None:
        return None
    keys = ("asset_id", "text", "n_lines", "mean_score", "source", "engine", "processed_at")
    return dict(zip(keys, tuple(row), strict=True))


def search_ocr_text(
    conn: sqlite3.Connection,
    user_query: str,
    *,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Substring search over OCR text, ranked by BM25.

    Terms of three or more characters go through the trigram FTS5 index;
    shorter terms (e.g. ``"7"`` in ``gleis 7``) are applied as ``LIKE``
    filters on the matching rows.

    Args:
        conn: Open database connection.
        user_query: Free text typed by the user (see :func:`parse_ocr_query`).
        limit: Page size.
        offset: Number of ranked results to skip.

    Returns:
        ``(results, total)`` where each result is
        ``{"asset_id", "rank", "snippet"}``.  ``rank`` is the BM25 score (lower
        is better; ``0`` when only short terms were given).  ``snippet``
        contains :data:`OCR_SNIPPET_START` / :data:`OCR_SNIPPET_END` around
        matches and is *not* HTML-escaped.
    """
    terms = parse_ocr_query(user_query)
    if not terms:
        return [], 0
    match = build_ocr_fts_query([search_key(t) for t in terms])
    short_terms = [t for t in terms if len(search_key(t)) < _FTS_MIN_TERM_CHARS]
    like_sql = "".join(" AND o.text_folded LIKE ? ESCAPE '\\'" for _ in short_terms)
    like_args = [_like_pattern(search_key(t)) for t in short_terms]

    if match:
        from_where = (
            " FROM ocr_text_fts JOIN ocr_text o ON o.asset_id = ocr_text_fts.rowid"
            " WHERE ocr_text_fts MATCH ?" + like_sql
        )
        args: list[Any] = [match, *like_args]
        total = conn.execute("SELECT COUNT(*)" + from_where, args).fetchone()[0]
        rows = conn.execute(
            "SELECT o.asset_id, bm25(ocr_text_fts), o.text" + from_where + " ORDER BY"
            " bm25(ocr_text_fts), o.asset_id LIMIT ? OFFSET ?",
            [*args, limit, offset],
        ).fetchall()
    else:
        from_where = " FROM ocr_text o WHERE o.text_folded != ''" + like_sql
        total = conn.execute("SELECT COUNT(*)" + from_where, like_args).fetchone()[0]
        rows = conn.execute(
            "SELECT o.asset_id, 0.0, o.text" + from_where + " ORDER BY o.asset_id LIMIT ? OFFSET ?",
            [*like_args, limit, offset],
        ).fetchall()

    results = [
        {"asset_id": int(r[0]), "rank": float(r[1]), "snippet": _highlight(r[2] or "", terms)}
        for r in rows
    ]
    return results, int(total)
