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
