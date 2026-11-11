"""SQL for the trips/events tables (schema v15).

Kept apart from :mod:`takeout_rater.db.queries` because the tables are owned by
one feature (:mod:`takeout_rater.events`) and are rebuilt wholesale on every
detection run.
"""

from __future__ import annotations

import sqlite3
from typing import Any

_EVENT_COLUMNS = """
    e.id, e.kind, e.parent_id, e.auto_name, e.custom_name,
    COALESCE(e.custom_name, e.auto_name) AS name,
    e.place_label, e.start_ts, e.end_ts, e.n_days,
    -- Live count and cover, so a rescan that removes photos is reflected
    -- before the next detection run.
    (SELECT COUNT(*) FROM event_assets ea WHERE ea.event_id = e.id) AS n_assets,
    COALESCE(
        e.cover_asset_id,
        (SELECT MIN(ea.asset_id) FROM event_assets ea WHERE ea.event_id = e.id)
    ) AS cover_asset_id,
    e.center_lat, e.center_lon, e.max_distance_km, e.detected_at
"""


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def load_detection_points(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return ``id, taken_at, lat, lon, favorited`` for every dated, non-trashed asset.

    Coordinates prefer Google's ``geoData`` and fall back to ``geoDataExif``;
    ``0.0/0.0`` (Takeout's "unknown") becomes ``None``.
    """
    rows = conn.execute(
        """
        SELECT id, taken_at, geo_lat, geo_lon, geo_exif_lat, geo_exif_lon, favorited
        FROM assets
        WHERE taken_at IS NOT NULL AND COALESCE(trashed, 0) = 0
        """
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        lat, lon = r[2], r[3]
        if lat is None or lon is None or (lat == 0.0 and lon == 0.0):
            lat, lon = r[4], r[5]
            if lat is None or lon is None or (lat == 0.0 and lon == 0.0):
                lat = lon = None
        out.append({"id": r[0], "taken_at": r[1], "lat": lat, "lon": lon, "favorited": bool(r[6])})
    return out


def load_aesthetic_scores(conn: sqlite3.Connection) -> dict[int, float]:
    """Best available per-asset aesthetic score (LAION ``aesthetic`` scorer), if any."""
    rows = conn.execute(
        """
        SELECT asset_id, MAX(value)
        FROM asset_scores
        WHERE scorer_id = 'aesthetic' AND metric_key = 'aesthetic'
        GROUP BY asset_id
        """
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def load_asset_places(
    conn: sqlite3.Connection,
) -> dict[int, tuple[str | None, str | None, str | None]] | None:
    """Return ``{asset_id: (city, admin1, country)}`` from ``asset_places``.

    ``asset_places`` is written by the reverse-geocoding feature; returns
    ``None`` when that table does not exist in this library.
    """
    if not _table_exists(conn, "asset_places"):
        return None
    rows = conn.execute(
        "SELECT asset_id, city, admin1, country FROM asset_places WHERE country_code IS NOT NULL"
    ).fetchall()
    return {r[0]: (r[1], r[2], r[3]) for r in rows}


def load_custom_names(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return ``kind, custom_name, asset_ids`` (a set) of every user-renamed group."""
    rows = conn.execute(
        "SELECT id, kind, custom_name FROM events WHERE custom_name IS NOT NULL"
    ).fetchall()
    out = []
    for r in rows:
        ids = {
            a[0]
            for a in conn.execute(
                "SELECT asset_id FROM event_assets WHERE event_id = ?", (r[0],)
            ).fetchall()
        }
        out.append({"kind": r[1], "custom_name": r[2], "asset_ids": ids})
    return out


def replace_events(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    """Atomically replace all trips/events.

    Each row dict holds the ``events`` columns (without ``id``/``parent_id``)
    plus ``asset_ids`` and an optional ``children`` list of rows of the same
    shape, which are stored with ``parent_id`` pointing at the row.
    """
    with conn:
        conn.execute("DELETE FROM event_assets")
        conn.execute("DELETE FROM events")

        def _insert(row: dict[str, Any], parent_id: int | None) -> None:
            cur = conn.execute(
                """
                INSERT INTO events (
                    kind, parent_id, auto_name, custom_name, place_label,
                    start_ts, end_ts, n_assets, n_days, center_lat, center_lon,
                    max_distance_km, cover_asset_id, detected_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["kind"],
                    parent_id,
                    row["auto_name"],
                    row.get("custom_name"),
                    row.get("place_label"),
                    row["start_ts"],
                    row["end_ts"],
                    len(row["asset_ids"]),
                    row["n_days"],
                    row.get("center_lat"),
                    row.get("center_lon"),
                    row.get("max_distance_km"),
                    row.get("cover_asset_id"),
                    row["detected_at"],
                ),
            )
            event_id = cur.lastrowid
            conn.executemany(
                "INSERT OR IGNORE INTO event_assets (event_id, asset_id) VALUES (?, ?)",
                [(event_id, aid) for aid in row["asset_ids"]],
            )
            for child in row.get("children", []):
                _insert(child, event_id)

        for row in rows:
            _insert(row, None)


def list_events(
    conn: sqlite3.Connection,
    *,
    kind: str | None = None,
    top_level_only: bool = True,
    parent_id: int | None = None,
) -> list[dict[str, Any]]:
    """List trips/events, newest first.

    Args:
        kind: ``"trip"`` or ``"event"`` to filter, ``None`` for both.
        top_level_only: Exclude events nested inside a trip.
        parent_id: Only children of this trip (overrides *top_level_only*), oldest first.
    """
    conditions: list[str] = []
    params: list[Any] = []
    if parent_id is not None:
        conditions.append("e.parent_id = ?")
        params.append(parent_id)
    elif top_level_only:
        conditions.append("e.parent_id IS NULL")
    if kind is not None:
        conditions.append("e.kind = ?")
        params.append(kind)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    order = "ASC" if parent_id is not None else "DESC"
    rows = conn.execute(
        f"""
        SELECT {_EVENT_COLUMNS},
               (SELECT COUNT(*) FROM events c WHERE c.parent_id = e.id) AS n_children
        FROM events e
        {where}
        ORDER BY e.start_ts {order}
        """,
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def get_event(conn: sqlite3.Connection, event_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        f"""
        SELECT {_EVENT_COLUMNS},
               (SELECT COUNT(*) FROM events c WHERE c.parent_id = e.id) AS n_children
        FROM events e WHERE e.id = ?
        """,
        (event_id,),
    ).fetchone()
    return dict(row) if row is not None else None


def get_event_assets(
    conn: sqlite3.Connection, event_id: int, *, limit: int, offset: int
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT a.id, a.filename, a.title, a.taken_at, a.favorited
        FROM event_assets ea
        JOIN assets a ON a.id = ea.asset_id
        WHERE ea.event_id = ?
        ORDER BY a.taken_at, a.id
        LIMIT ? OFFSET ?
        """,
        (event_id, limit, offset),
    ).fetchall()
    return [
        {"id": r[0], "filename": r[1], "title": r[2], "taken_at": r[3], "favorited": r[4]}
        for r in rows
    ]


def get_events_for_asset(conn: sqlite3.Connection, asset_id: int) -> list[dict[str, Any]]:
    """Trips/events containing *asset_id* (trip first)."""
    rows = conn.execute(
        f"""
        SELECT {_EVENT_COLUMNS}, 0 AS n_children
        FROM event_assets ea JOIN events e ON e.id = ea.event_id
        WHERE ea.asset_id = ?
        ORDER BY e.kind DESC, e.start_ts
        """,
        (asset_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def rename_event(conn: sqlite3.Connection, event_id: int, name: str | None) -> bool:
    """Set (or with ``None``/blank, clear) the custom name. Returns ``False`` if not found."""
    name = (name or "").strip() or None
    with conn:
        # Saving the detected name unchanged must not pin it as a custom name.
        cur = conn.execute(
            "UPDATE events SET custom_name = CASE WHEN ? = auto_name THEN NULL ELSE ? END"
            " WHERE id = ?",
            (name, name, event_id),
        )
    return cur.rowcount > 0


def event_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT
            SUM(kind = 'trip'),
            SUM(kind = 'event' AND parent_id IS NULL),
            SUM(kind = 'event' AND parent_id IS NOT NULL),
            MAX(detected_at)
        FROM events
        """
    ).fetchone()
    return {
        "n_trips": row[0] or 0,
        "n_events": row[1] or 0,
        "n_nested_events": row[2] or 0,
        "detected_at": row[3],
    }


def load_event_embeddings(conn: sqlite3.Connection, event_id: int) -> dict[int, Any]:
    """CLIP ViT-L/14 embeddings of the photos in one trip/event, as ``{asset_id: float32 array}``.

    Blobs whose length differs from the most common one are skipped.
    """
    import numpy as np

    rows = conn.execute(
        """
        SELECT ce.asset_id, ce.embedding
        FROM event_assets ea JOIN clip_embeddings ce ON ce.asset_id = ea.asset_id
        WHERE ea.event_id = ?
        """,
        (event_id,),
    ).fetchall()
    if not rows:
        return {}
    lengths: dict[int, int] = {}
    for _, blob in rows:
        lengths[len(blob)] = lengths.get(len(blob), 0) + 1
    size = max(lengths, key=lambda k: lengths[k])
    if size == 0 or size % 4:
        return {}
    return {aid: np.frombuffer(blob, dtype="<f4") for aid, blob in rows if len(blob) == size}


def load_event_aesthetic_scores(conn: sqlite3.Connection, event_id: int) -> dict[int, float]:
    """LAION aesthetic score of each photo in one trip/event that has one."""
    rows = conn.execute(
        """
        SELECT s.asset_id, MAX(s.value)
        FROM event_assets ea JOIN asset_scores s ON s.asset_id = ea.asset_id
        WHERE ea.event_id = ? AND s.scorer_id = 'aesthetic' AND s.metric_key = 'aesthetic'
        GROUP BY s.asset_id
        """,
        (event_id,),
    ).fetchall()
    return {r[0]: r[1] for r in rows}
