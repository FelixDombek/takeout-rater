"""SQL for the smart album tables (schema v19).

Kept apart from :mod:`takeout_rater.db.queries` because the tables are owned by
one feature (:mod:`takeout_rater.smart_albums`).  Suggestions are replaced on
every run; rows with ``kept = 1`` are the user's and are never touched by a run.
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

import numpy as np

_DIM = 768  # CLIP ViT-L/14

_COLUMNS = """
    s.id, s.auto_name, s.custom_name,
    COALESCE(s.custom_name, s.auto_name) AS name,
    s.terms, s.kept, s.cohesion, s.created_at,
    -- Live count and cover, so a rescan that removes photos is reflected.
    (SELECT COUNT(*) FROM smart_album_assets sa WHERE sa.smart_album_id = s.id) AS n_assets,
    COALESCE(
        s.cover_asset_id,
        (SELECT sa.asset_id FROM smart_album_assets sa WHERE sa.smart_album_id = s.id
         ORDER BY sa.similarity DESC LIMIT 1)
    ) AS cover_asset_id
"""

_KEYS = (
    "id",
    "auto_name",
    "custom_name",
    "name",
    "terms",
    "kept",
    "cohesion",
    "created_at",
    "n_assets",
    "cover_asset_id",
)


def _row(r: tuple) -> dict[str, Any]:
    d = dict(zip(_KEYS, r, strict=True))
    try:
        d["terms"] = [{"term": t, "score": s} for t, s in json.loads(d["terms"] or "[]")]
    except (ValueError, TypeError):
        d["terms"] = []
    d["kept"] = bool(d["kept"])
    return d


def load_embeddings(conn: sqlite3.Connection) -> tuple[list[int], np.ndarray]:
    """Return ``(asset_ids, matrix)`` of CLIP embeddings for non-trashed assets."""
    rows = conn.execute(
        """
        SELECT ce.asset_id, ce.embedding
        FROM clip_embeddings ce
        JOIN assets a ON a.id = ce.asset_id
        WHERE COALESCE(a.trashed, 0) = 0
        ORDER BY ce.asset_id
        """
    ).fetchall()
    rows = [r for r in rows if r[1] is not None and len(r[1]) == _DIM * 4]
    if not rows:
        return [], np.zeros((0, _DIM), dtype=np.float32)
    ids = [r[0] for r in rows]
    mat = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), -1)
    return ids, mat


def load_kept_members(conn: sqlite3.Connection) -> list[set[int]]:
    """Asset ids of every kept album (used to avoid re-suggesting them)."""
    out: dict[int, set[int]] = {}
    for album_id, asset_id in conn.execute(
        "SELECT sa.smart_album_id, sa.asset_id FROM smart_album_assets sa"
        " JOIN smart_albums s ON s.id = sa.smart_album_id WHERE s.kept = 1"
    ):
        out.setdefault(album_id, set()).add(asset_id)
    return list(out.values())


def replace_suggestions(conn: sqlite3.Connection, albums: list[dict[str, Any]]) -> int:
    """Delete all non-kept albums and insert *albums* in one transaction.

    Each album dict has ``name``, ``terms`` (list of ``(term, score)``),
    ``asset_ids``, ``similarities``, ``cohesion`` and ``cover_asset_id``.

    Returns:
        Number of albums inserted.
    """
    now = int(time.time())
    with conn:
        conn.execute(
            "DELETE FROM smart_album_assets WHERE smart_album_id IN"
            " (SELECT id FROM smart_albums WHERE kept = 0)"
        )
        conn.execute("DELETE FROM smart_albums WHERE kept = 0")
        for a in albums:
            cur = conn.execute(
                "INSERT INTO smart_albums (auto_name, terms, kept, n_assets, cohesion,"
                " cover_asset_id, created_at) VALUES (?, ?, 0, ?, ?, ?, ?)",
                (
                    a["name"],
                    json.dumps([[t, s] for t, s in a["terms"]]),
                    len(a["asset_ids"]),
                    a["cohesion"],
                    a.get("cover_asset_id"),
                    now,
                ),
            )
            album_id = cur.lastrowid
            conn.executemany(
                "INSERT INTO smart_album_assets (smart_album_id, asset_id, similarity)"
                " VALUES (?, ?, ?)",
                [
                    (album_id, aid, sim)
                    for aid, sim in zip(a["asset_ids"], a["similarities"], strict=True)
                ],
            )
    return len(albums)


def list_smart_albums(conn: sqlite3.Connection, *, kept: bool | None = None) -> list[dict]:
    """All smart albums; kept ones first, then by size."""
    where = "" if kept is None else f"WHERE s.kept = {1 if kept else 0}"
    rows = conn.execute(
        f"SELECT {_COLUMNS} FROM smart_albums s {where} ORDER BY s.kept DESC, n_assets DESC, s.id"
    ).fetchall()
    albums = [_row(r) for r in rows]
    _add_previews(conn, albums)
    return albums


def _add_previews(conn: sqlite3.Connection, albums: list[dict], n: int = 3) -> None:
    """Set ``preview_ids``: the cover followed by the most typical other photos."""
    top: dict[int, list[int]] = {}
    for album_id, asset_id in conn.execute(
        """
        SELECT smart_album_id, asset_id FROM (
            SELECT smart_album_id, asset_id, ROW_NUMBER() OVER (
                PARTITION BY smart_album_id ORDER BY similarity DESC, asset_id
            ) AS rn
            FROM smart_album_assets
        ) WHERE rn <= ?
        ORDER BY smart_album_id, rn
        """,
        (n + 1,),
    ):
        top.setdefault(album_id, []).append(asset_id)
    for a in albums:
        ids = [a["cover_asset_id"]] if a["cover_asset_id"] is not None else []
        ids += [i for i in top.get(a["id"], []) if i not in ids]
        a["preview_ids"] = ids[:n]


def get_smart_album(conn: sqlite3.Connection, album_id: int) -> dict | None:
    r = conn.execute(
        f"SELECT {_COLUMNS} FROM smart_albums s WHERE s.id = ?", (album_id,)
    ).fetchone()
    return _row(r) if r else None


def get_smart_album_assets(
    conn: sqlite3.Connection,
    album_id: int,
    *,
    sort: str = "typical",
    limit: int,
    offset: int,
) -> list[dict[str, Any]]:
    """Members of an album; ``sort`` is ``typical`` (most similar first) or ``date``."""
    order = "a.taken_at, a.id" if sort == "date" else "sa.similarity DESC, a.id"
    rows = conn.execute(
        f"""
        SELECT a.id, a.filename, a.title, a.taken_at, a.favorited, sa.similarity
        FROM smart_album_assets sa
        JOIN assets a ON a.id = sa.asset_id
        WHERE sa.smart_album_id = ?
        ORDER BY {order}
        LIMIT ? OFFSET ?
        """,
        (album_id, limit, offset),
    ).fetchall()
    return [
        {
            "id": r[0],
            "filename": r[1],
            "title": r[2],
            "taken_at": r[3],
            "favorited": r[4],
            "similarity": r[5],
        }
        for r in rows
    ]


def get_smart_albums_for_asset(conn: sqlite3.Connection, asset_id: int) -> list[dict]:
    rows = conn.execute(
        f"SELECT {_COLUMNS} FROM smart_album_assets m JOIN smart_albums s"
        " ON s.id = m.smart_album_id WHERE m.asset_id = ? ORDER BY s.kept DESC, s.id",
        (asset_id,),
    ).fetchall()
    return [_row(r) for r in rows]


def rename_smart_album(conn: sqlite3.Connection, album_id: int, name: str | None) -> bool:
    """Set (or clear, with an empty name) the custom name.  Renaming also keeps it."""
    name = (name or "").strip() or None
    with conn:
        cur = conn.execute(
            "UPDATE smart_albums SET custom_name = ?, kept = CASE WHEN ? IS NULL THEN kept"
            " ELSE 1 END WHERE id = ?",
            (name, name, album_id),
        )
    return cur.rowcount > 0


def set_smart_album_kept(conn: sqlite3.Connection, album_id: int, kept: bool) -> bool:
    with conn:
        cur = conn.execute(
            "UPDATE smart_albums SET kept = ? WHERE id = ?", (1 if kept else 0, album_id)
        )
    return cur.rowcount > 0


def delete_smart_album(conn: sqlite3.Connection, album_id: int) -> bool:
    with conn:
        conn.execute("DELETE FROM smart_album_assets WHERE smart_album_id = ?", (album_id,))
        cur = conn.execute("DELETE FROM smart_albums WHERE id = ?", (album_id,))
    return cur.rowcount > 0


def smart_album_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    r = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(kept), 0), MAX(CASE WHEN kept = 0 THEN created_at END)"
        " FROM smart_albums"
    ).fetchone()
    n_embedded = conn.execute("SELECT COUNT(*) FROM clip_embeddings").fetchone()[0]
    return {
        "n_albums": r[0],
        "n_kept": r[1],
        "n_suggested": r[0] - r[1],
        "suggested_at": r[2],
        "n_embedded": n_embedded,
    }
