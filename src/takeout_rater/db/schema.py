"""SQLite schema management: applying migrations on first open."""

from __future__ import annotations

import sqlite3
from pathlib import Path

# Directory containing numbered SQL migration files
_MIGRATIONS_DIR = Path(__file__).parent / "migrations"

# Ordered list of (version, migration_file) pairs.
# Extend this list when adding new migrations.
_MIGRATIONS: list[tuple[int, str]] = [
    (1, "0001_initial_schema.sql"),
    (2, "0002_view_presets.sql"),
]


def migrate(conn: sqlite3.Connection) -> None:
    """Apply any pending migrations to *conn*.

    Migrations are applied in order.  Each migration sets
    ``PRAGMA user_version = N`` at the end, so the function can determine
    which migrations have already been applied.

    Args:
        conn: An open SQLite connection with ``isolation_level`` suitable
            for DDL statements (i.e. not in a read-only transaction).
    """
    current_version: int = conn.execute("PRAGMA user_version").fetchone()[0]

    for version, filename in _MIGRATIONS:
        if version <= current_version:
            continue
        sql = (_MIGRATIONS_DIR / filename).read_text(encoding="utf-8")
        conn.executescript(sql)

    conn.commit()
