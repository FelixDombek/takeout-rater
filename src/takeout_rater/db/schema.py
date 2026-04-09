"""SQLite schema management: applying migrations on first open."""

from __future__ import annotations

import sqlite3
from pathlib import Path

# Directory containing numbered SQL migration files
_MIGRATIONS_DIR = Path(__file__).parent / "migrations"

# The single schema version this codebase targets.
CURRENT_SCHEMA_VERSION: int = 6

# The one migration file that creates the complete schema from scratch.
_MIGRATIONS: list[tuple[int, str]] = [
    (CURRENT_SCHEMA_VERSION, "0001_initial_schema.sql"),
]


class SchemaMismatchError(RuntimeError):
    """Raised when the on-disk database was created by an incompatible schema version.

    This is a breaking-change release.  Databases at schema versions 1-5 cannot
    be migrated automatically.  The library must be rebuilt from scratch with a
    complete re-scan of the Takeout folder.
    """

    def __init__(self, found_version: int) -> None:
        self.found_version = found_version
        super().__init__(
            f"Database schema version {found_version} is incompatible with the current "
            f"application (requires version {CURRENT_SCHEMA_VERSION}). "
            "Please delete the library database and run a full re-scan of your Takeout folder."
        )


def migrate(conn: sqlite3.Connection) -> None:
    """Apply any pending migrations to *conn*.

    For a fresh database (``user_version == 0``) the full schema is created and
    ``user_version`` is set to :data:`CURRENT_SCHEMA_VERSION`.

    If the database already exists at :data:`CURRENT_SCHEMA_VERSION` this
    function is a no-op.

    If the database exists at any *other* version, :class:`SchemaMismatchError`
    is raised.  Callers are expected to surface this to the user and ask them to
    delete the database and re-scan their Takeout folder.

    Args:
        conn: An open SQLite connection with ``isolation_level`` suitable
            for DDL statements (i.e. not in a read-only transaction).

    Raises:
        SchemaMismatchError: When the on-disk schema version is not 0 and not
            equal to :data:`CURRENT_SCHEMA_VERSION`.
    """
    current_version: int = conn.execute("PRAGMA user_version").fetchone()[0]

    if current_version == CURRENT_SCHEMA_VERSION:
        return

    if current_version != 0:
        raise SchemaMismatchError(current_version)

    # Fresh database – apply the single baseline schema.
    for _version, filename in _MIGRATIONS:
        sql = (_MIGRATIONS_DIR / filename).read_text(encoding="utf-8")
        conn.executescript(sql)

    conn.commit()
