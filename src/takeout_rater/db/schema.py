"""SQLite schema management: applying migrations on first open."""

from __future__ import annotations

import sqlite3
from pathlib import Path

# Directory containing the schema SQL file
_MIGRATIONS_DIR = Path(__file__).parent / "migrations"

# The single schema version this codebase targets.
CURRENT_SCHEMA_VERSION: int = 11

# Earliest schema version from which incremental migrations are supported.
# Databases older than this must be fully rebuilt (full re-scan).
_INCREMENTAL_MIGRATION_BASE: int = 6

# Map target_version → SQL file that upgrades from (target_version - 1) to target_version.
_INCREMENTAL_MIGRATIONS: dict[int, Path] = {
    7: _MIGRATIONS_DIR / "0002_cluster_diameter.sql",
    8: _MIGRATIONS_DIR / "0003_simple_scorer_rename.sql",
    9: _MIGRATIONS_DIR / "0004_clustering_runs.sql",
    10: _MIGRATIONS_DIR / "0005_clip_embeddings.sql",
    11: _MIGRATIONS_DIR / "0006_clip_user_tags.sql",
}


class SchemaMismatchError(RuntimeError):
    """Raised when the on-disk database was created by an incompatible schema version.

    Databases at schema versions 1–5 cannot be migrated automatically.  The
    library must be rebuilt from scratch with a complete re-scan of the Takeout
    folder.  Databases at version 6 are automatically migrated to the current
    version.
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

    If the database exists at :data:`_INCREMENTAL_MIGRATION_BASE` or higher
    (but below :data:`CURRENT_SCHEMA_VERSION`), each incremental migration SQL
    file is executed in order to bring the schema up to date.

    If the database exists at any *other* version (i.e. older than
    :data:`_INCREMENTAL_MIGRATION_BASE`), :class:`SchemaMismatchError` is
    raised.  Callers are expected to surface this to the user and ask them to
    delete the database and re-scan their Takeout folder.

    Args:
        conn: An open SQLite connection with ``isolation_level`` suitable
            for DDL statements (i.e. not in a read-only transaction).

    Raises:
        SchemaMismatchError: When the on-disk schema version cannot be
            migrated to :data:`CURRENT_SCHEMA_VERSION`.
    """
    current_version: int = conn.execute("PRAGMA user_version").fetchone()[0]

    if current_version == CURRENT_SCHEMA_VERSION:
        return

    if current_version == 0:
        # Fresh database – apply the consolidated baseline schema.
        sql = (_MIGRATIONS_DIR / "0001_initial_schema.sql").read_text(encoding="utf-8")
        conn.executescript(sql)
        conn.commit()
        return

    if _INCREMENTAL_MIGRATION_BASE <= current_version < CURRENT_SCHEMA_VERSION:
        # Apply each pending incremental migration in version order.
        for target in range(current_version + 1, CURRENT_SCHEMA_VERSION + 1):
            migration_file = _INCREMENTAL_MIGRATIONS[target]
            conn.executescript(migration_file.read_text(encoding="utf-8"))
        conn.commit()
        return

    raise SchemaMismatchError(current_version)
