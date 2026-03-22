"""Database connection factory for takeout-rater.

Usage::

    from takeout_rater.db.connection import open_library_db

    conn = open_library_db(library_root)
    # ... use conn ...
    conn.close()
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from takeout_rater.db.schema import migrate

# Sub-directory name inside the library root
_STATE_DIR = "takeout-rater"
_DB_FILENAME = "library.sqlite"


def library_state_dir(library_root: Path) -> Path:
    """Return the ``takeout-rater/`` state directory for *library_root*.

    Creates the directory (and any parents) if it does not exist.
    """
    state_dir = library_root / _STATE_DIR
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def library_db_path(library_root: Path) -> Path:
    """Return the path to the library SQLite database without creating any directories.

    Use this for existence checks.  To open the database, use
    :func:`open_library_db` instead.
    """
    return library_root / _STATE_DIR / _DB_FILENAME


def open_library_db(library_root: Path) -> sqlite3.Connection:
    """Open (or create) the library SQLite database for *library_root*.

    Applies any pending migrations automatically.

    Args:
        library_root: The directory that contains the ``Takeout/`` folder.
            The database will be created at
            ``<library_root>/takeout-rater/library.sqlite``.

    Returns:
        An open :class:`sqlite3.Connection` with ``row_factory`` set to
        :data:`sqlite3.Row` for convenient column access by name.
    """
    state_dir = library_state_dir(library_root)
    db_path = state_dir / _DB_FILENAME

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for better concurrent read performance
    conn.execute("PRAGMA journal_mode=WAL")
    # Enable foreign-key enforcement
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    return conn
