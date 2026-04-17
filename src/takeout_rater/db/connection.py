"""Database connection factory for takeout-rater.

Usage::

    from takeout_rater.db.connection import open_library_db

    conn = open_library_db(db_root)
    # ... use conn ...
    conn.close()
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from takeout_rater.db.schema import migrate

# Sub-directory name inside the db root
_STATE_DIR = "takeout-rater"
_DB_FILENAME = "library.sqlite"


def library_state_dir(db_root: Path) -> Path:
    """Return the ``takeout-rater/`` state directory for *db_root*.

    Creates the directory (and any parents) if it does not exist.

    Args:
        db_root: The directory under which the ``takeout-rater/`` state
            sub-directory should be created.  Previously this was called
            ``library_root``; it may now differ from the photos root when
            the user stores DB state in a separate location.
    """
    state_dir = db_root / _STATE_DIR
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def library_db_path(db_root: Path) -> Path:
    """Return the path to the library SQLite database without creating any directories.

    Use this for existence checks.  To open the database, use
    :func:`open_library_db` instead.

    Args:
        db_root: The directory that contains (or will contain) the
            ``takeout-rater/`` state sub-directory.
    """
    return db_root / _STATE_DIR / _DB_FILENAME


def open_db(db_path: Path) -> sqlite3.Connection:
    """Open an existing SQLite database at *db_path* without running migrations.

    This is intended for per-request connections where migrations have already
    been applied by :func:`open_library_db`.

    Returns:
        An open :class:`sqlite3.Connection` with ``row_factory`` set to
        :data:`sqlite3.Row` for convenient column access by name.
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    # Retry for up to 30 seconds when another connection holds the write lock.
    # The default (0 ms) causes immediate OperationalError under concurrent
    # writers, which can crash background worker threads.
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def open_library_db(db_root: Path) -> sqlite3.Connection:
    """Open (or create) the library SQLite database for *db_root*.

    Applies any pending migrations automatically.

    Args:
        db_root: The directory under which the ``takeout-rater/`` state
            directory (and therefore the database) will be created.  This
            may be the photos root itself, or a separate directory chosen by
            the user to keep DB state outside the photo archive.

    Returns:
        An open :class:`sqlite3.Connection` with ``row_factory`` set to
        :data:`sqlite3.Row` for convenient column access by name.
    """
    state_dir = library_state_dir(db_root)
    db_path = state_dir / _DB_FILENAME

    conn = open_db(db_path)
    migrate(conn)
    return conn
