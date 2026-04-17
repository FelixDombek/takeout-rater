"""Local configuration for takeout-rater.

Stores user preferences (e.g. the photos root path) in a JSON file at the
project root so they survive restarts without being committed to version control.
"""

from __future__ import annotations

import json
from pathlib import Path

# Config file lives at the repository root, alongside the launcher scripts.
# Directory layout:  <repo>/src/takeout_rater/config.py  →  parents[3] = <repo>
# It is listed in .gitignore and is never committed.
_CONFIG_FILE = Path(__file__).resolve().parents[3] / ".takeout-rater.json"


def _load() -> dict:
    """Load the config file, returning an empty dict if absent or unreadable."""
    try:
        return json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(data: dict) -> None:
    """Persist *data* to the config file."""
    _CONFIG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _migrate_legacy(data: dict) -> dict:
    """Migrate old ``takeout_path`` config key to ``photos_root``.

    The old design stored the directory that *contained* the ``Takeout/``
    folder.  The new design stores the photos root directly (the directory
    that contains the album sub-folders).  When an old config is detected we
    resolve the photos root using the old logic and update the config in-place.
    """
    if "takeout_path" in data and "photos_root" not in data:
        from takeout_rater.indexing.scanner import resolve_photos_root  # noqa: PLC0415

        legacy = Path(data["takeout_path"])
        data["photos_root"] = str(resolve_photos_root(legacy))
        del data["takeout_path"]
        _save(data)
    return data


def get_photos_root() -> Path | None:
    """Return the configured photos root directory, or *None* if not set."""
    data = _migrate_legacy(_load())
    raw = data.get("photos_root")
    return Path(raw) if raw else None


def set_photos_root(path: Path) -> None:
    """Persist *path* as the photos root directory."""
    data = _load()
    data.pop("takeout_path", None)  # remove legacy key if present
    data["photos_root"] = str(path)
    _save(data)


def get_db_root() -> Path | None:
    """Return the directory where the ``takeout-rater/`` state dir lives.

    When not explicitly set, the caller should fall back to :func:`get_photos_root`.
    """
    data = _migrate_legacy(_load())
    raw = data.get("db_root")
    return Path(raw) if raw else None


def set_db_root(path: Path | None) -> None:
    """Persist *path* as the DB root directory (``None`` clears the setting)."""
    data = _load()
    if path is None:
        data.pop("db_root", None)
    else:
        data["db_root"] = str(path)
    _save(data)


# ---------------------------------------------------------------------------
# Backward-compat aliases (kept so existing callers don't hard-break)
# ---------------------------------------------------------------------------


def get_takeout_path() -> Path | None:
    """Deprecated alias for :func:`get_photos_root`."""
    return get_photos_root()


def set_takeout_path(path: Path) -> None:
    """Deprecated alias for :func:`set_photos_root`."""
    set_photos_root(path)
