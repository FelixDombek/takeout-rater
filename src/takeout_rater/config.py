"""User-local configuration for takeout-rater.

The web launcher stores its active library and known library list under a
stable user-local directory so the same databases are found regardless of the
current working directory.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

_APP_DIR = Path.home() / ".takeout_rater"
_CONFIG_FILE = _APP_DIR / "config.json"


def get_app_dir() -> Path:
    """Return the user-local app directory used for web UI state."""
    return _CONFIG_FILE.parent


def _library_id_for_photos_root(photos_root: Path) -> str:
    """Return a stable, filesystem-safe ID for a photos root path."""
    raw = str(photos_root.expanduser().resolve())
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", photos_root.name.strip()).strip(".-_")
    if not slug:
        slug = "library"
    return f"{slug[:48]}-{digest}"


def default_db_root_for_photos_root(photos_root: Path) -> Path:
    """Return the user-local DB root for *photos_root*.

    The library database itself lives below this directory as
    ``takeout-rater/library.sqlite`` via :mod:`takeout_rater.db.connection`.
    """
    return get_app_dir() / _library_id_for_photos_root(photos_root)


def _load() -> dict:
    """Load the config file, returning an empty dict if absent or unreadable."""
    try:
        return json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save(data: dict) -> None:
    """Persist *data* to the config file."""
    _CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _library_record(photos_root: Path, db_root: Path) -> dict[str, str]:
    library_id = _library_id_for_photos_root(photos_root)
    now = datetime.now(UTC).isoformat()
    return {
        "id": library_id,
        "name": photos_root.name or str(photos_root),
        "photos_root": str(photos_root),
        "db_root": str(db_root),
        "last_opened_at": now,
    }


def _upsert_library(data: dict, photos_root: Path, db_root: Path) -> dict[str, str]:
    record = _library_record(photos_root, db_root)
    libraries = data.setdefault("libraries", [])
    kept = [item for item in libraries if item.get("id") != record["id"]]
    kept.append(record)
    data["libraries"] = kept
    return record


def set_current_library(photos_root: Path, db_root: Path | None = None) -> Path:
    """Persist *photos_root* as the current library and return its DB root."""
    photos_root = photos_root.expanduser().resolve()
    db_root = (db_root or default_db_root_for_photos_root(photos_root)).expanduser().resolve()
    db_root.mkdir(parents=True, exist_ok=True)

    data = _load()
    record = _upsert_library(data, photos_root, db_root)
    data["photos_root"] = record["photos_root"]
    data["db_root"] = record["db_root"]
    data["current_library_id"] = record["id"]
    _save(data)
    return db_root


def list_libraries() -> list[dict[str, str]]:
    """Return known libraries, newest first."""
    libraries = _load().get("libraries", [])
    if not isinstance(libraries, list):
        return []
    records = [item for item in libraries if isinstance(item, dict)]
    return sorted(records, key=lambda item: item.get("last_opened_at", ""), reverse=True)


def get_library(library_id: str) -> dict[str, str] | None:
    """Return a known library record by ID."""
    for record in list_libraries():
        if record.get("id") == library_id:
            return record
    return None


def get_photos_root() -> Path | None:
    """Return the configured photos root directory, or *None* if not set."""
    data = _load()
    raw = data.get("photos_root")
    return Path(raw) if raw else None


def set_photos_root(path: Path) -> None:
    """Persist *path* as the photos root directory."""
    set_current_library(path)


def get_db_root() -> Path | None:
    """Return the directory where the ``takeout-rater/`` state dir lives."""
    data = _load()
    raw = data.get("db_root")
    if raw:
        return Path(raw)
    photos_root = get_photos_root()
    if photos_root is None:
        return None
    return default_db_root_for_photos_root(photos_root)


def set_db_root(path: Path | None) -> None:
    """Persist *path* as the DB root directory (``None`` clears the setting)."""
    data = _load()
    if path is None:
        data.pop("db_root", None)
    else:
        db_root = path.expanduser().resolve()
        db_root.mkdir(parents=True, exist_ok=True)
        data["db_root"] = str(db_root)
        raw_photos_root = data.get("photos_root")
        if raw_photos_root:
            record = _upsert_library(data, Path(raw_photos_root).expanduser().resolve(), db_root)
            data["current_library_id"] = record["id"]
    _save(data)
