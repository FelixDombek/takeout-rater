"""Local configuration for takeout-rater.

Stores user preferences (e.g. the Takeout library path) in a JSON file at the
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


def get_takeout_path() -> Path | None:
    """Return the configured Takeout library root, or *None* if not set."""
    raw = _load().get("takeout_path")
    return Path(raw) if raw else None


def set_takeout_path(path: Path) -> None:
    """Persist *path* as the Takeout library root."""
    data = _load()
    data["takeout_path"] = str(path)
    _save(data)
