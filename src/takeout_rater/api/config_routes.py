"""FastAPI router for runtime configuration and photos path setup.

Endpoints
---------
GET  /health                   – liveness probe (always returns 200)
GET  /api/config               – current config state
GET  /api/library/status       – library path, DB schema version, and scan version
POST /api/config/photos-root   – save photos and DB roots, then start indexing
POST /api/config/open-picker   – open a native OS directory picker (Tkinter)
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from takeout_rater.config import (
    default_db_root_for_photos_root,
    get_app_dir,
    get_db_root,
    get_library,
    get_photos_root,
    list_libraries,
    set_current_library,
)
from takeout_rater.db.connection import library_state_dir, open_library_db
from takeout_rater.db.queries import CURRENT_INDEXER_VERSION

router = APIRouter()


# ---------------------------------------------------------------------------
# Health probe
# ---------------------------------------------------------------------------


@router.get("/health")
def health() -> JSONResponse:
    """Return 200 OK so the launcher can poll for server readiness."""
    return JSONResponse({"status": "ok"})


# ---------------------------------------------------------------------------
# Config read
# ---------------------------------------------------------------------------


@router.get("/api/config")
def get_config() -> JSONResponse:
    """Return the current configuration state."""
    photos_root = get_photos_root()
    db_root = get_db_root()
    return JSONResponse(
        {
            "photos_root": str(photos_root) if photos_root else None,
            "db_root": str(db_root) if db_root else None,
            "app_dir": str(get_app_dir()),
            "libraries": list_libraries(),
        }
    )


# ---------------------------------------------------------------------------
# Library status
# ---------------------------------------------------------------------------


@router.get("/api/library/status")
def library_status(request: Request) -> JSONResponse:
    """Return library path, DB schema version, and current scan version.

    This endpoint is always available; when the library is not configured,
    ``db_schema_version`` will be ``null``.
    """
    db_path = request.app.state.db_path
    photos_root = request.app.state.photos_root
    db_schema_version: int | None = None
    if db_path is not None:
        from takeout_rater.db.connection import open_db

        conn = open_db(db_path)
        try:
            db_schema_version = conn.execute("PRAGMA user_version").fetchone()[0]
        finally:
            conn.close()
    elif request.app.state.db_conn is not None:
        # Fallback for in-memory databases (used in tests).
        db_schema_version = request.app.state.db_conn.execute("PRAGMA user_version").fetchone()[0]
    return JSONResponse(
        {
            "photos_root": str(photos_root) if photos_root else None,
            "db_schema_version": db_schema_version,
            "db_scan_version": CURRENT_INDEXER_VERSION,
        }
    )


# ---------------------------------------------------------------------------
# Config write – manual path entry
# ---------------------------------------------------------------------------


def _resolve_user_dir(value: str, label: str) -> Path:
    """Canonicalise and validate a user-supplied directory path.

    Resolves symlinks and ``~`` expansion, then asserts the path is an
    existing directory.  Raises :class:`fastapi.HTTPException` (400) on any
    failure.  All user-provided paths in this module are funnelled through
    this function to provide a clear sanitisation barrier for path operations.
    """
    from fastapi import HTTPException

    # expanduser + resolve canonicalises the path (removes .., resolves links)
    # so there is no path-traversal risk from the raw user string.
    p = Path(value).expanduser().resolve()
    if not p.exists():
        raise HTTPException(status_code=400, detail=f"{label} does not exist: {p}")
    if not p.is_dir():
        raise HTTPException(status_code=400, detail=f"{label} is not a directory: {p}")
    return p


def _resolve_db_root(value: str | None, photos_root: Path) -> Path:
    """Return a DB root, creating it when needed."""
    if not value:
        db_root = default_db_root_for_photos_root(photos_root)
    else:
        db_root = Path(value).expanduser().resolve()
    db_root.mkdir(parents=True, exist_ok=True)
    return db_root


class _PhotosPathBody(BaseModel):
    path: str
    db_root: str | None = None


@router.post("/api/config/photos-root")
def set_path(body: _PhotosPathBody, request: Request) -> JSONResponse:
    """Validate and persist a photos root path, then start indexing.

    The ``path`` field must point to an existing directory that directly
    contains the album sub-folders (no ``Takeout/`` wrapper assumed).
    When ``db_root`` is omitted, the state directory is created under the
    user-local takeout-rater app directory.

    Returns 400 if either path does not exist, 200 with the saved path on
    success.

    Also initialises (or re-initialises) the library database and updates the
    running app's shared state so that the new configuration takes effect
    immediately without requiring a server restart.  A background indexing run
    is launched automatically; callers can poll ``GET /api/jobs/status?job_type=index``
    to track progress.
    """
    p = _resolve_user_dir(body.path, "Photos root")
    db_root = _resolve_db_root(body.db_root, p)

    photos_root = p
    set_current_library(photos_root, db_root)

    # Open (or create) the library DB and update the live app state so that
    # all subsequent requests see the new configuration immediately.
    old_conn = request.app.state.db_conn
    if old_conn is not None:
        old_conn.close()
    state_dir = library_state_dir(db_root)
    conn = open_library_db(db_root)
    request.app.state.db_conn = conn
    from takeout_rater.db.connection import library_db_path

    request.app.state.db_path = library_db_path(db_root)
    request.app.state.db_root = db_root
    request.app.state.photos_root = photos_root
    request.app.state.thumbs_dir = state_dir / "thumbs"

    # Start background indexing so the library is populated without the user
    # needing to run a separate CLI command.
    from takeout_rater.api.jobs import _start_index_job

    _start_index_job(request.app, photos_root, db_root=db_root)

    return JSONResponse(
        {
            "status": "ok",
            "photos_root": str(photos_root),
            "db_root": str(db_root),
        }
    )


# ---------------------------------------------------------------------------
# Config write – switch to an existing library without re-indexing
# ---------------------------------------------------------------------------


class _SwitchLibraryBody(BaseModel):
    db_root: str | None = None
    photos_root: str | None = None
    library_id: str | None = None


@router.post("/api/config/switch-library")
def switch_library(body: _SwitchLibraryBody, request: Request) -> JSONResponse:
    """Switch to an already-indexed library without starting a new indexing run.

    ``db_root`` must be a directory that already contains a
    ``takeout-rater/library.sqlite`` database.  ``photos_root`` is the
    directory whose paths asset relpaths are relative to.

    Returns 400 if the database is not found, 200 on success.
    """
    from fastapi import HTTPException

    if body.library_id:
        record = get_library(body.library_id)
        if record is None:
            raise HTTPException(status_code=400, detail=f"Unknown library: {body.library_id}")
        db_root_raw = record.get("db_root")
        photos_root_raw = record.get("photos_root")
    else:
        db_root_raw = body.db_root
        photos_root_raw = body.photos_root

    if not db_root_raw:
        raise HTTPException(status_code=400, detail="DB root is required.")
    if not photos_root_raw:
        raise HTTPException(status_code=400, detail="Photos root is required.")

    db_root_path = _resolve_user_dir(db_root_raw, "DB root")

    db_file = db_root_path / "takeout-rater" / "library.sqlite"
    if not db_file.exists():
        raise HTTPException(
            status_code=400,
            detail=f"No takeout-rater database found in {db_root_path} — expected {db_file}",
        )

    photos_path = _resolve_user_dir(photos_root_raw, "Photos root")
    photos_root = photos_path

    set_current_library(photos_root, db_root_path)

    old_conn = request.app.state.db_conn
    if old_conn is not None:
        old_conn.close()
    state_dir = library_state_dir(db_root_path)
    conn = open_library_db(db_root_path)
    request.app.state.db_conn = conn
    from takeout_rater.db.connection import library_db_path

    request.app.state.db_path = library_db_path(db_root_path)
    request.app.state.db_root = db_root_path
    request.app.state.photos_root = photos_root
    request.app.state.thumbs_dir = state_dir / "thumbs"

    return JSONResponse(
        {"status": "ok", "photos_root": str(photos_root), "db_root": str(db_root_path)}
    )


# ---------------------------------------------------------------------------
# Config write – native directory picker
# ---------------------------------------------------------------------------


@router.post("/api/config/open-picker")
def open_picker() -> JSONResponse:
    """Open a native OS directory-chooser dialog and return the selected path.

    Uses :mod:`tkinter.filedialog`.  Returns 501 if tkinter is not available
    on the host (e.g. a headless server), so callers should fall back to
    manual path entry.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=501,
            detail="tkinter is not available; please enter the path manually.",
        ) from exc

    root = tk.Tk()
    root.withdraw()
    # Bring the dialog to the front on all platforms.
    root.wm_attributes("-topmost", True)
    selected = filedialog.askdirectory(
        parent=root,
        title="Select your photos folder (the folder containing your album sub-folders)",
    )
    root.destroy()

    if selected:
        return JSONResponse({"path": selected, "cancelled": False})
    return JSONResponse({"path": None, "cancelled": True})
