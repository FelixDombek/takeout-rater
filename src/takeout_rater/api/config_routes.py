"""FastAPI router for runtime configuration and Takeout path setup.

Endpoints
---------
GET  /health                   – liveness probe (always returns 200)
GET  /api/config               – current config state
GET  /api/library/status       – library path, DB schema version, and scan version
POST /api/config/takeout-path  – save a Takeout library path and start indexing
POST /api/config/open-picker   – open a native OS directory picker (Tkinter)
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from takeout_rater.config import get_takeout_path, set_takeout_path
from takeout_rater.db.connection import open_library_db
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
    path = get_takeout_path()
    return JSONResponse(
        {
            "takeout_path": str(path) if path else None,
            "configured": path is not None,
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
    conn = request.app.state.db_conn
    library_root = request.app.state.library_root
    db_schema_version: int | None = None
    if conn is not None:
        db_schema_version = conn.execute("PRAGMA user_version").fetchone()[0]
    return JSONResponse(
        {
            "library_path": str(library_root) if library_root else None,
            "db_schema_version": db_schema_version,
            "db_scan_version": CURRENT_INDEXER_VERSION,
            "configured": library_root is not None,
        }
    )


# ---------------------------------------------------------------------------
# Config write – manual path entry
# ---------------------------------------------------------------------------


class _TakeoutPathBody(BaseModel):
    path: str


@router.post("/api/config/takeout-path")
def set_path(body: _TakeoutPathBody, request: Request) -> JSONResponse:
    """Validate and persist a Takeout library root path, then start indexing.

    The path must point to an existing directory.  Returns 400 if the path
    does not exist, 200 with the saved path on success.

    Also initialises (or re-initialises) the library database and updates the
    running app's shared state so that the new configuration takes effect
    immediately without requiring a server restart.  A background indexing run
    is launched automatically; callers can poll ``GET /api/jobs/status?job_type=index``
    to track progress.
    """
    from fastapi import HTTPException

    p = Path(body.path).expanduser().resolve()
    if not p.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {p}")
    if not p.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {p}")
    set_takeout_path(p)

    # Open (or create) the library DB and update the live app state so that
    # all subsequent requests see the new configuration immediately.
    old_conn = request.app.state.db_conn
    if old_conn is not None:
        old_conn.close()
    conn = open_library_db(p)
    request.app.state.db_conn = conn
    request.app.state.library_root = p
    # Compute the actual photos root (where relpath/sidecar_relpath are relative to).
    from takeout_rater.indexing.scanner import resolve_photos_root  # noqa: PLC0415

    request.app.state.takeout_root = resolve_photos_root(p)
    request.app.state.thumbs_dir = p / "takeout-rater" / "thumbs"

    # Start background indexing so the library is populated without the user
    # needing to run a separate CLI command.
    from takeout_rater.api.jobs import _start_index_job

    _start_index_job(request.app, p)

    return JSONResponse({"status": "ok", "takeout_path": str(p)})


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
        import tkinter as tk  # noqa: PLC0415
        from tkinter import filedialog  # noqa: PLC0415
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
        title="Select your Google Photos Takeout folder (the folder *containing* Takeout/)",
    )
    root.destroy()

    if selected:
        return JSONResponse({"path": selected, "cancelled": False})
    return JSONResponse({"path": None, "cancelled": True})
