"""FastAPI router for runtime configuration and photos path setup.

Endpoints
---------
GET  /health                   – liveness probe (always returns 200)
GET  /api/config               – current config state
GET  /api/library/status       – library path, DB schema version, and scan version
POST /api/config/takeout-path  – save a photos root path (and optional db_root) and start indexing
POST /api/config/open-picker   – open a native OS directory picker (Tkinter)
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from takeout_rater.config import get_db_root, get_photos_root, set_db_root, set_photos_root
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
    photos_root = get_photos_root()
    db_root = get_db_root()
    return JSONResponse(
        {
            "takeout_path": str(photos_root) if photos_root else None,
            "photos_root": str(photos_root) if photos_root else None,
            "db_root": str(db_root) if db_root else None,
            "configured": photos_root is not None,
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
    library_root = request.app.state.library_root
    db_schema_version: int | None = None
    if db_path is not None:
        from takeout_rater.db.connection import open_db  # noqa: PLC0415

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
    db_root: str | None = None


@router.post("/api/config/takeout-path")
def set_path(body: _TakeoutPathBody, request: Request) -> JSONResponse:
    """Validate and persist a photos root path, then start indexing.

    The ``path`` field must point to an existing directory that directly
    contains the album sub-folders (no ``Takeout/`` wrapper assumed).
    The optional ``db_root`` field overrides where the ``takeout-rater/``
    state directory (DB, thumbnails) is created; it defaults to ``path``
    when not supplied.

    Returns 400 if either path does not exist, 200 with the saved path on
    success.

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

    db_root: Path
    if body.db_root:
        db_root = Path(body.db_root).expanduser().resolve()
        if not db_root.exists():
            raise HTTPException(status_code=400, detail=f"DB root does not exist: {db_root}")
        if not db_root.is_dir():
            raise HTTPException(status_code=400, detail=f"DB root is not a directory: {db_root}")
    else:
        db_root = p

    set_photos_root(p)
    set_db_root(None if db_root == p else db_root)

    # Open (or create) the library DB and update the live app state so that
    # all subsequent requests see the new configuration immediately.
    old_conn = request.app.state.db_conn
    if old_conn is not None:
        old_conn.close()
    conn = open_library_db(db_root)
    request.app.state.db_conn = conn
    request.app.state.library_root = p
    from takeout_rater.db.connection import library_db_path  # noqa: PLC0415

    request.app.state.db_path = library_db_path(db_root)
    # photos root is used directly — no Takeout/ resolution needed.
    request.app.state.takeout_root = p
    request.app.state.thumbs_dir = db_root / "takeout-rater" / "thumbs"

    # Start background indexing so the library is populated without the user
    # needing to run a separate CLI command.
    from takeout_rater.api.jobs import _start_index_job

    _start_index_job(request.app, p, db_root=db_root)

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
        title="Select your photos folder (the folder containing your album sub-folders)",
    )
    root.destroy()

    if selected:
        return JSONResponse({"path": selected, "cancelled": False})
    return JSONResponse({"path": None, "cancelled": True})
