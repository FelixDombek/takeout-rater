"""FastAPI router for runtime configuration and Takeout path setup.

Endpoints
---------
GET  /health                   – liveness probe (always returns 200)
GET  /api/config               – current config state
POST /api/config/takeout-path  – save a Takeout library path and start indexing
POST /api/config/open-picker   – open a native OS directory picker (Tkinter)
GET  /api/index/status         – backward-compatible wrapper for the index job status
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from takeout_rater.config import get_takeout_path, set_takeout_path
from takeout_rater.db.connection import open_library_db

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
    is launched automatically; callers can poll ``GET /api/index/status`` to
    track progress.
    """
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
    request.app.state.takeout_root = p
    request.app.state.thumbs_dir = p / "takeout-rater" / "thumbs"

    # Start background indexing so the library is populated without the user
    # needing to run a separate CLI command.
    _start_background_index(request.app, p)

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


# ---------------------------------------------------------------------------
# Background indexing helpers
# ---------------------------------------------------------------------------


def _start_background_index(app: object, library_root: Path) -> None:
    """Launch a background index run for *library_root*.

    This is a thin wrapper around :func:`takeout_rater.api.jobs._start_index_job`
    that stores progress in ``app.state.jobs["index"]`` so it is visible in
    the unified job queue.  Kept for backward compatibility with call-sites in
    this module and in tests.
    """
    from takeout_rater.api.jobs import _start_index_job  # noqa: PLC0415

    _start_index_job(app, library_root)


# ---------------------------------------------------------------------------
# Index status (backward-compatible wrapper)
# ---------------------------------------------------------------------------


@router.get("/api/index/status")
def index_status(request: Request) -> JSONResponse:
    """Return the current background indexing progress.

    This endpoint is a backward-compatible wrapper around
    ``GET /api/jobs/status?job_type=index``.  New callers should use the
    jobs status endpoint directly.
    """
    from takeout_rater.api.jobs import JobProgress, _get_jobs  # noqa: PLC0415

    jobs = _get_jobs(request.app)
    p: JobProgress | None = jobs.get("index")
    if p is None:
        return JSONResponse(
            {
                "running": False,
                "done": False,
                "error": None,
                "message": "",
                "scored": 0,
                "total": 0,
            }
        )
    return JSONResponse(
        {
            "running": p.running,
            "done": p.done,
            "error": p.error,
            "message": p.message,
            "scored": p.scored,
            "total": p.total,
        }
    )
