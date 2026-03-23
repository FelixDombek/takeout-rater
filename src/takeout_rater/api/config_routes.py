"""FastAPI router for runtime configuration and Takeout path setup.

Endpoints
---------
GET  /health                   – liveness probe (always returns 200)
GET  /api/config               – current config state
POST /api/config/takeout-path  – save a Takeout library path and start indexing
POST /api/config/open-picker   – open a native OS directory picker (Tkinter)
GET  /api/index/status         – current background indexing progress
"""

from __future__ import annotations

import threading
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
    """Launch a background thread that indexes *library_root*.

    The thread opens its own database connection so that it does not compete
    with the main ASGI thread's connection for SQLite write locks.  Progress
    updates are stored in ``app.state.index_progress`` so that
    ``GET /api/index/status`` can report them to the browser.

    If an indexing run is already active, the request is ignored.
    """
    from takeout_rater.indexing.run import IndexProgress  # noqa: PLC0415

    # Ignore if already running
    existing: IndexProgress | None = getattr(app.state, "index_progress", None)
    if existing is not None and existing.running:
        return

    progress = IndexProgress(running=True)
    app.state.index_progress = progress

    def _worker() -> None:
        from takeout_rater.db.connection import open_library_db as _open  # noqa: PLC0415
        from takeout_rater.indexing.run import run_index  # noqa: PLC0415

        def _cb(p: object) -> None:
            app.state.index_progress = p  # type: ignore[union-attr]

        worker_conn = _open(library_root)
        try:
            result = run_index(library_root, worker_conn, on_progress=_cb)
        finally:
            worker_conn.close()
        # Replace the shared progress object once the run finishes.
        app.state.index_progress = result  # type: ignore[union-attr]

    thread = threading.Thread(target=_worker, daemon=True, name="takeout-rater-indexer")
    thread.start()


# ---------------------------------------------------------------------------
# Index status
# ---------------------------------------------------------------------------


@router.get("/api/index/status")
def index_status(request: Request) -> JSONResponse:
    """Return the current background indexing progress.

    Response fields
    ---------------
    running     bool      – ``true`` while the indexer thread is still active.
    done        bool      – ``true`` once the run has finished (success or error).
    error       str|null  – human-readable error message, or ``null`` on success.
    found       int       – total image files discovered during the scan.
    indexed     int       – assets upserted into the DB so far.
    thumbs_ok   int       – thumbnails generated successfully.
    thumbs_skip int       – thumbnails skipped (already existed or Pillow unavailable).
    phase       str       – ``"scanning"`` while scan_takeout runs; ``"indexing"`` once
                            the DB-upsert loop has started.
    total_dirs  int       – total directories to scan (0 until the first pass completes).
    dirs_scanned int      – directories fully processed so far.
    current_dir str       – name of the directory most recently processed.
    """
    from takeout_rater.indexing.run import IndexProgress  # noqa: PLC0415

    progress: IndexProgress | None = getattr(request.app.state, "index_progress", None)
    if progress is None:
        return JSONResponse(
            {
                "running": False,
                "done": False,
                "error": None,
                "found": 0,
                "indexed": 0,
                "thumbs_ok": 0,
                "thumbs_skip": 0,
                "phase": "scanning",
                "total_dirs": 0,
                "dirs_scanned": 0,
                "current_dir": "",
            }
        )
    return JSONResponse(
        {
            "running": progress.running,
            "done": progress.done,
            "error": progress.error,
            "found": progress.found,
            "indexed": progress.indexed,
            "thumbs_ok": progress.thumbs_ok,
            "thumbs_skip": progress.thumbs_skip,
            "phase": progress.phase,
            "total_dirs": progress.total_dirs,
            "dirs_scanned": progress.dirs_scanned,
            "current_dir": progress.current_dir,
        }
    )
