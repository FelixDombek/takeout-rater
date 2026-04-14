"""FastAPI application factory for the takeout-rater local web UI."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
from starlette.exceptions import HTTPException as StarletteHTTPException

from takeout_rater.api.assets import router as assets_router
from takeout_rater.api.clip_routes import router as clip_router
from takeout_rater.api.clusters import router as clusters_router
from takeout_rater.api.config_routes import router as config_router
from takeout_rater.api.faces import router as faces_router
from takeout_rater.api.jobs import router as jobs_router
from takeout_rater.api.presets import router as presets_router
from takeout_rater.api.search import router as search_router

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _make_templates(templates_dir: Path) -> Environment:
    """Create a Jinja2 environment with custom filters."""
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html"]),
    )

    def timestamp_fmt(ts: int | None) -> str:
        """Format a Unix timestamp as a human-readable UTC string."""
        if ts is None:
            return "—"
        dt = datetime.fromtimestamp(ts, tz=UTC)
        return dt.strftime("%Y-%m-%d %H:%M UTC")

    def filesizeformat(size: int | None) -> str:
        """Format a file size in bytes as a human-readable string."""
        if size is None:
            return "—"
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.0f} {unit}"
            size //= 1024
        return f"{size:.0f} TB"

    env.filters["timestamp_fmt"] = timestamp_fmt
    env.filters["filesizeformat"] = filesizeformat

    # Expose a TemplateResponse helper that mimics Starlette's Jinja2Templates
    class _TemplateResponder:
        def __init__(self, jinja_env: Environment) -> None:
            self._env = jinja_env

        def TemplateResponse(self, template_name: str, context: dict) -> HTMLResponse:  # noqa: N802
            tmpl = self._env.get_template(template_name)
            return HTMLResponse(tmpl.render(**context))

    return _TemplateResponder(env)  # type: ignore[return-value]


def create_app(
    library_root: Path | None,
    db_conn: sqlite3.Connection | None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        library_root: The directory containing the ``Takeout/`` folder, or
            *None* when the library path has not been configured yet.
        db_conn: An open SQLite connection to the library database, or *None*
            when the library is not available yet.

    Returns:
        A configured :class:`FastAPI` application instance.
    """
    app = FastAPI(
        title="takeout-rater",
        description="Local photo library browser for Google Photos Takeout",
        version="0.1.0",
        docs_url=None,  # disable Swagger UI in production use
        redoc_url=None,
    )

    # Attach shared state (may be None when not yet configured)
    app.state.db_conn = db_conn
    app.state.library_root = library_root
    # Path to the SQLite database file — used by per-request connections to
    # avoid sharing a single sqlite3.Connection across threads.
    if library_root is not None:
        from takeout_rater.db.connection import library_db_path  # noqa: PLC0415

        _candidate = library_db_path(library_root)
        app.state.db_path = _candidate if _candidate.exists() else None
    else:
        app.state.db_path = None
    # takeout_root is the photos root (the directory that relpath/sidecar_relpath
    # values are relative to — may be library_root/Takeout/Google Photos/ etc.).
    if library_root is not None:
        from takeout_rater.indexing.scanner import resolve_photos_root  # noqa: PLC0415

        app.state.takeout_root = resolve_photos_root(library_root)
    else:
        app.state.takeout_root = None
    app.state.thumbs_dir = library_root / "takeout-rater" / "thumbs" if library_root else None
    app.state.templates = _make_templates(_TEMPLATES_DIR)
    # Mount static assets (CSS, JS shared across pages)
    _STATIC_DIR = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
    # Background indexing state (set/updated by the config route)
    app.state.index_progress = None
    # Background job state (set/updated by the jobs router)
    app.state.jobs = {}

    # Config / health routes always available (no DB required)
    app.include_router(config_router)

    # Asset browsing routes (require a DB connection)
    app.include_router(assets_router)
    app.include_router(clusters_router)
    app.include_router(presets_router)
    app.include_router(jobs_router)
    app.include_router(search_router)
    app.include_router(clip_router)
    app.include_router(faces_router)

    @app.get("/")
    def redirect_to_browse(request: Request) -> RedirectResponse:
        if request.app.state.db_conn is None:
            return RedirectResponse(url="/setup")
        return RedirectResponse(url="/assets")

    @app.get("/setup", response_class=HTMLResponse)
    def setup_page(request: Request) -> HTMLResponse:
        from takeout_rater.config import get_takeout_path  # noqa: PLC0415

        current = get_takeout_path()
        templates = request.app.state.templates
        return templates.TemplateResponse(
            "setup.html",
            {"request": request, "current_path": str(current) if current else None},
        )

    @app.get("/jobs", response_class=HTMLResponse)
    def jobs_page(request: Request) -> HTMLResponse:
        if request.app.state.db_conn is None:
            from fastapi.responses import RedirectResponse as _RR  # noqa: PLC0415

            return _RR(url="/setup")  # type: ignore[return-value]
        templates = request.app.state.templates
        return templates.TemplateResponse("jobs.html", {"request": request})

    @app.get("/scoring", response_class=HTMLResponse)
    def scoring_page(request: Request) -> HTMLResponse:
        if request.app.state.db_conn is None:
            from fastapi.responses import RedirectResponse as _RR  # noqa: PLC0415

            return _RR(url="/setup")  # type: ignore[return-value]
        templates = request.app.state.templates
        return templates.TemplateResponse("scoring.html", {"request": request})

    @app.get("/search", response_class=HTMLResponse)
    def search_page(request: Request) -> HTMLResponse:
        if request.app.state.db_conn is None:
            from fastapi.responses import RedirectResponse as _RR  # noqa: PLC0415

            return _RR(url="/setup")  # type: ignore[return-value]
        templates = request.app.state.templates
        return templates.TemplateResponse("search.html", {"request": request})

    @app.get("/clip", response_class=HTMLResponse)
    def clip_page(request: Request) -> HTMLResponse:
        if request.app.state.db_conn is None:
            from fastapi.responses import RedirectResponse as _RR  # noqa: PLC0415

            return _RR(url="/setup")  # type: ignore[return-value]
        templates = request.app.state.templates
        return templates.TemplateResponse("clip.html", {"request": request})

    @app.get("/faces", response_class=HTMLResponse)
    def faces_page(request: Request) -> HTMLResponse:
        if request.app.state.db_conn is None:
            from fastapi.responses import RedirectResponse as _RR  # noqa: PLC0415

            return _RR(url="/setup")  # type: ignore[return-value]
        templates = request.app.state.templates
        return templates.TemplateResponse("faces.html", {"request": request})

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse | RedirectResponse:
        """Redirect browser navigation to /setup for 503 (library not configured).

        When an HTML-accepting client (e.g. a browser tab) navigates to a page
        route that requires a configured library, return a redirect to /setup
        rather than exposing raw JSON.  All other HTTP exceptions and all
        non-HTML clients continue to receive a JSON error response.
        """
        if exc.status_code == 503 and "text/html" in request.headers.get("accept", ""):
            return RedirectResponse(url="/setup")
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Return a compact JSON error instead of FastAPI's default verbose 422 body.

        This prevents raw validation error detail from being rendered directly
        in the browser when query parameters fail type coercion.
        """
        detail = "; ".join(
            f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        return JSONResponse({"detail": detail}, status_code=422)

    return app
