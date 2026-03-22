"""FastAPI application factory for the takeout-rater local web UI."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape

from takeout_rater.api.assets import router as assets_router
from takeout_rater.api.clusters import router as clusters_router

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
    library_root: Path,
    db_conn: sqlite3.Connection,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        library_root: The directory containing the ``Takeout/`` folder.
        db_conn: An open SQLite connection to the library database.

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

    # Attach shared state
    app.state.db_conn = db_conn
    app.state.takeout_root = library_root
    app.state.thumbs_dir = library_root / "takeout-rater" / "thumbs"
    app.state.templates = _make_templates(_TEMPLATES_DIR)

    app.include_router(assets_router)
    app.include_router(clusters_router)

    @app.get("/")
    def redirect_to_browse() -> RedirectResponse:
        return RedirectResponse(url="/assets")

    return app
