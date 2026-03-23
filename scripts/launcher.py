#!/usr/bin/env python3
"""takeout-rater one-step launcher.

Usage (after ``git clone``):

    ./run          # macOS / Linux
    run.bat        # Windows

What this script does
---------------------
1. Validates the Python version (3.12+).
2. Creates a local virtual environment in ``.venv/`` if it doesn't exist.
3. Installs/updates dependencies the first time, or when ``pyproject.toml``
   has changed (detected via a SHA-256 hash stored in ``.venv/.deps_hash``).
4. Starts the ``takeout-rater serve`` server in a subprocess.
5. Waits for the server to become ready (polls ``/health``).
6. Opens the default browser to the UI.
7. Forwards the server's output to the terminal.
8. On Ctrl-C, shuts the server down cleanly.

The server itself handles the "Takeout path not configured" case — it shows
a browser-based setup page when the library path has not been set yet.
"""

from __future__ import annotations

import hashlib
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_PYTHON = (3, 12)
ROOT = Path(__file__).resolve().parent.parent  # repository root
VENV = ROOT / ".venv"
DEPS_HASH_FILE = VENV / ".deps_hash"
PORT = 8765
HOST = "127.0.0.1"
HEALTH_URL = f"http://{HOST}:{PORT}/health"
READY_TIMEOUT = 60  # seconds to wait for the server to start

# Dependency definition files whose content determines whether a reinstall is
# needed.  Add ``requirements*.txt`` here if you introduce them in future.
_DEP_FILES = ["pyproject.toml"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _python_ok() -> bool:
    return sys.version_info >= MIN_PYTHON


def _venv_python() -> Path:
    if sys.platform == "win32":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def _compute_deps_hash() -> str:
    h = hashlib.sha256()
    for name in _DEP_FILES:
        f = ROOT / name
        if f.exists():
            h.update(f.read_bytes())
    return h.hexdigest()


def _needs_install() -> bool:
    venv_py = _venv_python()
    if not venv_py.exists():
        return True
    if not DEPS_HASH_FILE.exists():
        return True
    return DEPS_HASH_FILE.read_text().strip() != _compute_deps_hash()


def _run(cmd: list[str], **kwargs) -> None:  # type: ignore[type-arg]
    """Run a command, raising on non-zero exit."""
    subprocess.run(cmd, check=True, **kwargs)


def _create_venv() -> None:
    print("  Creating virtual environment …")
    _run([sys.executable, "-m", "venv", str(VENV)])


def _install_deps() -> None:
    venv_py = str(_venv_python())
    print("  Upgrading pip …")
    _run([venv_py, "-m", "pip", "install", "--upgrade", "pip"])
    print("  Installing takeout-rater …")
    _run([venv_py, "-m", "pip", "install", "-e", str(ROOT)])
    DEPS_HASH_FILE.write_text(_compute_deps_hash())


def _wait_for_server(timeout: int = READY_TIMEOUT) -> bool:
    """Poll the health endpoint until it responds or *timeout* seconds pass."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=1):
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.4)
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Python version check
    # ------------------------------------------------------------------
    if not _python_ok():
        v = ".".join(str(x) for x in MIN_PYTHON)
        print(
            f"Error: Python {v}+ is required, but you are running "
            f"{sys.version.split()[0]}.\n"
            "Download the latest Python from https://www.python.org/downloads/",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Virtual environment + dependency install
    # ------------------------------------------------------------------
    if _needs_install():
        print("Setting up environment (first run or dependencies changed) …")
        if not _venv_python().exists():
            _create_venv()
        _install_deps()
        print("Setup complete.\n")

    # ------------------------------------------------------------------
    # 3. Launch server
    # ------------------------------------------------------------------
    venv_py = str(_venv_python())
    cmd = [
        venv_py,
        "-m",
        "takeout_rater",
        "serve",
        "--host",
        HOST,
        "--port",
        str(PORT),
    ]
    print(f"Starting server on http://{HOST}:{PORT}/ …")
    proc = subprocess.Popen(cmd, cwd=str(ROOT))

    # ------------------------------------------------------------------
    # 4. Wait for readiness, then open browser
    # ------------------------------------------------------------------
    if _wait_for_server():
        webbrowser.open(f"http://{HOST}:{PORT}/")
    else:
        print(
            f"Warning: server did not respond at {HEALTH_URL} within "
            f"{READY_TIMEOUT}s.  Opening browser anyway …"
        )
        webbrowser.open(f"http://{HOST}:{PORT}/")

    # ------------------------------------------------------------------
    # 5. Keep running; forward Ctrl-C to the server
    # ------------------------------------------------------------------
    def _shutdown(_signum: int, _frame: object) -> None:
        print("\nShutting down …")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)

    proc.wait()


if __name__ == "__main__":
    main()
