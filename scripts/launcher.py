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
   3a. Before installing the rest of the package, installs the correct
       CUDA-enabled torch wheel when an NVIDIA GPU is detected.
4. Prints GPU/CPU diagnostics so the user knows which device will be used.
5. Starts the ``takeout-rater serve`` server in a subprocess.
6. Waits for the server to become ready (polls ``/health``).
7. Opens the default browser to the UI.
8. Forwards the server's output to the terminal.
9. On Ctrl-C, shuts the server down cleanly.

The server itself handles the "Takeout path not configured" case — it shows
a browser-based setup page when the library path has not been set yet.
"""

from __future__ import annotations

import hashlib
import re
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


# Mapping of (major, minor) CUDA version to PyTorch wheel index suffix.
# Versions older than the oldest entry are not supported.
_CUDA_WHL_MAP: list[tuple[tuple[int, int], str]] = [
    ((13, 0), "cu130"),
    ((12, 8), "cu128"),
    ((12, 6), "cu126"),
    ((12, 4), "cu124"),
    ((12, 1), "cu121"),
]
_PYTORCH_WHL_BASE = "https://download.pytorch.org/whl"
_MIN_SUPPORTED_CUDA = (12, 1)


def _detect_cuda_version() -> tuple[int, int] | None:
    """Return the (major, minor) CUDA version reported by ``nvidia-smi``.

    Returns ``None`` if ``nvidia-smi`` is not found, fails, or reports no
    recognisable CUDA version string.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _cuda_to_whl_url(cuda_ver: tuple[int, int]) -> str | None:
    """Map a ``(major, minor)`` CUDA version to the matching wheel index URL.

    Returns ``None`` when *cuda_ver* is older than the oldest supported entry
    in ``_CUDA_WHL_MAP``.
    """
    if cuda_ver < _MIN_SUPPORTED_CUDA:
        return None
    # Walk the map from newest to oldest; return the first entry whose
    # minimum version is satisfied.
    for min_ver, suffix in _CUDA_WHL_MAP:
        if cuda_ver >= min_ver:
            return f"{_PYTORCH_WHL_BASE}/{suffix}"
    return None


def _install_torch() -> None:
    """Install a CUDA-enabled torch wheel when an NVIDIA GPU is available.

    This must be called *before* ``_install_deps()`` so that the CUDA wheel
    takes precedence over the CPU-only wheel that PyPI would supply via the
    ``torch>=2.2`` dependency in ``pyproject.toml``.

    If ``nvidia-smi`` is absent, fails, or reports an unsupported CUDA
    version, the function returns without doing anything and the CPU wheel
    installed by ``pip install -e .`` will be used instead.
    """
    cuda_ver = _detect_cuda_version()
    if cuda_ver is None:
        return

    whl_url = _cuda_to_whl_url(cuda_ver)
    if whl_url is None:
        print(
            f"  CUDA {cuda_ver[0]}.{cuda_ver[1]} detected but no compatible "
            "PyTorch wheel is available — falling back to CPU torch."
        )
        return

    # Check whether the currently installed torch already matches.
    venv_py = str(_venv_python())
    try:
        result = subprocess.run(
            [venv_py, "-c", "import torch; print(torch.version.cuda)"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        installed_cuda = result.stdout.strip()
    except Exception:
        installed_cuda = ""

    # e.g. whl_url ends in "cu121" and torch.version.cuda is "12.1"
    expected_suffix = whl_url.rsplit("/", 1)[-1]  # e.g. "cu121"
    # Normalise "12.1" → "cu121" for comparison
    if installed_cuda and installed_cuda != "None":
        normalised = "cu" + installed_cuda.replace(".", "")
        if normalised == expected_suffix:
            return  # already correct CUDA build

    cuda_str = f"{cuda_ver[0]}.{cuda_ver[1]}"
    print(f"  Installing PyTorch with CUDA {cuda_str} support ({whl_url}) …")
    _run(
        [
            venv_py,
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "--index-url",
            whl_url,
        ]
    )


def _print_gpu_diagnostics() -> None:
    """Print a one-line summary of the torch device that will be used."""
    venv_py = str(_venv_python())
    try:
        result = subprocess.run(
            [
                venv_py,
                "-c",
                (
                    "import torch; "
                    "cuda = torch.cuda.is_available(); "
                    "name = torch.cuda.get_device_name(0) if cuda else ''; "
                    "cv = torch.version.cuda or ''; "
                    "print(f'GPU: {name} (CUDA {cv})' if cuda else 'GPU: none — running on CPU')"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        line = result.stdout.strip()
        if line:
            print(f"  {line}")
    except Exception:
        pass


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
        _install_torch()
        _install_deps()
        print("Setup complete.\n")

    # ------------------------------------------------------------------
    # 3. GPU diagnostics
    # ------------------------------------------------------------------
    _print_gpu_diagnostics()

    # ------------------------------------------------------------------
    # 4. Launch server
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
    # 5. Wait for readiness, then open browser
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
    # 6. Keep running; forward Ctrl-C to the server
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
