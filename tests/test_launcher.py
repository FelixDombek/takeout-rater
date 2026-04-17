"""Tests for CUDA-detection and wheel-URL helpers in scripts/launcher.py."""

from __future__ import annotations

import importlib.util
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Import the launcher module from scripts/ (not a package, so use spec)
# ---------------------------------------------------------------------------

_LAUNCHER_PATH = Path(__file__).resolve().parent.parent / "scripts" / "launcher.py"
_spec = importlib.util.spec_from_file_location("launcher", _LAUNCHER_PATH)
assert _spec is not None and _spec.loader is not None
launcher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launcher)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# _detect_cuda_version
# ---------------------------------------------------------------------------


def _make_smi_result(stdout: str, returncode: int = 0) -> MagicMock:
    r = MagicMock()
    r.returncode = returncode
    r.stdout = stdout
    return r


def test_detect_cuda_version_parses_output() -> None:
    smi_output = (
        "+-----------------------------------------------------------------------------+\n"
        "| NVIDIA-SMI 535.86.10    Driver Version: 535.86.10    CUDA Version: 12.2     |\n"
        "+-----------------------------------------------------------------------------+\n"
    )
    with patch("subprocess.run", return_value=_make_smi_result(smi_output)):
        assert launcher._detect_cuda_version() == (12, 2)


def test_detect_cuda_version_not_found() -> None:
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert launcher._detect_cuda_version() is None


def test_detect_cuda_version_nonzero_returncode() -> None:
    with patch("subprocess.run", return_value=_make_smi_result("", returncode=1)):
        assert launcher._detect_cuda_version() is None


def test_detect_cuda_version_no_match() -> None:
    with patch("subprocess.run", return_value=_make_smi_result("no version here")):
        assert launcher._detect_cuda_version() is None


def test_detect_cuda_version_cuda13() -> None:
    smi_output = "CUDA Version: 13.0"
    with patch("subprocess.run", return_value=_make_smi_result(smi_output)):
        assert launcher._detect_cuda_version() == (13, 0)


# ---------------------------------------------------------------------------
# _cuda_to_whl_url
# ---------------------------------------------------------------------------


def test_cuda_to_whl_url_exact_121() -> None:
    assert launcher._cuda_to_whl_url((12, 1)) == "https://download.pytorch.org/whl/cu121"


def test_cuda_to_whl_url_exact_124() -> None:
    assert launcher._cuda_to_whl_url((12, 4)) == "https://download.pytorch.org/whl/cu124"


def test_cuda_to_whl_url_exact_126() -> None:
    assert launcher._cuda_to_whl_url((12, 6)) == "https://download.pytorch.org/whl/cu126"


def test_cuda_to_whl_url_exact_128() -> None:
    assert launcher._cuda_to_whl_url((12, 8)) == "https://download.pytorch.org/whl/cu128"


def test_cuda_to_whl_url_exact_130() -> None:
    assert launcher._cuda_to_whl_url((13, 0)) == "https://download.pytorch.org/whl/cu130"


def test_cuda_to_whl_url_between_versions_uses_lower() -> None:
    # CUDA 12.3 → no exact match; falls through to cu121 (first entry >= 12.1)
    url = launcher._cuda_to_whl_url((12, 3))
    assert url == "https://download.pytorch.org/whl/cu121"


def test_cuda_to_whl_url_between_124_and_126() -> None:
    # CUDA 12.5 → no exact match; falls through to cu124
    url = launcher._cuda_to_whl_url((12, 5))
    assert url == "https://download.pytorch.org/whl/cu124"


def test_cuda_to_whl_url_future_version() -> None:
    # A hypothetical CUDA 14.0 should map to the newest known entry (cu130)
    url = launcher._cuda_to_whl_url((14, 0))
    assert url == "https://download.pytorch.org/whl/cu130"


def test_cuda_to_whl_url_too_old() -> None:
    assert launcher._cuda_to_whl_url((11, 8)) is None


def test_cuda_to_whl_url_exactly_at_minimum() -> None:
    assert launcher._cuda_to_whl_url((12, 1)) is not None


def test_cuda_to_whl_url_just_below_minimum() -> None:
    assert launcher._cuda_to_whl_url((12, 0)) is None


# ---------------------------------------------------------------------------
# _get_db_path
# ---------------------------------------------------------------------------


def test_get_db_path_returns_path_when_config_exists(tmp_path: Path) -> None:
    library_root = tmp_path / "my_photos"
    config = tmp_path / ".takeout-rater.json"
    config.write_text(f'{{"photos_root": "{library_root}"}}', encoding="utf-8")

    with patch.object(launcher, "ROOT", tmp_path):
        result = launcher._get_db_path()

    assert result == library_root / "takeout-rater" / "library.sqlite"


def test_get_db_path_legacy_takeout_path(tmp_path: Path) -> None:
    """Old config key takeout_path is still supported."""
    library_root = tmp_path / "my_photos"
    config = tmp_path / ".takeout-rater.json"
    config.write_text(f'{{"takeout_path": "{library_root}"}}', encoding="utf-8")

    with patch.object(launcher, "ROOT", tmp_path):
        result = launcher._get_db_path()

    assert result == library_root / "takeout-rater" / "library.sqlite"


def test_get_db_path_uses_db_root_when_present(tmp_path: Path) -> None:
    """db_root overrides photos_root for DB location."""
    photos = tmp_path / "photos"
    db_root = tmp_path / "state"
    config = tmp_path / ".takeout-rater.json"
    config.write_text(f'{{"photos_root": "{photos}", "db_root": "{db_root}"}}', encoding="utf-8")

    with patch.object(launcher, "ROOT", tmp_path):
        result = launcher._get_db_path()

    assert result == db_root / "takeout-rater" / "library.sqlite"


def test_get_db_path_returns_none_when_config_absent(tmp_path: Path) -> None:
    with patch.object(launcher, "ROOT", tmp_path):
        assert launcher._get_db_path() is None


def test_get_db_path_returns_none_when_config_has_no_takeout_path(
    tmp_path: Path,
) -> None:
    config = tmp_path / ".takeout-rater.json"
    config.write_text("{}", encoding="utf-8")

    with patch.object(launcher, "ROOT", tmp_path):
        assert launcher._get_db_path() is None


def test_get_db_path_returns_none_on_malformed_json(tmp_path: Path) -> None:
    config = tmp_path / ".takeout-rater.json"
    config.write_text("not-valid-json", encoding="utf-8")

    with patch.object(launcher, "ROOT", tmp_path):
        assert launcher._get_db_path() is None


# ---------------------------------------------------------------------------
# _prompt_and_delete_db
# ---------------------------------------------------------------------------


def test_prompt_and_delete_db_yes_deletes_file(tmp_path: Path) -> None:
    db = tmp_path / "library.sqlite"
    db.write_bytes(b"")

    with patch("builtins.input", return_value="y"):
        result = launcher._prompt_and_delete_db(db)

    assert result is True
    assert not db.exists()


def test_prompt_and_delete_db_yes_removes_wal_shm(tmp_path: Path) -> None:
    db = tmp_path / "library.sqlite"
    wal = tmp_path / "library.sqlite-wal"
    shm = tmp_path / "library.sqlite-shm"
    for f in (db, wal, shm):
        f.write_bytes(b"")

    with patch("builtins.input", return_value="yes"):
        launcher._prompt_and_delete_db(db)

    assert not wal.exists()
    assert not shm.exists()


def test_prompt_and_delete_db_no_keeps_file(tmp_path: Path) -> None:
    db = tmp_path / "library.sqlite"
    db.write_bytes(b"")

    with patch("builtins.input", return_value="n"):
        result = launcher._prompt_and_delete_db(db)

    assert result is False
    assert db.exists()


def test_prompt_and_delete_db_empty_answer_keeps_file(tmp_path: Path) -> None:
    db = tmp_path / "library.sqlite"
    db.write_bytes(b"")

    with patch("builtins.input", return_value=""):
        result = launcher._prompt_and_delete_db(db)

    assert result is False
    assert db.exists()


def test_prompt_and_delete_db_eof_returns_false(tmp_path: Path) -> None:
    db = tmp_path / "library.sqlite"
    db.write_bytes(b"")

    with patch("builtins.input", side_effect=EOFError):
        result = launcher._prompt_and_delete_db(db)

    assert result is False


# ---------------------------------------------------------------------------
# _wait_for_server — dead-process bail-out
# ---------------------------------------------------------------------------


def test_wait_for_server_returns_false_when_process_already_dead() -> None:
    dead_proc = MagicMock()
    dead_proc.poll.return_value = 3  # process exited

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
        result = launcher._wait_for_server(dead_proc, timeout=5)

    assert result is False
