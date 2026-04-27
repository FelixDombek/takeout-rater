"""Pytest configuration for the local test suite."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import pytest

from takeout_rater.scoring.scorers import clip_backbone

_TEST_TOTAL_DURATIONS: defaultdict[str, float] = defaultdict(float)
_TEST_RUNTIME_REPORT_THRESHOLD_SECONDS = 1.0
_PYTEST_CONFIG: pytest.Config | None = None

# Pytest's --basetemp creates the final directory but not missing parents.
# Keep the shared parent present so the configured .pytest-tmp/default works
# on fresh CI checkouts while concurrent runs can still use .pytest-tmp/<name>.
Path(".pytest-tmp").mkdir(exist_ok=True)


@pytest.fixture(autouse=True)
def _reset_clip_backbone_singleton():
    """Keep shared CLIP model state from leaking across tests."""
    with clip_backbone._lock:  # noqa: SLF001
        clip_backbone._clip_model = None  # noqa: SLF001
        clip_backbone._preprocess = None  # noqa: SLF001
        clip_backbone._tokenizer = None  # noqa: SLF001
        clip_backbone._device = None  # noqa: SLF001

    yield

    with clip_backbone._lock:  # noqa: SLF001
        clip_backbone._clip_model = None  # noqa: SLF001
        clip_backbone._preprocess = None  # noqa: SLF001
        clip_backbone._tokenizer = None  # noqa: SLF001
        clip_backbone._device = None  # noqa: SLF001


if os.name == "nt":
    _ORIGINAL_MKDIR = Path.mkdir

    def _mkdir_sandbox_safe(
        self: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        """Avoid unreadable pytest temp dirs in the Windows sandbox.

        Python 3.14 applies restrictive ACLs for ``mode=0o700`` on Windows.
        The Codex sandbox can create those directories but cannot read them
        back, which breaks pytest's ``tmp_path`` fixture.  Pytest uses 0o700
        for its own temporary roots, so relax only that mode during tests.
        """
        if mode == 0o700:
            mode = 0o777
        return _ORIGINAL_MKDIR(self, mode=mode, parents=parents, exist_ok=exist_ok)

    Path.mkdir = _mkdir_sandbox_safe  # type: ignore[method-assign]


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Print total runtime for test cases that exceed the reporting threshold."""
    _TEST_TOTAL_DURATIONS[report.nodeid] += report.duration
    if report.when != "teardown":
        return

    total_duration = _TEST_TOTAL_DURATIONS.pop(report.nodeid)
    if total_duration <= _TEST_RUNTIME_REPORT_THRESHOLD_SECONDS:
        return

    if _PYTEST_CONFIG is None:
        return
    terminal_reporter = _PYTEST_CONFIG.pluginmanager.get_plugin("terminalreporter")
    if terminal_reporter is not None:
        terminal_reporter.write_line(
            f"runtime {report.nodeid}: {total_duration:.4f}s",
        )


def pytest_configure(config: pytest.Config) -> None:
    """Capture pytest config so report hooks can access terminal reporter."""
    global _PYTEST_CONFIG  # noqa: PLW0603
    _PYTEST_CONFIG = config
