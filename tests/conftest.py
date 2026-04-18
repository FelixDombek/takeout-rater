"""Pytest configuration for the local test suite."""

from __future__ import annotations

import os
from pathlib import Path

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
