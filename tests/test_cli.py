"""Tests for the CLI entry-point."""

from __future__ import annotations

from takeout_rater.cli import build_parser, main


def test_help_exits_cleanly() -> None:
    parser = build_parser()
    # --help would raise SystemExit(0); just test that build succeeds
    assert parser is not None


def test_no_command_returns_zero() -> None:
    assert main([]) == 0


def test_unimplemented_command_returns_nonzero() -> None:
    assert main(["index"]) == 1
