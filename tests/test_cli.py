"""Tests for the CLI entry-point."""

from __future__ import annotations

from takeout_rater.cli import build_parser, main


def test_help_exits_cleanly() -> None:
    parser = build_parser()
    # --help would raise SystemExit(0); just test that build succeeds
    assert parser is not None


def test_no_command_returns_zero() -> None:
    assert main([]) == 0


def test_unimplemented_command_score_returns_nonzero() -> None:
    assert main(["score"]) == 1


def test_unimplemented_command_export_returns_nonzero() -> None:
    assert main(["export"]) == 1


def test_index_subcommand_is_registered() -> None:
    """The index sub-command must be listed in the parser."""
    parser = build_parser()
    # Parse with the index sub-command and a dummy path to confirm it's registered
    args = parser.parse_args(["index", "/tmp/fake"])
    assert args.command == "index"


def test_browse_subcommand_is_registered() -> None:
    """The browse sub-command must be listed in the parser."""
    parser = build_parser()
    args = parser.parse_args(["browse", "/tmp/fake"])
    assert args.command == "browse"


def test_browse_default_port() -> None:
    parser = build_parser()
    args = parser.parse_args(["browse", "/tmp/fake"])
    assert args.port == 8765


def test_index_no_thumbs_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["index", "--no-thumbs", "/tmp/fake"])
    assert args.no_thumbs is True
