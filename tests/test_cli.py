"""Tests for the CLI entry-point."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from takeout_rater.cli import _EXIT_SCHEMA_MISMATCH, build_parser, main
from takeout_rater.db.schema import SchemaMismatchError


def test_help_exits_cleanly() -> None:
    parser = build_parser()
    # --help would raise SystemExit(0); just test that build succeeds
    assert parser is not None


def test_no_command_returns_zero() -> None:
    assert main([]) == 0


def test_score_subcommand_is_registered() -> None:
    """The score sub-command must be listed in the parser."""
    parser = build_parser()
    args = parser.parse_args(["score", "/tmp/fake"])
    assert args.command == "score"


def test_score_subcommand_default_batch_size() -> None:
    parser = build_parser()
    args = parser.parse_args(["score", "/tmp/fake"])
    assert args.batch_size == 32


def test_export_subcommand_is_registered() -> None:
    """The export sub-command must be listed in the parser."""
    parser = build_parser()
    args = parser.parse_args(["export", "/tmp/fake"])
    assert args.command == "export"


def test_cluster_subcommand_is_registered() -> None:
    """The cluster sub-command must be listed in the parser."""
    parser = build_parser()
    args = parser.parse_args(["cluster", "/tmp/fake"])
    assert args.command == "cluster"


def test_cluster_subcommand_default_threshold() -> None:
    parser = build_parser()
    args = parser.parse_args(["cluster", "/tmp/fake"])
    assert args.threshold == 10


def test_cluster_subcommand_default_window() -> None:
    parser = build_parser()
    args = parser.parse_args(["cluster", "/tmp/fake"])
    assert args.window == 200


def test_cluster_subcommand_custom_params() -> None:
    parser = build_parser()
    args = parser.parse_args(["cluster", "--threshold", "5", "--window", "50", "/tmp/fake"])
    assert args.threshold == 5
    assert args.window == 50


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


# ---------------------------------------------------------------------------
# serve sub-command: schema mismatch handling
# ---------------------------------------------------------------------------


def test_serve_schema_mismatch_returns_exit_code(tmp_path: Path) -> None:
    """_cmd_serve must return _EXIT_SCHEMA_MISMATCH when the DB has a stale schema."""
    db_path = tmp_path / "takeout-rater" / "library.sqlite"
    db_path.parent.mkdir(parents=True)
    db_path.touch()  # file must exist so the existence check passes

    with (
        patch("takeout_rater.config.get_takeout_path", return_value=tmp_path),
        patch(
            "takeout_rater.db.connection.open_library_db",
            side_effect=SchemaMismatchError(5),
        ),
    ):
        result = main(["serve"])

    assert result == _EXIT_SCHEMA_MISMATCH


def test_serve_schema_mismatch_does_not_start_server(tmp_path: Path) -> None:
    """_cmd_serve must not call uvicorn.run when schema mismatch is detected."""
    db_path = tmp_path / "takeout-rater" / "library.sqlite"
    db_path.parent.mkdir(parents=True)
    db_path.touch()

    mock_uvicorn = MagicMock()

    with (
        patch("takeout_rater.config.get_takeout_path", return_value=tmp_path),
        patch(
            "takeout_rater.db.connection.open_library_db",
            side_effect=SchemaMismatchError(3),
        ),
        patch.dict("sys.modules", {"uvicorn": mock_uvicorn}),
    ):
        main(["serve"])

    mock_uvicorn.run.assert_not_called()
