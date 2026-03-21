"""CLI entry-point for takeout-rater.

Run ``python -m takeout_rater --help`` or ``takeout-rater --help``
(after installing the package) to see available commands.

Sub-commands will be added in later iterations:
- ``index``   – scan a Takeout directory and build the library DB
- ``score``   – run scorer(s) over indexed assets
- ``browse``  – launch the local web UI
- ``export``  – copy selected assets to an export folder
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="takeout-rater",
        description=(
            "Aesthetics scoring orchestrator for Google Photos Takeout folders.\n\n"
            "Point it at the directory that *contains* your Takeout/ folder and it\n"
            "will build a sibling takeout-rater/ directory with the library DB,\n"
            "thumbnail cache, and exports."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # Placeholders – implementation comes in later iterations
    sub.add_parser("index", help="(Iteration 1) Index a Takeout directory")
    sub.add_parser("score", help="(Iteration 2) Run scorer(s) over indexed assets")
    sub.add_parser("browse", help="(Iteration 1) Launch the local web UI")
    sub.add_parser("export", help="(Iteration 3) Copy selected assets to an export folder")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    print(f"Command '{args.command}' is not yet implemented (see roadmap in README).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
