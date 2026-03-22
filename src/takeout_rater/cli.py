"""CLI entry-point for takeout-rater.

Run ``python -m takeout_rater --help`` or ``takeout-rater --help``
(after installing the package) to see available commands.

Sub-commands:
- ``index``   – scan a Takeout directory and build the library DB + thumbnail cache
- ``score``   – run scorer(s) over indexed assets  *(Iteration 2)*
- ``browse``  – launch the local web UI
- ``export``  – copy selected assets to an export folder  *(Iteration 3)*
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


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

    # index
    index_parser = sub.add_parser("index", help="Index a Takeout directory")
    index_parser.add_argument(
        "library_root",
        metavar="LIBRARY_ROOT",
        help=(
            "Directory that *contains* the Takeout/ folder. "
            "All library state is written to a sibling takeout-rater/ directory."
        ),
    )
    index_parser.add_argument(
        "--no-thumbs",
        action="store_true",
        default=False,
        help="Skip thumbnail generation (useful when Pillow is not installed).",
    )

    # browse
    browse_parser = sub.add_parser("browse", help="Launch the local web UI")
    browse_parser.add_argument(
        "library_root",
        metavar="LIBRARY_ROOT",
        help="Directory that *contains* the Takeout/ folder (same as used with 'index').",
    )
    browse_parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="TCP port to listen on (default: 8765).",
    )
    browse_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host/IP to bind to (default: 127.0.0.1).",
    )

    # Placeholders – implementation comes in later iterations
    sub.add_parser("score", help="(Iteration 2) Run scorer(s) over indexed assets")
    sub.add_parser("export", help="(Iteration 3) Copy selected assets to an export folder")

    return parser


def _cmd_index(args: argparse.Namespace) -> int:
    """Execute the ``index`` sub-command."""
    from takeout_rater.db.connection import library_state_dir, open_library_db  # noqa: PLC0415
    from takeout_rater.indexing.scanner import scan_takeout  # noqa: PLC0415
    from takeout_rater.indexing.sidecar import parse_sidecar  # noqa: PLC0415
    from takeout_rater.indexing.thumbnailer import (  # noqa: PLC0415
        generate_thumbnail,
        thumb_path_for_id,
    )

    library_root = Path(args.library_root).resolve()
    takeout_dir = library_root / "Takeout"

    if not library_root.exists():
        print(f"error: library root does not exist: {library_root}", file=sys.stderr)
        return 1
    if not takeout_dir.exists():
        # Accept the user passing the Takeout/ dir directly
        if (library_root / "Photos from").exists() or list(library_root.glob("Photos from *")):
            takeout_dir = library_root
            library_root = library_root.parent
        else:
            print(
                f"error: no Takeout/ directory found inside {library_root}\n"
                "       Pass the directory that *contains* your Takeout/ folder.",
                file=sys.stderr,
            )
            return 1

    print(f"Scanning {takeout_dir} …")
    assets = scan_takeout(takeout_dir)
    print(f"Found {len(assets)} image file(s).")

    if not assets:
        print("Nothing to index.")
        return 0

    conn = open_library_db(library_root)
    thumbs_dir = library_state_dir(library_root) / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    indexed = 0
    thumbs_ok = 0
    thumbs_skip = 0
    now = int(time.time())

    for asset_file in assets:
        # Parse sidecar (if available)
        sidecar = None
        if asset_file.sidecar_path is not None:
            try:
                sidecar = parse_sidecar(asset_file.sidecar_path)
            except ValueError as exc:
                print(f"  warning: {exc}", file=sys.stderr)

        row: dict = {
            "relpath": asset_file.relpath,
            "filename": Path(asset_file.relpath).name,
            "ext": Path(asset_file.relpath).suffix.lower(),
            "size_bytes": asset_file.size_bytes,
            "mime": asset_file.mime,
            "sidecar_relpath": (
                str(asset_file.sidecar_path.relative_to(takeout_dir))
                if asset_file.sidecar_path
                else None
            ),
            "indexed_at": now,
        }

        if sidecar is not None:
            row.update(
                {
                    "title": sidecar.title,
                    "description": sidecar.description,
                    "google_photos_url": sidecar.google_photos_url,
                    "taken_at": sidecar.taken_at,
                    "created_at_sidecar": sidecar.created_at_sidecar,
                    "image_views": sidecar.image_views,
                    "geo_lat": sidecar.geo_lat,
                    "geo_lon": sidecar.geo_lon,
                    "geo_alt": sidecar.geo_alt,
                    "geo_exif_lat": sidecar.geo_exif_lat,
                    "geo_exif_lon": sidecar.geo_exif_lon,
                    "geo_exif_alt": sidecar.geo_exif_alt,
                    "favorited": (
                        1 if sidecar.favorited else 0 if sidecar.favorited is not None else None
                    ),
                    "archived": (
                        1 if sidecar.archived else 0 if sidecar.archived is not None else None
                    ),
                    "trashed": (
                        1 if sidecar.trashed else 0 if sidecar.trashed is not None else None
                    ),
                    "origin_type": sidecar.origin_type,
                    "origin_device_type": sidecar.origin_device_type,
                    "origin_device_folder": sidecar.origin_device_folder,
                    "app_source_package": sidecar.app_source_package,
                }
            )

        from takeout_rater.db.queries import upsert_asset  # noqa: PLC0415

        asset_id = upsert_asset(conn, row)
        indexed += 1

        if not args.no_thumbs:
            thumb = thumb_path_for_id(thumbs_dir, asset_id)
            if not thumb.exists():
                try:
                    generate_thumbnail(asset_file.abspath, thumb)
                    thumbs_ok += 1
                except (ImportError, OSError) as exc:
                    print(
                        f"  thumb skipped ({Path(asset_file.relpath).name}): {exc}", file=sys.stderr
                    )
                    thumbs_skip += 1
            else:
                thumbs_skip += 1

    conn.close()

    print(f"Indexed {indexed} asset(s).")
    if not args.no_thumbs:
        print(f"Thumbnails: {thumbs_ok} generated, {thumbs_skip} skipped.")
    print(f"Library: {library_root / 'takeout-rater'}")
    return 0


def _cmd_browse(args: argparse.Namespace) -> int:
    """Execute the ``browse`` sub-command."""
    try:
        import uvicorn  # noqa: PLC0415
    except ImportError:
        print(
            "error: uvicorn is required for the browse command.\n"
            "       Install it with: pip install uvicorn",
            file=sys.stderr,
        )
        return 1

    library_root = Path(args.library_root).resolve()
    db_path = library_root / "takeout-rater" / "library.sqlite"

    if not db_path.exists():
        print(
            f"error: no library database found at {db_path}\n"
            "       Run 'takeout-rater index <library_root>' first.",
            file=sys.stderr,
        )
        return 1

    from takeout_rater.db.connection import open_library_db  # noqa: PLC0415
    from takeout_rater.ui.app import create_app  # noqa: PLC0415

    conn = open_library_db(library_root)
    app = create_app(library_root, conn)

    url = f"http://{args.host}:{args.port}/assets"
    print(f"Starting takeout-rater UI at {url}")
    print("Press Ctrl+C to stop.")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    conn.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "index":
        return _cmd_index(args)

    if args.command == "browse":
        return _cmd_browse(args)

    print(f"Command '{args.command}' is not yet implemented (see roadmap in README).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
