"""CLI entry-point for takeout-rater.

Run ``python -m takeout_rater --help`` or ``takeout-rater --help``
(after installing the package) to see available commands.

Sub-commands:
- ``index``   – scan a Takeout directory and build the library DB + thumbnail cache
- ``score``   – run scorer(s) over indexed assets
- ``browse``  – launch the local web UI
- ``export``  – copy selected assets to an export folder  *(Iteration 3)*
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

# Exit code returned by `serve` when the on-disk database was built by an
# incompatible schema version.  The launcher script checks this code and
# offers to delete the stale database before restarting.
# NOTE: keep in sync with _EXIT_SCHEMA_MISMATCH in scripts/launcher.py.
_EXIT_SCHEMA_MISMATCH: int = 3


def _find_free_port(host: str, start_port: int, max_tries: int = 16) -> int:
    """Return the first free TCP port starting from *start_port*.

    Tries up to *max_tries* consecutive ports and raises :exc:`OSError` if
    none are available.
    """
    for offset in range(max_tries):
        port = start_port + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise OSError(
        f"Could not find a free port in the range "
        f"{start_port}–{start_port + max_tries - 1} on {host!r}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="takeout-rater",
        description=(
            "Aesthetics scoring orchestrator for photo folders.\n\n"
            "Point it at the directory that directly contains your album sub-folders\n"
            "(e.g. the 'Google Photos' directory from a Takeout export, or any\n"
            "arbitrary folder of albums) and it will build a 'takeout-rater/' state\n"
            "directory with the library DB, thumbnail cache, and exports.\n\n"
            "Use --db-root to store the state directory outside the photo folder."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    def _add_db_root(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--db-root",
            metavar="DIR",
            default=None,
            help=(
                "Directory where the takeout-rater/ state folder (DB, thumbnails) is stored. "
                "Defaults to PHOTOS_ROOT when not given."
            ),
        )

    # index
    index_parser = sub.add_parser("index", help="Index a photos directory")
    index_parser.add_argument(
        "photos_root",
        metavar="PHOTOS_ROOT",
        help=(
            "Directory that directly contains the album sub-folders. "
            "All library state is written to a 'takeout-rater/' sub-directory "
            "(or to --db-root if given)."
        ),
    )
    _add_db_root(index_parser)

    # browse
    browse_parser = sub.add_parser("browse", help="Launch the local web UI")
    browse_parser.add_argument(
        "photos_root",
        metavar="PHOTOS_ROOT",
        help="Directory that directly contains the album sub-folders (same as used with 'index').",
    )
    _add_db_root(browse_parser)
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

    # score
    score_parser = sub.add_parser("score", help="Run scorer(s) over indexed assets")
    score_parser.add_argument(
        "photos_root",
        metavar="PHOTOS_ROOT",
        help="Directory that directly contains the album sub-folders (same as used with 'index').",
    )
    _add_db_root(score_parser)
    score_parser.add_argument(
        "--scorer",
        metavar="SCORER_ID",
        default=None,
        help=(
            "ID of the scorer to run (e.g. 'blur').  "
            "If not specified, all available scorers are run."
        ),
    )
    score_parser.add_argument(
        "--phash",
        action="store_true",
        default=False,
        help="Compute perceptual hashes (dhash) for all assets and store in the phash table.",
    )
    score_parser.add_argument(
        "--rerun",
        action="store_true",
        default=False,
        help="Re-score even if scores already exist (overwrites previous scores).",
    )
    score_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="Number of images per scoring batch (default: 32).",
    )

    # cluster
    cluster_parser = sub.add_parser(
        "cluster", help="Group near-duplicate photos by perceptual hash"
    )
    cluster_parser.add_argument(
        "photos_root",
        metavar="PHOTOS_ROOT",
        help="Directory that directly contains the album sub-folders (same as used with 'index').",
    )
    _add_db_root(cluster_parser)
    cluster_parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        metavar="T",
        help=(
            "Maximum Hamming distance (out of 64 bits) to consider two images near-duplicates"
            " (default: 10)."
        ),
    )
    cluster_parser.add_argument(
        "--window",
        type=int,
        default=200,
        metavar="W",
        help=(
            "Sliding-window size over sorted hashes; larger values find more pairs at higher"
            " CPU cost (default: 200)."
        ),
    )
    cluster_parser.add_argument(
        "--min-size",
        type=int,
        default=2,
        metavar="N",
        help="Minimum cluster size to store (default: 2; singletons are discarded).",
    )

    # export
    export_parser = sub.add_parser("export", help="Copy best-of-cluster photos to an export folder")
    export_parser.add_argument(
        "photos_root",
        metavar="PHOTOS_ROOT",
        help="Directory that directly contains the album sub-folders.",
    )
    _add_db_root(export_parser)
    export_parser.add_argument(
        "--scorer",
        metavar="SCORER_ID",
        default=None,
        help=(
            "Scorer ID to rank cluster members by (e.g. 'blur').  "
            "The representative with the highest score is exported."
            "  If omitted, the cluster representative (lowest asset ID) is used."
        ),
    )
    export_parser.add_argument(
        "--metric",
        metavar="METRIC_KEY",
        default=None,
        help="Metric key to use for ranking (required when --scorer is given).",
    )
    export_parser.add_argument(
        "--out",
        metavar="DIR",
        default=None,
        help=(
            "Destination directory for exported files.  "
            "Defaults to <db-root>/takeout-rater/exports/."
        ),
    )

    # serve – launcher-friendly variant of browse that reads config for the path
    serve_parser = sub.add_parser(
        "serve",
        help=(
            "Launch the web UI (reads photos path from config; "
            "shows setup page when not yet configured)"
        ),
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="TCP port to listen on (default: 8765).",
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host/IP to bind to (default: 127.0.0.1).",
    )

    return parser


def _cmd_index(args: argparse.Namespace) -> int:
    """Execute the ``index`` sub-command."""
    from takeout_rater.db.connection import open_library_db  # noqa: PLC0415
    from takeout_rater.indexing.run import IndexProgress, run_index  # noqa: PLC0415

    photos_root = Path(args.photos_root).resolve()
    if not photos_root.exists():
        print(f"error: photos root does not exist: {photos_root}", file=sys.stderr)
        return 1

    db_root = Path(args.db_root).resolve() if args.db_root else photos_root

    last_phase: list[str] = ["scanning"]

    def _on_progress(p: IndexProgress) -> None:
        if p.phase != last_phase[0]:
            last_phase[0] = p.phase
            print()  # newline before each new phase
        if p.phase == "scanning" and p.total_dirs > 0:
            print(
                f"\rScanning… {p.dirs_scanned}/{p.total_dirs} dirs",
                end="",
                flush=True,
            )
        elif p.phase == "processing" and p.found > 0:
            print(
                f"\rProcessing… {p.indexed}/{p.found} assets",
                end="",
                flush=True,
            )

    conn = open_library_db(db_root)
    try:
        result = run_index(photos_root, conn, db_root=db_root, on_progress=_on_progress)
    finally:
        conn.close()

    print()  # newline after final progress line

    if result.error:
        print(f"error: {result.error}", file=sys.stderr)
        return 1

    print(f"Indexed {result.indexed} photo(s).")
    print(f"Library: {db_root / 'takeout-rater'}")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    """Execute the ``score`` sub-command."""
    from takeout_rater.db.connection import (  # noqa: PLC0415
        library_db_path,
        library_state_dir,
        open_library_db,
    )
    from takeout_rater.scorers.registry import list_scorers  # noqa: PLC0415

    photos_root = Path(args.photos_root).resolve()
    db_root = Path(args.db_root).resolve() if args.db_root else photos_root
    db_path = library_db_path(db_root)

    if not db_path.exists():
        print(
            f"error: no library database found at {db_path}\n"
            "       Run 'takeout-rater index <photos_root>' first.",
            file=sys.stderr,
        )
        return 1

    conn = open_library_db(db_root)
    thumbs_dir = library_state_dir(db_root) / "thumbs"

    exit_code = 0

    # -- pHash computation ---------------------------------------------------
    if args.phash:
        from takeout_rater.scoring.phash import compute_phash_all  # noqa: PLC0415

        print("Computing perceptual hashes (dhash) …")

        def _phash_progress(done: int, total: int) -> None:
            print(f"  phash: {done}/{total}", end="\r", flush=True)

        count = compute_phash_all(conn, thumbs_dir, on_progress=_phash_progress)
        print(f"\nStored {count} perceptual hash(es).")

    # -- Scorer runs ---------------------------------------------------------
    scorer_id: str | None = args.scorer

    if scorer_id is not None:
        # Run the named scorer only
        scorer_classes = [cls for cls in list_scorers() if cls.spec().scorer_id == scorer_id]
        if not scorer_classes:
            print(f"error: unknown scorer id '{scorer_id}'.", file=sys.stderr)
            print(
                "       Available scorers: "
                + ", ".join(cls.spec().scorer_id for cls in list_scorers()),
                file=sys.stderr,
            )
            conn.close()
            return 1
        scorer_classes_to_run = scorer_classes
    else:
        # Run all available scorers (skip unavailable ones with a warning)
        scorer_classes_to_run = list_scorers(available_only=True)
        if not scorer_classes_to_run:
            print("No available scorers found.  Nothing to do.")
            conn.close()
            return 0

    for cls in scorer_classes_to_run:
        spec = cls.spec()
        if not cls.is_available():
            print(
                f"  skipping {spec.scorer_id!r}: not available "
                f"(install extras: {', '.join(spec.requires_extras) or 'none'})"
            )
            exit_code = 1
            continue

        scorer = cls.create()
        print(f"Running scorer '{spec.display_name}' ({spec.scorer_id}) …")

        from takeout_rater.scoring.pipeline import run_scorer  # noqa: PLC0415

        scorer_id_label = spec.scorer_id  # bind to avoid B023 (loop variable in closure)

        def _score_progress(done: int, total: int, _label: str = scorer_id_label) -> None:
            print(f"  {_label}: {done}/{total}", end="\r", flush=True)

        num_scored = run_scorer(
            conn,
            scorer,
            thumbs_dir,
            batch_size=args.batch_size,
            skip_existing=not args.rerun,
            on_progress=_score_progress,
        )
        print(f"\n  Done ({num_scored} assets scored).")

    conn.close()
    return exit_code


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

    photos_root = Path(args.photos_root).resolve()
    db_root = Path(args.db_root).resolve() if args.db_root else photos_root
    from takeout_rater.db.connection import library_db_path, open_library_db  # noqa: PLC0415

    db_path = library_db_path(db_root)

    if not db_path.exists():
        print(
            f"error: no library database found at {db_path}\n"
            "       Run 'takeout-rater index <photos_root>' first.",
            file=sys.stderr,
        )
        return 1

    from takeout_rater.ui.app import create_app  # noqa: PLC0415

    conn = open_library_db(db_root)
    app = create_app(photos_root, conn, db_root=db_root)

    url = f"http://{args.host}:{args.port}/assets"
    try:
        port = _find_free_port(args.host, args.port)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if port != args.port:
        print(f"note: port {args.port} is in use, using port {port} instead.")
    url = f"http://{args.host}:{port}/assets"
    print(f"Starting takeout-rater UI at {url}")
    print("Press Ctrl+C to stop.")

    uvicorn.run(app, host=args.host, port=port, log_level="warning")
    conn.close()
    return 0


def _cmd_cluster(args: argparse.Namespace) -> int:
    """Execute the ``cluster`` sub-command."""
    from takeout_rater.clustering.builder import build_clusters  # noqa: PLC0415
    from takeout_rater.db.connection import library_db_path, open_library_db  # noqa: PLC0415

    photos_root = Path(args.photos_root).resolve()
    db_root = Path(args.db_root).resolve() if args.db_root else photos_root
    db_path = library_db_path(db_root)

    if not db_path.exists():
        print(
            f"error: no library database found at {db_path}\n"
            "       Run 'takeout-rater index <photos_root>' first.",
            file=sys.stderr,
        )
        return 1

    conn = open_library_db(db_root)

    print(f"Building clusters (threshold={args.threshold} bits, window={args.window}) …")

    def _progress(done: int, total: int) -> None:
        print(f"  {done}/{total}", end="\r", flush=True)

    n_clusters, n_skipped = build_clusters(
        conn,
        threshold=args.threshold,
        window=args.window,
        min_cluster_size=args.min_size,
        on_progress=_progress,
    )
    skipped_info = f", {n_skipped} skipped" if n_skipped else ""
    print(f"\nFound {n_clusters} cluster(s){skipped_info}.")
    conn.close()
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    """Execute the ``export`` sub-command.

    Copies the best representative from each cluster to the export directory.
    When no scorer is specified, uses the cluster representative (lowest asset ID).
    """
    import shutil  # noqa: PLC0415

    from takeout_rater.db.connection import (  # noqa: PLC0415
        library_db_path,
        library_state_dir,
        open_library_db,
    )
    from takeout_rater.db.queries import (  # noqa: PLC0415
        get_asset_by_id,
        get_asset_scores,
        get_cluster_members,
        list_clusters_with_representatives,
    )

    photos_root = Path(args.photos_root).resolve()
    db_root = Path(args.db_root).resolve() if args.db_root else photos_root
    db_path = library_db_path(db_root)

    if not db_path.exists():
        print(
            f"error: no library database found at {db_path}\n"
            "       Run 'takeout-rater index <photos_root>' first.",
            file=sys.stderr,
        )
        return 1

    if args.scorer and not args.metric:
        print(
            "error: --metric is required when --scorer is specified.",
            file=sys.stderr,
        )
        return 1

    conn = open_library_db(db_root)
    state_dir = library_state_dir(db_root)

    export_dir = Path(args.out).resolve() if args.out else state_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Check there are any clusters before iterating
    from takeout_rater.db.queries import count_clusters  # noqa: PLC0415

    if count_clusters(conn) == 0:
        print("No clusters found.  Run 'takeout-rater cluster <photos_root>' first.")
        conn.close()
        return 0

    copied = 0
    skipped = 0
    _EXPORT_BATCH = 200

    offset = 0
    while True:
        clusters = list_clusters_with_representatives(conn, limit=_EXPORT_BATCH, offset=offset)
        if not clusters:
            break
        offset += len(clusters)

        for cluster_info in clusters:
            cluster_id = cluster_info["cluster_id"]
            members = get_cluster_members(conn, cluster_id)

            if args.scorer and args.metric:
                # Pick member with highest score for the given scorer+metric
                best_asset_id: int | None = None
                best_score: float = float("-inf")
                for asset, _dist, _is_rep in members:
                    scores = get_asset_scores(conn, asset.id)
                    for s in scores:
                        if s["scorer_id"] == args.scorer and s["metric_key"] == args.metric:
                            if s["value"] > best_score:
                                best_score = s["value"]
                                best_asset_id = asset.id
                            break
                if best_asset_id is None:
                    # Fall back to representative if no scores available
                    best_asset_id = next(
                        (a.id for a, _d, is_rep in members if is_rep),
                        members[0][0].id if members else None,
                    )
            else:
                # Use representative (first is_rep=True member)
                best_asset_id = next(
                    (a.id for a, _d, is_rep in members if is_rep),
                    members[0][0].id if members else None,
                )

            if best_asset_id is None:
                continue

            asset = get_asset_by_id(conn, best_asset_id)
            if asset is None:
                continue

            src = photos_root / asset.relpath
            if not src.exists():
                skipped += 1
                continue

            dest = export_dir / f"cluster{cluster_id:06d}_{asset.filename}"
            shutil.copy2(src, dest)
            copied += 1

    conn.close()
    print(f"Exported {copied} file(s) to {export_dir}")
    if skipped:
        print(f"  ({skipped} file(s) skipped — originals not found)")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    """Execute the ``serve`` sub-command.

    Unlike ``browse``, this command reads the photos path from the
    local config file and starts the server even when no library has been
    configured yet.  In that case the UI shows a setup page that lets the user
    select the folder from their browser.
    """
    print("Importing server framework …", flush=True)
    try:
        import uvicorn  # noqa: PLC0415
    except ImportError:
        print(
            "error: uvicorn is required for the serve command.\n"
            "       Install it with: pip install 'takeout-rater[web]'",
            file=sys.stderr,
        )
        return 1

    print("Loading application modules …", flush=True)
    from takeout_rater.config import get_db_root, get_photos_root  # noqa: PLC0415
    from takeout_rater.ui.app import create_app  # noqa: PLC0415

    photos_root = get_photos_root()
    db_root = get_db_root() or photos_root
    conn = None

    if photos_root is not None and db_root is not None:
        from takeout_rater.db.connection import library_db_path, open_library_db  # noqa: PLC0415
        from takeout_rater.db.schema import SchemaMismatchError  # noqa: PLC0415

        db_path = library_db_path(db_root)
        if db_path.exists():
            print("Opening library database …", flush=True)
            try:
                conn = open_library_db(db_root)
            except SchemaMismatchError as exc:
                print(
                    f"error: {exc}\n       Database path: {db_path}",
                    file=sys.stderr,
                    flush=True,
                )
                return _EXIT_SCHEMA_MISMATCH
            print("Database ready.", flush=True)
        else:
            print(
                f"note: library database not found at {db_path}\n"
                "      Run 'takeout-rater index <photos_root>' to index your photos folder.\n"
                "      The UI will remind you.",
            )

    print("Building application …", flush=True)
    app = create_app(photos_root, conn, db_root=db_root)

    try:
        port = _find_free_port(args.host, args.port)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if port != args.port:
        print(f"note: port {args.port} is in use, using port {port} instead.")

    if conn is not None:
        print(f"Starting takeout-rater UI at http://{args.host}:{port}/assets")
    else:
        print(f"Starting takeout-rater UI at http://{args.host}:{port}/setup")
    print("Press Ctrl+C to stop.")

    uvicorn.run(app, host=args.host, port=port, log_level="info")
    if conn is not None:
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

    if args.command == "score":
        return _cmd_score(args)

    if args.command == "browse":
        return _cmd_browse(args)

    if args.command == "cluster":
        return _cmd_cluster(args)

    if args.command == "export":
        return _cmd_export(args)

    if args.command == "serve":
        return _cmd_serve(args)

    print(f"Command '{args.command}' is not yet implemented (see roadmap in README).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
