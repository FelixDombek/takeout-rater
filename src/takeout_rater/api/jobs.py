"""FastAPI router for background job management.

Exposes endpoints to trigger and monitor long-running operations:
- Index (initial scan or re-index of the Takeout folder)
- Scoring (run available scorers over indexed assets)
- Clustering (group near-duplicates by pHash)
- Export (copy best-of-cluster assets to the export folder)
- Rehash (compute SHA-256 for already-indexed assets)
- Rescan (re-process assets through the indexing pipeline, updating indexer_version)

Each operation runs in a background thread.  Progress is stored in
``app.state.jobs`` (a dict keyed by job type string) and can be polled via
``GET /api/jobs/status``.

Endpoints
---------
GET  /api/jobs/status            – status of all (or a specific) background job
GET  /api/jobs/scorers           – list available scorers
POST /api/jobs/index/start       – start initial indexing (or re-index)
POST /api/jobs/score/start       – start scoring (optional scorer_id body field)
POST /api/jobs/cluster/start     – start clustering
POST /api/jobs/export/start      – start best-of-cluster export
POST /api/jobs/rehash/start      – start SHA-256 rehash
POST /api/jobs/rescan/start      – start library rescan (update indexer_version)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter()

# ---------------------------------------------------------------------------
# Job progress dataclass
# ---------------------------------------------------------------------------

_JOB_TYPES = ("index", "score", "cluster", "export", "rehash", "rescan")


@dataclass
class JobProgress:
    """State for a single background job.

    Attributes:
        running: ``True`` while the job thread is active.
        done: ``True`` once the job has finished (success or error).
        error: Human-readable error message, or ``None`` on success.
        message: Short human-readable status line (updated during the run).
        processed: General-purpose "items processed so far" counter.  For the
            ``"score"`` job this is the number of assets scored; for
            ``"index"`` / ``"rescan"`` / ``"rehash"`` jobs it is the number of
            assets indexed / rescanned / rehashed respectively.
        total: Total items to process (0 until the count is known).
        current_item: Current item being processed (e.g., directory path for
            ``"index"`` / ``"rescan"``, file path for ``"rehash"``).  Empty
            string if unavailable or not applicable for the job type.
        current_scorer_id: For the ``"score"`` job, the scorer currently being
            run.  Empty string when no specific scorer is active.
        current_variant_id: For the ``"score"`` job, the variant of the scorer
            currently being run.  Empty string when no variant is active or the
            scorer does not use variants.
        job_type: One of ``"index"``, ``"score"``, ``"cluster"``,
            ``"export"``, ``"rehash"``, ``"rescan"``.
        cancel_event: Set this event to request cancellation of the running
            job.  The worker checks it between batches and exits early.
    """

    job_type: str = ""
    running: bool = False
    done: bool = False
    error: str | None = None
    message: str = ""
    processed: int = 0
    total: int = 0
    current_item: str = ""
    current_scorer_id: str = ""
    current_variant_id: str = ""
    cancel_event: threading.Event = field(default_factory=threading.Event)


def _get_jobs(app: object) -> dict[str, JobProgress]:
    """Return (and lazily initialise) ``app.state.jobs``."""
    jobs: dict[str, JobProgress] | None = getattr(app.state, "jobs", None)  # type: ignore[union-attr]
    if jobs is None:
        jobs = {}
        app.state.jobs = jobs  # type: ignore[union-attr]
    return jobs


def _job_status_dict(p: JobProgress) -> dict:
    return {
        "job_type": p.job_type,
        "running": p.running,
        "done": p.done,
        "error": p.error,
        "message": p.message,
        "processed": p.processed,
        "total": p.total,
        "current_item": p.current_item,
        "current_scorer_id": p.current_scorer_id,
        "current_variant_id": p.current_variant_id,
        "cancelled": p.cancel_event.is_set(),
    }


def _require_db(request: Request) -> None:
    if request.app.state.db_conn is None:
        raise HTTPException(status_code=503, detail="Library not configured.")


def _require_library_root(request: Request) -> Path:
    _require_db(request)
    lr: Path | None = request.app.state.library_root
    if lr is None:
        raise HTTPException(status_code=503, detail="Library not configured.")
    return lr


# ---------------------------------------------------------------------------
# GET /api/jobs/status
# ---------------------------------------------------------------------------


@router.get("/api/jobs/status")
def jobs_status(request: Request, job_type: str | None = None) -> JSONResponse:
    """Return current status for all jobs (or a specific job).

    Query params
    ------------
    job_type : str, optional
        One of ``score``, ``cluster``, ``export``, ``rehash``.  When omitted
        all job statuses are returned as a list.
    """
    jobs = _get_jobs(request.app)
    if job_type is not None:
        if job_type not in _JOB_TYPES:
            raise HTTPException(status_code=400, detail=f"Unknown job_type: {job_type!r}")
        p = jobs.get(job_type)
        if p is None:
            return JSONResponse(
                {
                    "job_type": job_type,
                    "running": False,
                    "done": False,
                    "error": None,
                    "message": "",
                    "processed": 0,
                    "total": 0,
                    "current_item": "",
                    "current_scorer_id": "",
                    "cancelled": False,
                }
            )
        return JSONResponse(_job_status_dict(p))

    # Return all
    statuses = []
    for jt in _JOB_TYPES:
        p = jobs.get(jt)
        if p is None:
            statuses.append(
                {
                    "job_type": jt,
                    "running": False,
                    "done": False,
                    "error": None,
                    "message": "",
                    "processed": 0,
                    "total": 0,
                    "current_item": "",
                    "current_scorer_id": "",
                    "cancelled": False,
                }
            )
        else:
            statuses.append(_job_status_dict(p))
    return JSONResponse(statuses)


# ---------------------------------------------------------------------------
# Index job helper (used by both POST /api/jobs/index/start and by config_routes)
# ---------------------------------------------------------------------------


def _start_index_job(app: object, library_root: Path) -> None:
    """Launch a background thread that indexes *library_root*.

    Progress is stored in ``app.state.jobs["index"]`` as a :class:`JobProgress`
    entry so that it is visible in the unified job queue.  The ``message``
    field surfaces directory-level progress (phase, dirs_scanned, current_dir).

    If an indexing run is already active, this call is a no-op.
    """
    jobs = _get_jobs(app)
    existing = jobs.get("index")
    if existing is not None and existing.running:
        return

    progress = JobProgress(job_type="index", running=True, message="Starting\u2026")
    jobs["index"] = progress

    def _worker() -> None:
        from takeout_rater.db.connection import open_library_db as _open  # noqa: PLC0415
        from takeout_rater.indexing.run import IndexProgress, run_index  # noqa: PLC0415

        def _cb(p: IndexProgress) -> None:
            progress.total = p.found
            progress.processed = p.indexed
            progress.current_item = p.current_dir
            if p.phase == "scanning" and p.total_dirs > 0:
                msg = (
                    f"Scanning folders ({p.dirs_scanned}\u202f/\u202f{p.total_dirs})"
                    + (f"\u2002\u2013\u2002{p.current_dir}" if p.current_dir else "")
                    + "\u2026"
                )
            elif p.phase == "indexing" and p.found > 0:
                msg = f"Indexing\u2026 {p.indexed}\u202f/\u202f{p.found} photos" + (
                    f"\u2002\u2013\u2002{p.current_dir}"
                    if p.current_dir and p.current_dir != "."
                    else ""
                )
            else:
                msg = "Scanning for photos\u2026"
            progress.message = msg

        worker_conn = _open(library_root)
        try:
            result = run_index(library_root, worker_conn, on_progress=_cb)
            progress.total = result.found
            progress.processed = result.indexed
            progress.current_item = ""
            if result.error:
                progress.error = result.error
                progress.message = f"Error: {result.error}"
            else:
                progress.message = f"Indexed {result.indexed} photo(s)."
            progress.running = False
            progress.done = True
        except Exception as exc:  # noqa: BLE001
            progress.error = str(exc)
            progress.message = f"Error: {exc}"
            progress.current_item = ""
            progress.running = False
            progress.done = True
        finally:
            worker_conn.close()

    thread = threading.Thread(target=_worker, daemon=True, name="takeout-rater-indexer")
    thread.start()


# ---------------------------------------------------------------------------
# POST /api/jobs/index/start
# ---------------------------------------------------------------------------


@router.post("/api/jobs/index/start")
def start_index_job(request: Request) -> JSONResponse:
    """Start a background index run (or re-index).

    Scans the Takeout folder and upserts newly discovered assets into the
    library.  Useful after adding new Takeout archives or if the initial
    setup index was interrupted.

    Returns ``409`` if an index job is already running.
    """
    _require_library_root(request)
    jobs = _get_jobs(request.app)

    existing = jobs.get("index")
    if existing is not None and existing.running:
        raise HTTPException(status_code=409, detail="An index job is already running.")

    _start_index_job(request.app, request.app.state.library_root)
    return JSONResponse({"status": "started"})


# ---------------------------------------------------------------------------
# GET /api/jobs/scorers
# ---------------------------------------------------------------------------


@router.get("/api/jobs/scorers")
def list_available_scorers() -> JSONResponse:
    """Return a list of scorers with metadata for the Scoring page.

    Each item has ``id``, ``name``, ``description``, ``technical_description``,
    ``version``, ``available``, and ``variants`` fields.
    """
    from takeout_rater.scorers.registry import list_scorers  # noqa: PLC0415

    result = []
    for cls in list_scorers():
        spec = cls.spec()
        result.append(
            {
                "id": spec.scorer_id,
                "name": spec.display_name,
                "description": spec.description,
                "technical_description": spec.technical_description,
                "version": spec.version,
                "available": cls.is_available(),
                "requires_extras": list(spec.requires_extras),
                "variants": [
                    {"id": v.variant_id, "name": v.display_name, "description": v.description}
                    for v in spec.variants
                ],
            }
        )
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# POST /api/jobs/score/start
# ---------------------------------------------------------------------------


class _ScoreStartBody(BaseModel):
    scorer_id: str | None = None
    variant_id: str | None = None
    rerun: bool = False


@router.post("/api/jobs/score/start")
def start_score_job(body: _ScoreStartBody, request: Request) -> JSONResponse:
    """Start a background scoring run.

    Triggers scoring for all available scorers (or a specific one when
    ``scorer_id`` is supplied).  Returns ``409`` if a score job is already
    running.
    """
    _require_library_root(request)
    jobs = _get_jobs(request.app)

    existing = jobs.get("score")
    if existing is not None and existing.running:
        raise HTTPException(status_code=409, detail="A score job is already running.")

    library_root: Path = request.app.state.library_root
    progress = JobProgress(job_type="score", running=True, message="Starting…")
    jobs["score"] = progress

    scorer_id = body.scorer_id
    variant_id = body.variant_id
    rerun = body.rerun

    def _worker() -> None:
        from takeout_rater.db.connection import (
            library_state_dir,  # noqa: PLC0415
            open_library_db,  # noqa: PLC0415
        )
        from takeout_rater.scorers.registry import list_scorers  # noqa: PLC0415
        from takeout_rater.scoring.phash import compute_phash_all  # noqa: PLC0415
        from takeout_rater.scoring.pipeline import run_scorers_parallel  # noqa: PLC0415

        worker_conn = open_library_db(library_root)
        thumbs_dir = library_state_dir(library_root) / "thumbs"
        try:
            # ── Phase 1: Compute pHashes (needed for clustering later) ──────
            progress.message = "Computing perceptual hashes…"
            progress.processed = 0
            progress.total = 0
            progress.current_item = ""

            # Pre-build id→relpath map for current-item display.
            _id_to_relpath: dict[int, str] = {
                row[0]: row[1]
                for row in worker_conn.execute("SELECT id, relpath FROM assets").fetchall()
            }

            def _phash_progress(done: int, total: int) -> None:
                progress.processed = done
                progress.total = total
                progress.message = f"Computing perceptual hashes… {done}\u202f/\u202f{total}"

            def _phash_item(asset_id: int, done: int, total: int) -> None:
                progress.current_item = _id_to_relpath.get(asset_id, "")

            compute_phash_all(
                worker_conn,
                thumbs_dir,
                on_progress=_phash_progress,
                on_item=_phash_item,
                cancel_check=progress.cancel_event.is_set,
            )

            progress.current_item = ""

            # ── Phase 2: Run scorers in parallel ─────────────────────────────
            # Build the list of scorer instances to run.
            # When no specific scorer is requested every available scorer is
            # run for *all* of its variants (or its single default variant if
            # the scorer defines no explicit variants).
            scorer_instances = []
            if scorer_id:
                cls_map = {cls.spec().scorer_id: cls for cls in list_scorers()}
                target_cls = cls_map.get(scorer_id)
                if target_cls:
                    spec = target_cls.spec()
                    if variant_id:
                        scorer_instances.append(target_cls.create(variant_id=variant_id))
                    elif spec.variants:
                        for v in spec.variants:
                            scorer_instances.append(target_cls.create(variant_id=v.variant_id))
                    else:
                        scorer_instances.append(target_cls.create())
                else:
                    # Unknown scorer id — will surface as empty run.
                    pass
            else:
                for cls in list_scorers(available_only=True):
                    spec = cls.spec()
                    if spec.variants:
                        for v in spec.variants:
                            scorer_instances.append(cls.create(variant_id=v.variant_id))
                    else:
                        scorer_instances.append(cls.create())

            n_scorers = len(scorer_instances)
            progress.message = f"Scoring with {n_scorers} scorer(s)…"
            progress.processed = 0
            progress.total = 0

            def _parallel_progress(done: int, total: int) -> None:
                progress.processed = done
                progress.total = total
                progress.message = f"Scoring… {done}\u202f/\u202f{total} assets"

            run_scorers_parallel(
                worker_conn,
                scorer_instances,
                thumbs_dir,
                skip_existing=not rerun,
                on_progress=_parallel_progress,
                cancel_check=progress.cancel_event.is_set,
            )

            progress.current_scorer_id = ""
            progress.current_variant_id = ""
            if progress.cancel_event.is_set():
                progress.message = "Scoring cancelled."
            else:
                progress.message = "Scoring complete."
            progress.current_item = ""
            progress.running = False
            progress.done = True
        except Exception as exc:  # noqa: BLE001
            progress.error = str(exc)
            progress.message = f"Error: {exc}"
            progress.current_item = ""
            progress.current_scorer_id = ""
            progress.current_variant_id = ""
            progress.running = False
            progress.done = True
        finally:
            worker_conn.close()

    thread = threading.Thread(target=_worker, daemon=True, name="takeout-rater-scorer")
    thread.start()

    return JSONResponse({"status": "started"})


# ---------------------------------------------------------------------------
# POST /api/jobs/score/cancel
# ---------------------------------------------------------------------------


@router.post("/api/jobs/score/cancel")
def cancel_score_job(request: Request) -> JSONResponse:
    """Request cancellation of the currently running score job.

    Sets the cancel event on the running job so the worker exits after the
    current batch completes.  Returns ``404`` if no score job is running.
    """
    jobs = _get_jobs(request.app)
    p = jobs.get("score")
    if p is None or not p.running:
        raise HTTPException(status_code=404, detail="No score job is currently running.")
    p.cancel_event.set()
    return JSONResponse({"status": "cancelling"})


# ---------------------------------------------------------------------------
# POST /api/jobs/cluster/start
# ---------------------------------------------------------------------------


class _ClusterStartBody(BaseModel):
    threshold: int = 10
    window: int = 200
    min_size: int = 2


@router.post("/api/jobs/cluster/start")
def start_cluster_job(body: _ClusterStartBody, request: Request) -> JSONResponse:
    """Start a background clustering run.

    Returns ``409`` if a cluster job is already running.
    """
    _require_library_root(request)
    jobs = _get_jobs(request.app)

    existing = jobs.get("cluster")
    if existing is not None and existing.running:
        raise HTTPException(status_code=409, detail="A cluster job is already running.")

    library_root: Path = request.app.state.library_root
    progress = JobProgress(job_type="cluster", running=True, message="Starting…")
    jobs["cluster"] = progress

    threshold = body.threshold
    window = body.window
    min_size = body.min_size

    def _worker() -> None:
        from takeout_rater.clustering.builder import build_clusters  # noqa: PLC0415
        from takeout_rater.db.connection import open_library_db  # noqa: PLC0415

        worker_conn = open_library_db(library_root)
        try:
            progress.message = "Building clusters…"

            def _cb(processed: int, total: int) -> None:
                progress.processed = processed
                progress.total = total
                if total > 0:
                    progress.message = f"Clustering… {processed}/{total} hashes"

            n_clusters = build_clusters(
                worker_conn,
                threshold=threshold,
                window=window,
                min_cluster_size=min_size,
                on_progress=_cb,
            )
            progress.message = f"Clustering complete — {n_clusters} cluster(s) found."
            progress.running = False
            progress.done = True
        except Exception as exc:  # noqa: BLE001
            progress.error = str(exc)
            progress.message = f"Error: {exc}"
            progress.running = False
            progress.done = True
        finally:
            worker_conn.close()

    thread = threading.Thread(target=_worker, daemon=True, name="takeout-rater-cluster")
    thread.start()

    return JSONResponse({"status": "started"})


# ---------------------------------------------------------------------------
# POST /api/jobs/export/start
# ---------------------------------------------------------------------------


class _ExportStartBody(BaseModel):
    scorer_id: str | None = None
    metric_key: str | None = None


@router.post("/api/jobs/export/start")
def start_export_job(body: _ExportStartBody, request: Request) -> JSONResponse:
    """Start a background export run.

    Copies the best representative from each cluster to the exports folder.
    Returns ``409`` if an export job is already running.
    """
    _require_library_root(request)
    jobs = _get_jobs(request.app)

    existing = jobs.get("export")
    if existing is not None and existing.running:
        raise HTTPException(status_code=409, detail="An export job is already running.")

    library_root: Path = request.app.state.library_root
    scorer_id = body.scorer_id
    metric_key = body.metric_key

    if scorer_id and not metric_key:
        raise HTTPException(
            status_code=400, detail="metric_key is required when scorer_id is specified."
        )

    progress = JobProgress(job_type="export", running=True, message="Starting…")
    jobs["export"] = progress

    def _worker() -> None:
        import shutil  # noqa: PLC0415

        from takeout_rater.db.connection import (  # noqa: PLC0415
            library_state_dir,
            open_library_db,
        )
        from takeout_rater.db.queries import (  # noqa: PLC0415
            count_clusters,
            get_asset_by_id,
            get_asset_scores,
            get_cluster_members,
            list_clusters_with_representatives,
        )
        from takeout_rater.indexing.scanner import find_google_photos_root  # noqa: PLC0415

        worker_conn = open_library_db(library_root)
        try:
            if count_clusters(worker_conn) == 0:
                progress.message = "No clusters found. Run 'Cluster' first."
                progress.running = False
                progress.done = True
                worker_conn.close()
                return

            takeout_root = find_google_photos_root(library_root / "Takeout")
            export_dir = library_state_dir(library_root) / "exports"
            export_dir.mkdir(parents=True, exist_ok=True)

            _BATCH = 200
            offset = 0
            copied = 0
            skipped = 0

            # Count total clusters for progress reporting
            total_clusters = count_clusters(worker_conn)
            progress.total = total_clusters

            while True:
                clusters = list_clusters_with_representatives(
                    worker_conn, limit=_BATCH, offset=offset
                )
                if not clusters:
                    break
                offset += len(clusters)

                for cluster_info in clusters:
                    cluster_id = cluster_info["cluster_id"]
                    members = get_cluster_members(worker_conn, cluster_id)

                    if scorer_id and metric_key:
                        best_asset_id: int | None = None
                        best_score: float = float("-inf")
                        for asset, _dist, _is_rep in members:
                            scores = get_asset_scores(worker_conn, asset.id)
                            for s in scores:
                                if s["scorer_id"] == scorer_id and s["metric_key"] == metric_key:
                                    if s["value"] > best_score:
                                        best_score = s["value"]
                                        best_asset_id = asset.id
                                    break
                        if best_asset_id is None:
                            best_asset_id = next(
                                (a.id for a, _d, is_rep in members if is_rep),
                                members[0][0].id if members else None,
                            )
                    else:
                        best_asset_id = next(
                            (a.id for a, _d, is_rep in members if is_rep),
                            members[0][0].id if members else None,
                        )

                    if best_asset_id is None:
                        continue

                    asset = get_asset_by_id(worker_conn, best_asset_id)
                    if asset is None:
                        continue

                    src = takeout_root / asset.relpath
                    if not src.exists():
                        skipped += 1
                        continue

                    dest = export_dir / f"cluster{cluster_id:06d}_{asset.filename}"
                    shutil.copy2(src, dest)
                    copied += 1
                    progress.processed = copied
                    progress.message = f"Exported {copied} file(s)…"

            progress.message = f"Export complete — {copied} file(s) copied to {export_dir}" + (
                f" ({skipped} skipped)" if skipped else ""
            )
            progress.running = False
            progress.done = True
        except Exception as exc:  # noqa: BLE001
            progress.error = str(exc)
            progress.message = f"Error: {exc}"
            progress.running = False
            progress.done = True
        finally:
            worker_conn.close()

    thread = threading.Thread(target=_worker, daemon=True, name="takeout-rater-export")
    thread.start()

    return JSONResponse({"status": "started"})


# ---------------------------------------------------------------------------
# POST /api/jobs/rehash/start
# ---------------------------------------------------------------------------


class _RehashStartBody(BaseModel):
    rehash_all: bool = False


@router.post("/api/jobs/rehash/start")
def start_rehash_job(body: _RehashStartBody, request: Request) -> JSONResponse:
    """Start a background SHA-256 rehash run.

    Returns ``409`` if a rehash job is already running.
    """
    _require_library_root(request)
    jobs = _get_jobs(request.app)

    existing = jobs.get("rehash")
    if existing is not None and existing.running:
        raise HTTPException(status_code=409, detail="A rehash job is already running.")

    library_root: Path = request.app.state.library_root
    rehash_all = body.rehash_all
    progress = JobProgress(job_type="rehash", running=True, message="Starting…")
    jobs["rehash"] = progress

    def _worker() -> None:
        import hashlib  # noqa: PLC0415

        from takeout_rater.db.connection import open_library_db  # noqa: PLC0415
        from takeout_rater.indexing.scanner import find_google_photos_root  # noqa: PLC0415

        worker_conn = open_library_db(library_root)
        try:
            takeout_root = find_google_photos_root(library_root / "Takeout")

            if rehash_all:
                cur = worker_conn.execute("SELECT id, relpath FROM assets ORDER BY id")
            else:
                cur = worker_conn.execute(
                    "SELECT id, relpath FROM assets WHERE sha256 IS NULL ORDER BY id"
                )
            rows = cur.fetchall()
            total = len(rows)
            progress.total = total
            progress.message = f"Rehashing {total} asset(s)…"

            hashed = 0
            skipped = 0
            for asset_id, relpath in rows:
                src = takeout_root / relpath
                progress.current_item = relpath
                if not src.exists():
                    skipped += 1
                    progress.processed = hashed
                    progress.message = f"Rehashing {hashed + skipped}\u202f/\u202f{total}\u2026"
                    continue
                h = hashlib.sha256()
                try:
                    with open(src, "rb") as f:
                        for chunk in iter(lambda: f.read(65536), b""):
                            h.update(chunk)
                    digest = h.hexdigest()
                    worker_conn.execute(
                        "UPDATE assets SET sha256 = ? WHERE id = ?", (digest, asset_id)
                    )
                    hashed += 1
                except OSError:
                    skipped += 1
                    continue

                progress.processed = hashed
                progress.message = f"Rehashing {hashed + skipped}\u202f/\u202f{total}\u2026"
                if hashed % 100 == 0:
                    worker_conn.commit()

            worker_conn.commit()
            progress.processed = hashed
            progress.current_item = ""
            progress.message = f"Rehash complete — {hashed} hash(es) computed" + (
                f" ({skipped} skipped)" if skipped else ""
            )
            progress.running = False
            progress.done = True
        except Exception as exc:  # noqa: BLE001
            progress.error = str(exc)
            progress.message = f"Error: {exc}"
            progress.current_item = ""
            progress.running = False
            progress.done = True
        finally:
            worker_conn.close()

    thread = threading.Thread(target=_worker, daemon=True, name="takeout-rater-rehash")
    thread.start()

    return JSONResponse({"status": "started"})


# ---------------------------------------------------------------------------
# POST /api/jobs/rescan/start
# ---------------------------------------------------------------------------


class _RescanStartBody(BaseModel):
    mode: str = "missing_only"  # "missing_only" | "full"


@router.post("/api/jobs/rescan/start")
def start_rescan_job(body: _RescanStartBody, request: Request) -> JSONResponse:
    """Start a background library rescan.

    Re-processes existing assets through the indexing pipeline: re-parses
    sidecar metadata, regenerates missing or stale thumbnails, and stamps
    ``indexer_version = CURRENT_INDEXER_VERSION`` on each asset.

    Body fields
    -----------
    mode : str
        ``"missing_only"`` (default) processes only assets whose
        ``indexer_version`` is ``NULL`` or less than
        ``CURRENT_INDEXER_VERSION``.  ``"full"`` processes all assets and
        unconditionally regenerates all thumbnails.

    Returns ``409`` if a rescan job is already running.
    """
    _require_library_root(request)
    jobs = _get_jobs(request.app)

    existing = jobs.get("rescan")
    if existing is not None and existing.running:
        raise HTTPException(status_code=409, detail="A rescan job is already running.")

    if body.mode not in ("missing_only", "full"):
        raise HTTPException(status_code=400, detail="mode must be 'missing_only' or 'full'.")

    library_root: Path = request.app.state.library_root
    mode = body.mode
    progress = JobProgress(job_type="rescan", running=True, message="Starting…")
    jobs["rescan"] = progress

    def _worker() -> None:
        from takeout_rater.db.connection import (  # noqa: PLC0415
            library_state_dir,
            open_library_db,
        )
        from takeout_rater.db.queries import (  # noqa: PLC0415
            CURRENT_INDEXER_VERSION,
            list_asset_ids_needing_rescan,
        )

        worker_conn = open_library_db(library_root)
        try:
            # Try to locate the photos root for sidecar re-parsing and
            # thumbnail regeneration; continue even if the Takeout directory
            # is not present (e.g. in tests).
            photos_root = None
            try:
                from takeout_rater.indexing.scanner import (  # noqa: PLC0415
                    find_google_photos_root,
                )

                photos_root = find_google_photos_root(library_root / "Takeout")
            except (FileNotFoundError, ValueError, OSError):
                pass

            thumbs_dir = library_state_dir(library_root) / "thumbs"
            thumbs_dir.mkdir(parents=True, exist_ok=True)

            rows = list_asset_ids_needing_rescan(worker_conn, full=(mode == "full"))
            total = len(rows)
            progress.total = total
            progress.message = f"Rescanning {total} asset(s)…"

            processed = 0
            skipped = 0
            thumbs_ok = 0
            thumbs_skip = 0

            for asset_id, _relpath, sidecar_relpath in rows:
                progress.current_item = sidecar_relpath or _relpath
                updates: dict = {}

                # Re-parse sidecar when the library files are accessible.
                if photos_root is not None and sidecar_relpath:
                    sidecar_path = photos_root / sidecar_relpath
                    if sidecar_path.exists():
                        try:
                            from takeout_rater.indexing.sidecar import (  # noqa: PLC0415
                                parse_sidecar,
                            )

                            sidecar = parse_sidecar(sidecar_path)
                            updates.update(
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
                                        None
                                        if sidecar.favorited is None
                                        else int(sidecar.favorited)
                                    ),
                                    "archived": (
                                        None if sidecar.archived is None else int(sidecar.archived)
                                    ),
                                    "trashed": (
                                        None if sidecar.trashed is None else int(sidecar.trashed)
                                    ),
                                    "origin_type": sidecar.origin_type,
                                    "origin_device_type": sidecar.origin_device_type,
                                    "origin_device_folder": sidecar.origin_device_folder,
                                    "app_source_package": sidecar.app_source_package,
                                }
                            )
                        except (ValueError, OSError):
                            skipped += 1

                # Always stamp the indexer version.
                updates["indexer_version"] = CURRENT_INDEXER_VERSION

                # Guard: only allow known asset columns to avoid accidental
                # or future-introduced SQL injection through dict key names.
                _ALLOWED_ASSET_COLS = frozenset(
                    {
                        "title",
                        "description",
                        "google_photos_url",
                        "taken_at",
                        "created_at_sidecar",
                        "image_views",
                        "geo_lat",
                        "geo_lon",
                        "geo_alt",
                        "geo_exif_lat",
                        "geo_exif_lon",
                        "geo_exif_alt",
                        "favorited",
                        "archived",
                        "trashed",
                        "origin_type",
                        "origin_device_type",
                        "origin_device_folder",
                        "app_source_package",
                        "indexer_version",
                    }
                )
                safe_updates = {k: v for k, v in updates.items() if k in _ALLOWED_ASSET_COLS}

                set_clause = ", ".join(f"{k} = ?" for k in safe_updates)
                worker_conn.execute(
                    f"UPDATE assets SET {set_clause} WHERE id = ?",  # noqa: S608
                    [*safe_updates.values(), asset_id],
                )

                # Regenerate thumbnail when the original file is accessible.
                # missing_only: generate only if the thumb file is absent.
                # full: always regenerate (fixes stale/corrupt thumbnails).
                if photos_root is not None:
                    from takeout_rater.indexing.thumbnailer import (  # noqa: PLC0415
                        generate_thumbnail,
                        thumb_path_for_id,
                    )

                    image_path = photos_root / _relpath
                    thumb = thumb_path_for_id(thumbs_dir, asset_id)
                    if image_path.exists() and (mode == "full" or not thumb.exists()):
                        try:
                            generate_thumbnail(image_path, thumb)
                            thumbs_ok += 1
                        except (ImportError, OSError):
                            thumbs_skip += 1
                    else:
                        thumbs_skip += 1

                processed += 1
                progress.processed = processed
                progress.message = f"Rescanning {processed}\u202f/\u202f{total}\u2026"
                if processed % 100 == 0:
                    worker_conn.commit()

            worker_conn.commit()
            progress.processed = processed
            progress.current_item = ""
            extras: list[str] = []
            if skipped:
                extras.append(f"{skipped} sidecar error(s)")
            if thumbs_ok:
                extras.append(f"{thumbs_ok} thumbnail(s) regenerated")
            progress.message = f"Rescan complete — {processed} asset(s) processed." + (
                f" ({', '.join(extras)})" if extras else ""
            )
            progress.running = False
            progress.done = True
        except Exception as exc:  # noqa: BLE001
            progress.error = str(exc)
            progress.message = f"Error: {exc}"
            progress.current_item = ""
            progress.running = False
            progress.done = True
        finally:
            worker_conn.close()

    thread = threading.Thread(target=_worker, daemon=True, name="takeout-rater-rescan")
    thread.start()

    return JSONResponse({"status": "started"})
