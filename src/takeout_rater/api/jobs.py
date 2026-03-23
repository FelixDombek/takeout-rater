"""FastAPI router for background job management.

Exposes endpoints to trigger and monitor long-running operations:
- Scoring (run available scorers over indexed assets)
- Clustering (group near-duplicates by pHash)
- Export (copy best-of-cluster assets to the export folder)
- Rehash (compute SHA-256 for already-indexed assets)

Each operation runs in a background thread.  Progress is stored in
``app.state.jobs`` (a dict keyed by job type string) and can be polled via
``GET /api/jobs/status``.

Endpoints
---------
GET  /api/jobs/status            – status of all (or a specific) background job
GET  /api/jobs/scorers           – list available scorers
POST /api/jobs/score/start       – start scoring (optional scorer_id body field)
POST /api/jobs/cluster/start     – start clustering
POST /api/jobs/export/start      – start best-of-cluster export
POST /api/jobs/rehash/start      – start SHA-256 rehash
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter()

# ---------------------------------------------------------------------------
# Job progress dataclass
# ---------------------------------------------------------------------------

_JOB_TYPES = ("score", "cluster", "export", "rehash")


@dataclass
class JobProgress:
    """State for a single background job.

    Attributes:
        running: ``True`` while the job thread is active.
        done: ``True`` once the job has finished (success or error).
        error: Human-readable error message, or ``None`` on success.
        message: Short human-readable status line (updated during the run).
        scored: Number of assets scored so far (scoring job only).
        total: Total items to process (scoring job only).
        job_type: One of ``"score"``, ``"cluster"``, ``"export"``, ``"rehash"``.
    """

    job_type: str = ""
    running: bool = False
    done: bool = False
    error: str | None = None
    message: str = ""
    scored: int = 0
    total: int = 0


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
        "scored": p.scored,
        "total": p.total,
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
                    "scored": 0,
                    "total": 0,
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
                    "scored": 0,
                    "total": 0,
                }
            )
        else:
            statuses.append(_job_status_dict(p))
    return JSONResponse(statuses)


# ---------------------------------------------------------------------------
# GET /api/jobs/scorers
# ---------------------------------------------------------------------------


@router.get("/api/jobs/scorers")
def list_available_scorers() -> JSONResponse:
    """Return a list of available scorers.

    Each item has ``id`` and ``name`` fields.
    """
    from takeout_rater.scorers.registry import list_scorers  # noqa: PLC0415

    result = []
    for cls in list_scorers():
        spec = cls.spec()
        result.append(
            {
                "id": spec.scorer_id,
                "name": spec.display_name,
                "available": cls.is_available(),
            }
        )
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# POST /api/jobs/score/start
# ---------------------------------------------------------------------------


class _ScoreStartBody(BaseModel):
    scorer_id: str | None = None
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
    rerun = body.rerun

    def _worker() -> None:
        from takeout_rater.db.connection import (
            library_state_dir,  # noqa: PLC0415
            open_library_db,  # noqa: PLC0415
        )
        from takeout_rater.scorers.registry import list_scorers  # noqa: PLC0415
        from takeout_rater.scoring.phash import compute_phashes  # noqa: PLC0415
        from takeout_rater.scoring.pipeline import run_scorer_by_id  # noqa: PLC0415

        worker_conn = open_library_db(library_root)
        thumbs_dir = library_state_dir(library_root) / "thumbs"
        try:
            # Compute pHashes first (needed for clustering later)
            progress.message = "Computing perceptual hashes…"
            compute_phashes(worker_conn, thumbs_dir)

            # Determine which scorers to run
            if scorer_id:
                scorer_ids = [scorer_id]
            else:
                scorer_ids = [cls.spec().scorer_id for cls in list_scorers(available_only=True)]

            total_scorers = len(scorer_ids)
            for idx, sid in enumerate(scorer_ids):
                progress.message = f"Scoring with {sid!r} ({idx + 1}/{total_scorers})…"
                progress.scored = 0
                progress.total = 0

                def _cb(scored: int, total: int) -> None:
                    progress.scored = scored
                    progress.total = total

                run_scorer_by_id(
                    worker_conn,
                    sid,
                    thumbs_dir,
                    skip_existing=not rerun,
                    on_progress=_cb,
                )

            progress.message = "Scoring complete."
            progress.running = False
            progress.done = True
        except Exception as exc:  # noqa: BLE001
            progress.error = str(exc)
            progress.message = f"Error: {exc}"
            progress.running = False
            progress.done = True
        finally:
            worker_conn.close()

    thread = threading.Thread(target=_worker, daemon=True, name="takeout-rater-scorer")
    thread.start()

    return JSONResponse({"status": "started"})


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
                progress.scored = processed
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
                    progress.scored = copied
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
                if not src.exists():
                    skipped += 1
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

                if hashed % 100 == 0:
                    worker_conn.commit()
                    progress.scored = hashed
                    progress.message = f"Rehashed {hashed}/{total} asset(s)…"

            worker_conn.commit()
            progress.scored = hashed
            progress.message = f"Rehash complete — {hashed} hash(es) computed" + (
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

    thread = threading.Thread(target=_worker, daemon=True, name="takeout-rater-rehash")
    thread.start()

    return JSONResponse({"status": "started"})
