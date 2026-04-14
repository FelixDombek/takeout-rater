"""FastAPI router for asset listing, detail, and thumbnail serving."""

from __future__ import annotations

import datetime
import json
import sqlite3
from collections import defaultdict
from collections.abc import Generator
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from takeout_rater.db.queries import (
    AssetRow,
    SortCriterion,
    count_assets,
    count_assets_deduped,
    count_assets_multi_sort,
    count_assets_newer_than,
    count_assets_with_score,
    get_asset_alias_paths,
    get_asset_by_id,
    get_asset_scores,
    get_clip_embedding_for_asset,
    get_phash,
    get_taken_at_range,
    list_assets,
    list_assets_by_score,
    list_assets_deduped,
    list_assets_multi_sort,
    list_available_score_metrics_with_variants,
    list_clip_user_tags,
    list_view_presets,
)
from takeout_rater.indexing.thumbnailer import thumb_path_for_id
from takeout_rater.scorers.registry import list_specs

router = APIRouter()

_PAGE_SIZE = 50


def _get_conn(request: Request) -> Generator[sqlite3.Connection, None, None]:
    """Dependency: open a per-request DB connection and close it when done.

    Each request gets its own ``sqlite3.Connection`` so that concurrent
    requests handled in separate threads do not share the same connection
    object, which would cause ``sqlite3.InterfaceError`` under load.

    Redirects to the setup page if the library has not been configured yet.
    """
    db_path = request.app.state.db_path
    if db_path is not None:
        from takeout_rater.db.connection import open_db  # noqa: PLC0415

        conn = open_db(db_path)
        try:
            yield conn
        finally:
            conn.close()
        return

    # Fallback for in-memory databases (used in tests).
    conn = request.app.state.db_conn
    if conn is None:
        raise HTTPException(status_code=503, detail="Library not configured — visit /setup")
    yield conn


def _get_takeout_root(request: Request) -> Path:
    """Dependency: retrieve the takeout root path from the app state."""
    return request.app.state.takeout_root


def _get_thumbs_dir(request: Request) -> Path:
    """Dependency: retrieve the thumbs directory path from the app state."""
    thumbs_dir = request.app.state.thumbs_dir
    if thumbs_dir is None:
        raise HTTPException(status_code=503, detail="Library not configured — visit /setup")
    return thumbs_dir


def _read_sidecar_json(takeout_root: Path | None, asset: AssetRow) -> str | None:
    """Return the pretty-printed raw sidecar JSON for *asset*, or ``None``.

    Reads ``takeout_root / asset.sidecar_relpath`` when both values are
    available and the file exists.  Returns ``None`` on any I/O or parse error
    so that the UI can gracefully degrade.
    """
    if takeout_root is None or asset.sidecar_relpath is None:
        return None
    sidecar_path = takeout_root / asset.sidecar_relpath
    try:
        raw = json.loads(sidecar_path.read_text(encoding="utf-8"))
        return json.dumps(raw, indent=2, ensure_ascii=False)
    except (OSError, json.JSONDecodeError):
        return None


def _read_exif_data(takeout_root: Path | None, asset: AssetRow) -> str | None:
    """Return a pretty-printed JSON string of all EXIF data for *asset*, or ``None``.

    Reads EXIF tags from the original image file using Pillow, including the
    main IFD, EXIF sub-IFD, and GPS sub-IFD.  Tag IDs are resolved to their
    human-readable names where possible.  Returns ``None`` on any error so
    that the UI can gracefully degrade.
    """
    if takeout_root is None or asset.relpath is None:
        return None
    image_path = takeout_root / asset.relpath
    if not image_path.exists():
        return None
    try:
        from PIL import ExifTags, Image  # noqa: PLC0415
    except ImportError:
        return None

    try:
        import pillow_heif  # noqa: PLC0415

        pillow_heif.register_heif_opener()
    except ImportError:
        pass

    def _make_serializable(value: object) -> object:
        """Recursively convert a value to a JSON-serialisable type."""
        if isinstance(value, bytes):
            # Try UTF-8 first; fall back to Latin-1 which guarantees every byte
            # round-trips without loss (unlike UTF-8, which may raise on arbitrary bytes).
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return value.decode("latin-1")
        if isinstance(value, (list, tuple)):
            return [_make_serializable(v) for v in value]
        if isinstance(value, dict):
            return {str(k): _make_serializable(v) for k, v in value.items()}
        # IFDRational (Pillow) → float; anything else that is numeric stays as-is.
        try:
            return float(value) if hasattr(value, "numerator") else value
        except Exception:  # noqa: BLE001
            return str(value)

    try:
        with Image.open(image_path) as img:
            exif = img.getexif()
            if not exif:
                return None

            data: dict[str, object] = {}

            # Main IFD tags.
            for tag_id, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, f"Tag_{tag_id:#06x}")
                data[tag_name] = _make_serializable(value)

            # EXIF sub-IFD.
            try:
                exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
                if exif_ifd:
                    exif_section: dict[str, object] = {}
                    for tag_id, value in exif_ifd.items():
                        tag_name = ExifTags.TAGS.get(tag_id, f"Tag_{tag_id:#06x}")
                        exif_section[tag_name] = _make_serializable(value)
                    data["ExifIFD"] = exif_section
            except Exception:  # noqa: BLE001
                pass

            # GPS sub-IFD.
            try:
                gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
                if gps_ifd:
                    gps_section: dict[str, object] = {}
                    for tag_id, value in gps_ifd.items():
                        tag_name = ExifTags.GPSTAGS.get(tag_id, f"Tag_{tag_id:#06x}")
                        gps_section[tag_name] = _make_serializable(value)
                    data["GPSIFD"] = gps_section
            except Exception:  # noqa: BLE001
                pass

            return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return None


def _parse_score(raw: str | None) -> float | None:
    """Parse a score query parameter, treating blank/missing values as ``None``.

    Silently ignores non-numeric values rather than raising a validation error,
    so that submitting an empty ``min_score=`` or ``max_score=`` form field
    never causes a 422 response.
    """
    if not raw or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_sort_by(sort_by: str | None) -> tuple[str, str, str] | None:
    """Parse a ``sort_by`` query parameter of the form ``scorer_id:variant_id:metric_key``.

    Returns a ``(scorer_id, variant_id, metric_key)`` triple, or ``None`` if
    *sort_by* is absent or malformed.
    """
    if not sort_by:
        return None
    parts = sort_by.split(":")
    if len(parts) != 3:
        return None
    scorer_id, variant_id, metric_key = parts
    if not scorer_id or not variant_id or not metric_key:
        return None
    return scorer_id, variant_id, metric_key


@router.get("/assets", response_class=HTMLResponse)
def browse_assets(
    request: Request,
    page: int = 1,
    favorited: str | None = None,
    sort_by: str | None = None,
    min_score: str | None = None,
    max_score: str | None = None,
    sort_by_2: str | None = None,
    min_score_2: str | None = None,
    max_score_2: str | None = None,
    sort_by_3: str | None = None,
    min_score_3: str | None = None,
    max_score_3: str | None = None,
    sort_by_4: str | None = None,
    min_score_4: str | None = None,
    max_score_4: str | None = None,
    dedupe: str = "1",
    partial: str = "0",
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> HTMLResponse:
    """Render the browse page with paginated asset thumbnails.

    Query parameters:
    - ``page``: 1-based page number (default 1).
    - ``favorited``: ``"1"`` to show only favorited assets.
    - ``sort_by``: ``"scorer_id:variant_id:metric_key"`` to sort by a score metric
      (e.g. ``"blur:default:sharpness"``).  Only scored assets are shown.
    - ``min_score`` / ``max_score``: Inclusive score range for the primary
      sort metric.  Blank or non-numeric values are silently ignored.
    - ``sort_by_2`` … ``sort_by_4``: Optional secondary / tertiary / quaternary
      sort metrics, each in ``"scorer_id:variant_id:metric_key"`` form.  Each can
      have its own ``min_score_N`` / ``max_score_N`` range filter.
      Secondary+ criteria use LEFT JOINs: assets without a secondary score still
      appear, sorted last.
    - ``dedupe``: ``"1"`` (default) to hide exact duplicate files (same SHA-256
      content hash); ``"0"`` to show all physical copies.
    - ``partial``: ``"1"`` to return only a card fragment (for infinite scroll
      fetch requests); ``"0"`` (default) to return the full page.
    """
    offset = max(0, (page - 1) * _PAGE_SIZE)
    fav_filter: bool | None = True if favorited == "1" else None
    dedup_enabled = dedupe != "0"

    sort_parsed = _parse_sort_by(sort_by)
    # Normalize sort_by: use the canonical form when valid, None when malformed.
    # This prevents untrusted query text from being propagated into template links.
    canonical_sort_by: str | None = None

    # Score map: asset_id → score value (populated when sorting by score)
    score_map: dict[int, float] = {}

    # Score range is only meaningful when sorting by score; parse safely to float.
    eff_min = _parse_score(min_score) if sort_parsed else None
    eff_max = _parse_score(max_score) if sort_parsed else None

    # Build additional sort criteria (levels 2–4).
    extra_raw = [
        (sort_by_2, min_score_2, max_score_2),
        (sort_by_3, min_score_3, max_score_3),
        (sort_by_4, min_score_4, max_score_4),
    ]
    extra_criteria: list[tuple[str, str | None, str | None]] = []
    # canonical forms for template use
    canonical_extra: list[tuple[str, float | None, float | None]] = []
    for sb, mn, mx in extra_raw:
        parsed = _parse_sort_by(sb)
        if parsed is None:
            break  # stop at the first empty/invalid level
        sid, vid, mkey = parsed
        eff_mn = _parse_score(mn)
        eff_mx = _parse_score(mx)
        extra_criteria.append((f"{sid}:{vid}:{mkey}", eff_mn, eff_mx))
        canonical_extra.append((f"{sid}:{vid}:{mkey}", eff_mn, eff_mx))

    # Determine whether we are in multi-sort mode.
    use_multi_sort = sort_parsed is not None and len(extra_criteria) > 0

    if sort_parsed is not None:
        scorer_id, variant_id, metric_key = sort_parsed
        canonical_sort_by = f"{scorer_id}:{variant_id}:{metric_key}"

        if use_multi_sort:
            criteria = [SortCriterion(scorer_id, variant_id, metric_key, eff_min, eff_max)]
            for sb_canon, eff_mn, eff_mx in extra_criteria:
                s_id, v_id, m_key = sb_canon.split(":")
                criteria.append(SortCriterion(s_id, v_id, m_key, eff_mn, eff_mx))

            asset_score_pairs = list_assets_multi_sort(
                conn,
                criteria,
                limit=_PAGE_SIZE,
                offset=offset,
                favorited=fav_filter,
            )
            assets = [a for a, _ in asset_score_pairs]
            score_map = {a.id: s for a, s in asset_score_pairs}
            total = count_assets_multi_sort(conn, criteria, favorited=fav_filter)
        else:
            asset_score_pairs = list_assets_by_score(
                conn,
                scorer_id,
                metric_key,
                variant_id=variant_id,
                limit=_PAGE_SIZE,
                offset=offset,
                favorited=fav_filter,
                min_score=eff_min,
                max_score=eff_max,
            )
            assets = [a for a, _ in asset_score_pairs]
            score_map = {a.id: s for a, s in asset_score_pairs}
            total = count_assets_with_score(
                conn,
                scorer_id,
                metric_key,
                variant_id=variant_id,
                favorited=fav_filter,
                min_score=eff_min,
                max_score=eff_max,
            )
    elif dedup_enabled:
        assets = list_assets_deduped(conn, limit=_PAGE_SIZE, offset=offset, favorited=fav_filter)
        total = count_assets_deduped(conn, favorited=fav_filter)
    else:
        assets = list_assets(conn, limit=_PAGE_SIZE, offset=offset, favorited=fav_filter)
        total = count_assets(conn, favorited=fav_filter)

    total_pages = max(1, (total + _PAGE_SIZE - 1) // _PAGE_SIZE)

    # Build sort options from registered scorer specs, but only include
    # metrics that actually have scored results from a completed run.
    available_triples = list_available_score_metrics_with_variants(conn)
    # Determine which (scorer_id, metric_key) combos have multiple variants scored —
    # those need the variant display name in the label to disambiguate.
    variants_per_scorer_metric: dict[tuple[str, str], int] = defaultdict(int)
    for sid, _vid, mk in available_triples:
        variants_per_scorer_metric[(sid, mk)] += 1

    spec_by_id = {spec.scorer_id: spec for spec in list_specs()}
    variant_by_key = {
        (spec.scorer_id, v.variant_id): v for spec in list_specs() for v in spec.variants
    }
    sort_options = []
    for scorer_id, variant_id, metric_key in sorted(available_triples):
        spec = spec_by_id.get(scorer_id)
        if spec is None:
            continue
        metric = next((m for m in spec.metrics if m.key == metric_key), None)
        if metric is None:
            continue
        value = f"{scorer_id}:{variant_id}:{metric_key}"
        if variants_per_scorer_metric.get((scorer_id, metric_key), 0) > 1:
            v_spec = variant_by_key.get((scorer_id, variant_id))
            v_name = v_spec.display_name if v_spec else variant_id
            label = f"{spec.display_name} – {v_name} – {metric.display_name}"
        else:
            label = f"{spec.display_name} – {metric.display_name}"
        sort_options.append((value, label))

    # Load saved presets for the toolbar
    presets = list_view_presets(conn)

    # When sorting by score, also fetch the total indexed count so the
    # template can distinguish "no indexed photos" from "no scored photos".
    total_indexed = count_assets(conn) if sort_parsed is not None and total == 0 else None

    templates = request.app.state.templates
    if partial == "1":
        return templates.TemplateResponse(
            "browse_partial.html",
            {
                "request": request,
                "assets": assets,
                "page": page,
                "total_pages": total_pages,
                "score_map": score_map,
            },
        )
    return templates.TemplateResponse(
        "browse.html",
        {
            "request": request,
            "assets": assets,
            "page": page,
            "total_pages": total_pages,
            "total": total,
            "total_indexed": total_indexed,
            "favorited": favorited,
            "sort_by": canonical_sort_by,
            "sort_options": sort_options,
            "score_map": score_map,
            "min_score": eff_min,
            "max_score": eff_max,
            # Extra sort criteria passed as a list of (sort_by, min, max) tuples
            # so the template can render them without index-specific variable names.
            "extra_sort_criteria": canonical_extra,
            "presets": presets,
            # dedupe is only actively applied when not sorting by score metric
            "dedupe": dedup_enabled and sort_parsed is None,
        },
    )


@router.get("/assets/{asset_id}", response_class=HTMLResponse)
def asset_detail(
    asset_id: int,
    request: Request,
    partial: str = "0",
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
    takeout_root: Path | None = Depends(_get_takeout_root),  # noqa: B008
) -> HTMLResponse:
    """Render the detail page for a single asset.

    Query parameters:
    - ``partial``: ``"1"`` to return only the detail fragment (for the
      lightbox panel); ``"0"`` (default) to return the full page.
    """
    asset: AssetRow | None = get_asset_by_id(conn, asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail=f"Asset {asset_id} not found")

    scores = get_asset_scores(conn, asset_id)

    # Collect alias paths — other locations in the Takeout archive where this
    # exact binary file also appears (stored in asset_paths after dedup).
    alias_paths: list[str] = get_asset_alias_paths(conn, asset_id)

    # Load raw sidecar JSON for the main asset.
    sidecar_json = _read_sidecar_json(takeout_root, asset)

    # Load EXIF data from the original image file.
    exif_data = _read_exif_data(takeout_root, asset)

    # Load pHash for the asset (used in both partial and full views).
    phash_row = get_phash(conn, asset_id)
    phash_hex: str | None = phash_row["phash_hex"] if phash_row else None

    templates = request.app.state.templates
    ctx = {
        "request": request,
        "asset": asset,
        "scores": scores,
        "alias_paths": alias_paths,
        "sidecar_json": sidecar_json,
        "exif_data": exif_data,
        "phash_hex": phash_hex,
    }
    if partial == "1":
        return templates.TemplateResponse("detail_partial.html", ctx)
    return templates.TemplateResponse("detail.html", ctx)


# ---------------------------------------------------------------------------
# CLIP vocabulary matrix cache
# ---------------------------------------------------------------------------

_VOCAB_MATRIX_LOCK = __import__("threading").Lock()
_USER_TAGS_MATRIX_LOCK = __import__("threading").Lock()


def _get_clip_vocab_matrix(
    request: Request,
) -> tuple[__import__("numpy").ndarray, list[str]] | None:  # type: ignore[name-defined]
    """Return the cached ``(vocab_matrix, vocab_terms)`` pair.

    The matrix is built lazily on the first call and stored in
    ``app.state.clip_vocab_matrix``.  Each row is the L2-normalised CLIP text
    embedding for one vocabulary term (shape ``(V, 768)``, float32).

    Returns ``None`` if the CLIP backbone dependencies are not installed.
    """
    cached = getattr(request.app.state, "clip_vocab_matrix", None)
    if cached is not None:
        return cached

    with _VOCAB_MATRIX_LOCK:
        cached = getattr(request.app.state, "clip_vocab_matrix", None)
        if cached is not None:
            return cached

        try:
            import numpy as np  # noqa: PLC0415
            import torch  # noqa: PLC0415

            from takeout_rater.scorers.adapters.clip_backbone import (  # noqa: PLC0415
                get_clip_model,
            )
            from takeout_rater.scorers.adapters.clip_vocab import (  # noqa: PLC0415
                CLIP_VOCAB_TERMS,
            )
        except ImportError:
            return None

        model, _preprocess, tokenizer, device = get_clip_model()
        terms = list(CLIP_VOCAB_TERMS)

        # Encode in small batches to avoid OOM on large vocabularies
        batch_size = 64
        vecs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(terms), batch_size):
                batch = terms[start : start + batch_size]
                tokens = tokenizer(batch).to(device)
                features = model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                vecs.append(features.cpu().float().numpy())

        matrix = np.vstack(vecs).astype(np.float32)  # shape (V, 768)
        result = (matrix, terms)
        request.app.state.clip_vocab_matrix = result
        return result


def _get_user_tags_matrix(
    request: Request,
    user_tags: list[str],
) -> tuple[__import__("numpy").ndarray, list[str]] | None:  # type: ignore[name-defined]
    """Return the cached ``(matrix, terms)`` pair for user-defined CLIP tags.

    The matrix is built lazily and stored in ``app.state.clip_user_tags_matrix``
    as a ``(frozenset_of_terms, matrix, terms_list)`` tuple.  The cache is
    invalidated (set to ``None``) by the CLIP tag management API whenever tags
    are added or removed.

    Returns ``None`` when *user_tags* is empty or CLIP is not available.
    """
    if not user_tags:
        return None

    tags_key = frozenset(user_tags)

    cached = getattr(request.app.state, "clip_user_tags_matrix", None)
    if cached is not None:
        cached_key, cached_matrix, cached_terms = cached
        if cached_key == tags_key:
            return (cached_matrix, cached_terms)

    with _USER_TAGS_MATRIX_LOCK:
        cached = getattr(request.app.state, "clip_user_tags_matrix", None)
        if cached is not None:
            cached_key, cached_matrix, cached_terms = cached
            if cached_key == tags_key:
                return (cached_matrix, cached_terms)

        try:
            import numpy as np  # noqa: PLC0415
            import torch  # noqa: PLC0415

            from takeout_rater.scorers.adapters.clip_backbone import (  # noqa: PLC0415
                get_clip_model,
            )
        except ImportError:
            return None

        model, _preprocess, tokenizer, device = get_clip_model()
        terms = list(user_tags)

        batch_size = 64
        vecs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(terms), batch_size):
                batch = terms[start : start + batch_size]
                tokens = tokenizer(batch).to(device)
                features = model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                vecs.append(features.cpu().float().numpy())

        matrix = np.vstack(vecs).astype(np.float32)
        request.app.state.clip_user_tags_matrix = (tags_key, matrix, terms)
        return (matrix, terms)


@router.get("/api/assets/{asset_id}/clip-words")
def get_clip_words(
    asset_id: int,
    request: Request,
    top_k: int = 20,
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse:
    """Return the top-K visual concepts matching this asset's CLIP embedding.

    Compares the stored image embedding against a curated vocabulary of visual
    concept phrases (predefined terms plus any user-defined tags) using cosine
    similarity and returns the best matches.

    Query parameters:
    - ``top_k``: Number of top matches to return (default 20, max 100).

    Returns JSON ``{"words": [{"word": str, "score": float, "user_tag": bool}, ...]}``
    or ``{"words": [], "error": "no_embedding"}`` when no embedding is stored.
    """
    import struct  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    top_k = max(1, min(top_k, 100))

    blob = get_clip_embedding_for_asset(conn, asset_id)
    if blob is None:
        return JSONResponse({"words": [], "error": "no_embedding"})

    vocab = _get_clip_vocab_matrix(request)
    if vocab is None:
        return JSONResponse({"words": [], "error": "clip_unavailable"})

    vocab_matrix, vocab_terms = vocab

    dim = 768
    expected_bytes = dim * 4  # float32 = 4 bytes each
    if len(blob) != expected_bytes:
        return JSONResponse({"words": [], "error": "embedding_corrupt"})

    image_vec = np.array(struct.unpack(f"{dim}f", blob), dtype=np.float32)
    norm = np.linalg.norm(image_vec)
    if norm > 0:
        image_vec = image_vec / norm

    # Build combined matrix: predefined vocab + user-defined tags
    user_tags = list_clip_user_tags(conn)
    user_tag_set = set(user_tags)
    # Filter out user tags that duplicate predefined terms to avoid overlap
    novel_user_tags = [t for t in user_tags if t not in set(vocab_terms)]

    user_matrix_result = _get_user_tags_matrix(request, novel_user_tags)

    if user_matrix_result is not None:
        user_matrix, user_terms = user_matrix_result
        combined_matrix = np.vstack([vocab_matrix, user_matrix])
        combined_terms = vocab_terms + user_terms
    else:
        combined_matrix = vocab_matrix
        combined_terms = vocab_terms

    scores = combined_matrix @ image_vec  # shape (V,)
    top_k = min(top_k, len(combined_terms))
    ranked = np.argsort(-scores)[:top_k]

    words = [
        {
            "word": combined_terms[int(i)],
            "score": round(float(scores[i]), 4),
            "user_tag": combined_terms[int(i)] in user_tag_set,
        }
        for i in ranked
    ]
    return JSONResponse({"words": words})


@router.get("/thumbs/{asset_id}")
def serve_thumbnail(
    asset_id: int,
    request: Request,
    thumbs_dir: Path = Depends(_get_thumbs_dir),  # noqa: B008
) -> FileResponse:
    """Serve the JPEG thumbnail for an asset.

    Returns 404 if the thumbnail has not been generated yet.
    """
    thumb = thumb_path_for_id(thumbs_dir, asset_id)
    if not thumb.exists():
        raise HTTPException(status_code=404, detail=f"Thumbnail for asset {asset_id} not found")
    return FileResponse(str(thumb), media_type="image/jpeg")


@router.get("/api/timeline")
def get_timeline(
    request: Request,
    favorited: str | None = None,
    dedupe: str = "1",
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse:
    """Return the year range of photos available in the library.

    Query parameters mirror those of ``/assets`` so the timeline always reflects
    the currently active view filter.

    - ``favorited``: ``"1"`` to restrict to favorited assets.
    - ``dedupe``: ``"0"`` to include duplicate files; defaults to ``"1"``.

    Returns a JSON object with:
    - ``has_data``: ``false`` when no photos have ``taken_at`` metadata.
    - ``min_year``: integer – oldest year with photos.
    - ``max_year``: integer – newest year with photos.
    """
    min_ts, max_ts = get_taken_at_range(conn)
    if min_ts is None or max_ts is None:
        return JSONResponse({"has_data": False})

    min_year = datetime.datetime.fromtimestamp(min_ts, tz=datetime.UTC).year
    max_year = datetime.datetime.fromtimestamp(max_ts, tz=datetime.UTC).year

    return JSONResponse({"has_data": True, "min_year": min_year, "max_year": max_year})


@router.get("/api/timeline/seek")
def timeline_seek(
    request: Request,
    timestamp: int,
    favorited: str | None = None,
    dedupe: str = "1",
    conn: sqlite3.Connection = Depends(_get_conn),  # noqa: B008
) -> JSONResponse:
    """Return the 1-based page number for the given Unix timestamp.

    Assets are sorted by ``taken_at DESC`` (newest first), so this counts the
    number of assets *newer* than *timestamp* and divides by the page size.

    Query parameters:
    - ``timestamp``: Unix timestamp (seconds) to seek to.
    - ``favorited``: ``"1"`` to restrict to favorited assets.
    - ``dedupe``: ``"0"`` to include duplicate files; defaults to ``"1"``.

    Returns ``{"page": N}`` where *N* is the 1-based page that contains photos
    taken around the requested time.
    """
    fav_filter: bool | None = True if favorited == "1" else None
    dedup_enabled = dedupe != "0"

    count = count_assets_newer_than(
        conn,
        timestamp,
        favorited=fav_filter,
        deduped=dedup_enabled,
    )
    page = max(1, count // _PAGE_SIZE + 1)
    return JSONResponse({"page": page})
