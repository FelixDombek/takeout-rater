"""Split a trip or event into story chapters where the photos change subject.

A trip's photos are walked in time order.  At every gap between two adjacent
photos we compare the mean CLIP embedding of the few photos before the gap
with the few photos after it; a large cosine distance means the scene changed
(beach → old town → dinner).  Long pauses in time amplify that score, so a
night's sleep or a long drive tips an otherwise borderline change, but a pause
alone never starts a chapter (unless the photos have no embeddings at all).

Chapters start at the strongest of those breaks, as long as each break stands
out from the trip's own typical change (median + k·MAD, so a trip shot in one
place and a trip through five countries both get sensible chapters) and every
chapter keeps a minimum number of photos.

Everything here is pure numpy and runs per request: a 2,000-photo trip takes a
few milliseconds, so chapters need no table and follow the sensitivity the user
picks.  Labels come from the zero-shot CLIP vocabulary when it is available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from takeout_rater.events.detector import _utc_offset_s

_HOUR = 3600
_DAY = 86400
#: Without embeddings, a gap must be a pause of a few hours to start a chapter.
_TIME_ONLY_MIN_SCORE = 0.6


@dataclass(frozen=True)
class ChapterParams:
    """Tuning knobs for :func:`split_chapters`."""

    #: Photos on each side of a gap that are averaged before comparing.
    window: int = 3
    #: A break must exceed ``median + mad_k * MAD`` of the trip's break scores.
    mad_k: float = 1.5
    #: Absolute floor on the break score, so a trip with almost no variation
    #: is not chopped up just because some gap is "relatively" large.
    min_score: float = 0.12
    #: Smallest chapter, in photos.  ``None`` scales with the trip size.
    min_photos: int | None = None
    #: Hard cap on the number of chapters.
    max_chapters: int = 40
    #: How much a long pause amplifies a change of scene: the score is
    #: multiplied by up to ``1 + time_weight`` (reached at ``time_full_h``).
    time_weight: float = 0.3
    time_full_h: float = 12.0


#: Presets offered in the UI.
SENSITIVITY: dict[str, ChapterParams] = {
    "fewer": ChapterParams(mad_k=2.5, min_score=0.16),
    "normal": ChapterParams(),
    "more": ChapterParams(mad_k=0.75, min_score=0.08),
}
_MIN_PHOTOS_SCALE = {"fewer": 2.0, "normal": 1.0, "more": 0.5}


def params_for(sensitivity: str) -> ChapterParams:
    """Return the preset for *sensitivity* (``fewer``/``normal``/``more``)."""
    return SENSITIVITY.get(sensitivity, SENSITIVITY["normal"])


def default_min_photos(n: int, sensitivity: str = "normal") -> int:
    """Minimum chapter size for a trip of *n* photos (4 for small trips, ~n/50 for big ones)."""
    base = max(4, round(n / 50))
    return max(2, round(base * _MIN_PHOTOS_SCALE.get(sensitivity, 1.0)))


@dataclass
class Chapter:
    """A run of consecutive photos (indexes into the time-sorted input)."""

    start: int
    end: int  # exclusive
    asset_ids: list[int]
    start_ts: int
    end_ts: int
    #: Score of the break that opened this chapter (``None`` for the first one).
    break_score: float | None = None
    cover_asset_id: int | None = None
    centroid: Any = field(default=None, repr=False)  # unit-length vector or None


@dataclass
class StoryResult:
    """Chapters plus the per-gap break curve (``len == n - 1``) for plotting."""

    chapters: list[Chapter]
    scores: list[float]
    threshold: float
    n_with_embedding: int
    n_photos: int


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def _fill_missing(
    asset_ids: list[int], embeddings: dict[int, np.ndarray]
) -> tuple[np.ndarray | None, int]:
    """Stack embeddings in order; photos without one borrow their nearest earlier neighbour.

    Returns ``(matrix, n_with_embedding)``; ``matrix`` is ``None`` when no photo has one.
    """
    present = [i for i, a in enumerate(asset_ids) if a in embeddings]
    if not present:
        return None, 0
    dim = len(embeddings[asset_ids[present[0]]])
    out = np.empty((len(asset_ids), dim), dtype=np.float32)
    last = embeddings[asset_ids[present[0]]]  # leading gaps borrow from the first one
    for i, a in enumerate(asset_ids):
        vec = embeddings.get(a)
        if vec is not None:
            last = vec
        out[i] = last
    return _normalise(out), len(present)


def break_scores(
    ts: list[int], matrix: np.ndarray | None, params: ChapterParams = SENSITIVITY["normal"]
) -> np.ndarray:
    """Score each of the ``n - 1`` gaps; higher means a stronger change of scene.

    The semantic part is ``1 - cos(mean(before), mean(after))`` over up to
    ``params.window`` photos per side.  It is multiplied by
    ``1 + time_weight * pause``, where ``pause`` grows logarithmically with the
    time between the two photos and reaches 1 at ``params.time_full_h``.
    Without embeddings (``matrix is None``) the score is ``pause`` alone.
    """
    n = len(ts)
    if n < 2:
        return np.zeros(0, dtype=np.float32)
    sem = np.zeros(n - 1, dtype=np.float32)
    if matrix is not None:
        k = max(1, params.window)
        csum = np.vstack([np.zeros((1, matrix.shape[1]), np.float32), np.cumsum(matrix, axis=0)])
        idx = np.arange(n - 1)
        # Same number of photos on both sides, so gaps near the ends (where a
        # single photo would otherwise face a whole window) are not noisier.
        kk = np.minimum(k, np.minimum(idx + 1, n - idx - 1))
        lo = idx + 1 - kk
        hi = idx + 1 + kk
        before = (csum[idx + 1] - csum[lo]) / (idx + 1 - lo)[:, None]
        after = (csum[hi] - csum[idx + 1]) / (hi - idx - 1)[:, None]
        sem = 1.0 - np.sum(_normalise(before) * _normalise(after), axis=1)
    gaps_h = np.maximum(0, np.diff(np.asarray(ts, dtype=np.float64))) / _HOUR
    pause = np.minimum(1.0, np.log1p(gaps_h) / math.log1p(params.time_full_h))
    if matrix is None:
        # No embeddings at all: fall back to pauses alone.
        return pause.astype(np.float32)
    # A pause amplifies a change of scene but never creates one by itself, so a
    # night's sleep in the same city does not start a chapter.
    return (sem * (1.0 + params.time_weight * pause)).astype(np.float32)


def _pick_cover(
    ids: list[int], vecs: np.ndarray | None, centroid: np.ndarray | None, scores: dict[int, float]
) -> int:
    """A photo that represents the chapter: of the ~5 closest to its centroid, the best-scored."""
    if vecs is None or centroid is None:
        cands = ids
    else:
        sims = vecs @ centroid
        order = np.argsort(-sims)[: max(1, min(5, len(ids)))]
        cands = [ids[i] for i in order]
    scored = [a for a in cands if a in scores]
    if scored:
        return max(scored, key=lambda a: scores[a])
    return cands[0] if vecs is not None else ids[len(ids) // 2]


def split_chapters(
    photos: list[tuple[int, int]],
    embeddings: dict[int, np.ndarray],
    params: ChapterParams = SENSITIVITY["normal"],
    *,
    min_photos: int | None = None,
    aesthetic: dict[int, float] | None = None,
) -> StoryResult:
    """Split time-ordered photos into chapters at the strongest changes of scene.

    Args:
        photos: ``(asset_id, taken_at)`` pairs, in any order.
        embeddings: ``{asset_id: vector}``; photos without one inherit their
            previous neighbour's, so they never cause a break by themselves.
        params: Tuning knobs (see :data:`SENSITIVITY`).
        min_photos: Smallest chapter; default from :func:`default_min_photos`.
        aesthetic: Optional ``{asset_id: score}`` used to pick chapter covers.
    """
    photos = sorted(photos, key=lambda p: (p[1], p[0]))
    ids = [p[0] for p in photos]
    ts = [p[1] for p in photos]
    n = len(ids)
    matrix, n_emb = _fill_missing(ids, embeddings)
    scores = break_scores(ts, matrix, params)
    min_size = min_photos or params.min_photos or default_min_photos(n)

    threshold = params.min_score
    if len(scores):
        med = float(np.median(scores))
        mad = float(np.median(np.abs(scores - med))) * 1.4826
        threshold = max(params.min_score, med + params.mad_k * mad)
        if matrix is None:
            # Pauses alone: only gaps of a few hours count (0.6 ≈ 3.5 h).
            threshold = max(threshold, _TIME_ONLY_MIN_SCORE)

    # Greedy: strongest breaks first, keep one only if it is the peak of its
    # neighbourhood (the windows smear one change over a few gaps) and both
    # sides stay big enough.
    cuts: list[int] = []  # gap index g means a chapter starts at photo g + 1
    if n >= 2 * min_size:
        r = max(1, params.window)
        for g in np.argsort(-scores, kind="stable"):
            g = int(g)
            # Strictly above, so a flat curve (MAD 0) stays one chapter.
            if scores[g] <= threshold or len(cuts) + 1 >= params.max_chapters:
                break
            if scores[g] < scores[max(0, g - r) : g + r + 1].max():
                continue
            start = g + 1
            left = max((c + 1 for c in cuts if c + 1 <= start), default=0)
            right = min((c + 1 for c in cuts if c + 1 > start), default=n)
            if start - left >= min_size and right - start >= min_size:
                cuts.append(g)
    cuts.sort()

    aesthetic = aesthetic or {}
    bounds = [0, *(c + 1 for c in cuts), n]
    chapters: list[Chapter] = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if a >= b:
            continue
        vecs = matrix[a:b] if matrix is not None else None
        centroid = None
        if vecs is not None:
            c = vecs.mean(axis=0)
            norm = float(np.linalg.norm(c))
            centroid = c / norm if norm > 0 else None
        chapters.append(
            Chapter(
                start=a,
                end=b,
                asset_ids=ids[a:b],
                start_ts=ts[a],
                end_ts=ts[b - 1],
                break_score=float(scores[a - 1]) if a > 0 else None,
                cover_asset_id=_pick_cover(ids[a:b], vecs, centroid, aesthetic),
                centroid=centroid,
            )
        )
    return StoryResult(
        chapters=chapters,
        scores=[round(float(s), 4) for s in scores],
        threshold=round(float(threshold), 4),
        n_with_embedding=n_emb,
        n_photos=n,
    )


# ── Labels and display ────────────────────────────────────────────────────────


def label_terms() -> set[str]:
    """Vocabulary terms that make good chapter titles (places, things, activities)."""
    from takeout_rater.scoring.scorers import clip_vocab as v

    skip = {"documentary", "abstract", "product photography", "fashion", "nature", "landscape"}
    terms = v._SCENES + v._ACTIVITIES + v._OBJECTS + v._PHOTOGRAPHY_SUBJECTS
    return {t for t in terms if t not in skip}


def label_chapters(
    chapters: list[Chapter],
    vocab: np.ndarray,
    terms: list[str],
    *,
    allowed: set[str] | None = None,
    n_terms: int = 2,
    shortlist: int = 12,
) -> list[list[str]]:
    """Name each chapter by the vocabulary terms that set it apart from the rest of the trip.

    For every chapter the ``shortlist`` terms closest to its centroid are
    re-ranked by how much closer they are to this chapter than to the trip as
    a whole, so a beach trip's chapters read "harbour", "dinner"… rather than
    "beach" every time.  Chapters without a centroid get no terms.
    """
    keep = np.array([allowed is None or t in allowed for t in terms])
    if not keep.any():
        return [[] for _ in chapters]
    vocab = vocab[keep]
    kept_terms = [t for t, k in zip(terms, keep, strict=True) if k]
    cents = [c.centroid for c in chapters if c.centroid is not None]
    if not cents:
        return [[] for _ in chapters]
    weights = np.array([len(c.asset_ids) for c in chapters if c.centroid is not None], np.float32)
    trip = (np.vstack(cents) * weights[:, None]).sum(axis=0)
    trip /= max(float(np.linalg.norm(trip)), 1e-9)
    trip_sims = vocab @ trip
    out: list[list[str]] = []
    for ch in chapters:
        if ch.centroid is None:
            out.append([])
            continue
        sims = vocab @ ch.centroid
        top = np.argsort(-sims)[:shortlist]
        ranked = sorted(top, key=lambda i: (-(sims[i] - trip_sims[i]), -sims[i]))
        out.append([kept_terms[i] for i in ranked[:n_terms]])
    return out


def chapter_title(terms: list[str]) -> str:
    """``["beach", "swimming"]`` → ``"Beach & swimming"``."""
    if not terms:
        return ""
    text = " & ".join(terms)
    return text[0].upper() + text[1:]


def time_label(
    start_ts: int, end_ts: int, trip_start_ts: int, lon: float | None, *, show_day: bool = True
) -> str:
    """Local-time span of a chapter.

    Examples: ``"Day 2 · 09:10–12:40"``, ``"Day 2 18:00 – Day 3 10:15"``,
    ``"Days 3–5"``, or
    just ``"09:10–12:40"`` with ``show_day=False`` (single-day events).
    """
    off = _utc_offset_s(lon)
    day0 = (trip_start_ts + off) // _DAY
    a = datetime.fromtimestamp(start_ts + off, tz=UTC)
    b = datetime.fromtimestamp(end_ts + off, tz=UTC)
    da = (start_ts + off) // _DAY - day0 + 1
    db = (end_ts + off) // _DAY - day0 + 1
    if da == db:
        span = a.strftime("%H:%M") if start_ts == end_ts else f"{a:%H:%M}–{b:%H:%M}"
        return f"Day {da} · {span}" if show_day else span
    if db - da == 1:
        return f"Day {da} {a:%H:%M} – Day {db} {b:%H:%M}"
    return f"Days {da}–{db}"
