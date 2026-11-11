"""Compute a "mood timeline" from stored image embeddings.

Each mood word (``"happy"``, ``"peaceful"`` …) is turned into a text embedding
with the same backbone the image embeddings came from (CLIP ViT-L/14 or
SigLIP 2), averaged over a few prompt templates.  Every photo is then
classified zero-shot against the whole mood set: the cosine similarities are
turned into a softmax distribution, so each photo spreads one unit of "mood"
over the words.  Averaging those distributions per time bucket gives the
*share* of each mood in that period, which is comparable across periods with
very different photo counts.

The *lift* of a mood in a bucket is its share there divided by its share in
the whole library: ``1.5`` means "50 % more of this mood than usual".

Everything here is plain numpy so it can be tested without a model.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_MOODS: tuple[str, ...] = (
    "happy",
    "peaceful",
    "energetic",
    "romantic",
    "nostalgic",
    "melancholic",
    "adventurous",
    "cozy",
)

#: Prompt templates averaged into one text embedding per mood.
PROMPT_TEMPLATES: tuple[str, ...] = (
    "a {} photo.",
    "a photo that feels {}.",
    "a photo with a {} mood.",
)

#: Softmax temperature; CLIP's learned logit scale is 100.
SOFTMAX_SCALE = 100.0

MAX_MOODS = 8  # one per categorical chart colour
MAX_MOOD_LENGTH = 60

GRANULARITIES: tuple[str, ...] = ("week", "month", "quarter", "year")

_WEEK_SECONDS = 7 * 86400
# 1970-01-01 was a Thursday; shifting by 3 days makes weeks start on Monday.
_MONDAY_OFFSET = 3 * 86400


def normalize_moods(raw: Iterable[str]) -> list[str]:
    """Clean a user-supplied mood list.

    Strips whitespace, drops empty and duplicate (case-insensitive) entries,
    truncates overly long words and keeps at most :data:`MAX_MOODS`.
    """
    seen: set[str] = set()
    moods: list[str] = []
    for item in raw:
        mood = " ".join(str(item).split())[:MAX_MOOD_LENGTH]
        key = mood.casefold()
        if not mood or key in seen:
            continue
        seen.add(key)
        moods.append(mood)
        if len(moods) >= MAX_MOODS:
            break
    return moods


def mood_prompts(mood: str) -> list[str]:
    """Return the prompts used to embed *mood*."""
    return [template.format(mood) for template in PROMPT_TEMPLATES]


def mood_text_matrix(moods: Sequence[str], encode: Callable[[str], np.ndarray]) -> np.ndarray:
    """Embed each mood (prompt-ensembled) into an L2-normalised ``(k, dim)`` matrix."""
    rows = []
    for mood in moods:
        vecs = np.stack([np.asarray(encode(p), dtype=np.float32) for p in mood_prompts(mood)])
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        mean = vecs.mean(axis=0)
        rows.append(mean / (np.linalg.norm(mean) + 1e-12))
    return np.stack(rows).astype(np.float32)


def mood_probabilities(
    embeddings: np.ndarray, text_matrix: np.ndarray, scale: float = SOFTMAX_SCALE
) -> np.ndarray:
    """Zero-shot classify every photo against the moods.

    Args:
        embeddings: ``(n, dim)`` L2-normalised image embeddings.
        text_matrix: ``(k, dim)`` L2-normalised mood embeddings.
        scale: Softmax temperature applied to the cosine similarities.

    Returns:
        ``(n, k)`` float32 matrix whose rows sum to 1.
    """
    n = embeddings.shape[0]
    k = text_matrix.shape[0]
    out = np.empty((n, k), dtype=np.float32)
    chunk = 65536  # bound the temporary memory on very large libraries
    for start in range(0, n, chunk):
        logits = (embeddings[start : start + chunk] @ text_matrix.T) * scale
        logits -= logits.max(axis=1, keepdims=True)
        np.exp(logits, out=logits)
        logits /= logits.sum(axis=1, keepdims=True)
        out[start : start + chunk] = logits
    return out


@dataclass(frozen=True)
class Buckets:
    """Assignment of photos to time buckets."""

    index: np.ndarray  # (n,) bucket position of each photo, 0..len(keys)-1
    keys: list[str]  # bucket labels, e.g. "2021-03", in time order
    starts: list[int]  # Unix start of each bucket (UTC)
    ends: list[int]  # Unix end (exclusive) of each bucket (UTC)


def _ts(d: _dt.date) -> int:
    return int(_dt.datetime(d.year, d.month, d.day, tzinfo=_dt.UTC).timestamp())


def _bucket_ids(taken_at: np.ndarray, granularity: str) -> np.ndarray:
    """Map Unix timestamps to monotonically increasing integer bucket ids."""
    if granularity == "week":
        return (taken_at + _MONDAY_OFFSET) // _WEEK_SECONDS
    months = taken_at.astype("datetime64[s]").astype("datetime64[M]").astype(np.int64)
    if granularity == "month":
        return months
    if granularity == "quarter":
        return months // 3
    if granularity == "year":
        return months // 12
    raise ValueError(f"Unknown granularity: {granularity!r}")


def _bucket_bounds(bucket_id: int, granularity: str) -> tuple[str, int, int]:
    """Return ``(label, start_ts, end_ts)`` for one integer bucket id."""
    if granularity == "week":
        start = bucket_id * _WEEK_SECONDS - _MONDAY_OFFSET
        day = _dt.datetime.fromtimestamp(start, tz=_dt.UTC).date()
        return day.isoformat(), start, start + _WEEK_SECONDS
    if granularity == "month":
        y, m = divmod(bucket_id, 12)
        y += 1970
        start_d = _dt.date(y, m + 1, 1)
        end_d = _dt.date(y + (m + 1) // 12, (m + 1) % 12 + 1, 1)
        return f"{y:04d}-{m + 1:02d}", _ts(start_d), _ts(end_d)
    if granularity == "quarter":
        y, q = divmod(bucket_id, 4)
        y += 1970
        start_d = _dt.date(y, q * 3 + 1, 1)
        end_d = _dt.date(y + 1, 1, 1) if q == 3 else _dt.date(y, q * 3 + 4, 1)
        return f"{y:04d}-Q{q + 1}", _ts(start_d), _ts(end_d)
    y = bucket_id + 1970
    return f"{y:04d}", _ts(_dt.date(y, 1, 1)), _ts(_dt.date(y + 1, 1, 1))


def assign_buckets(taken_at: np.ndarray, granularity: str) -> Buckets:
    """Group photos into contiguous time buckets (empty buckets in between included)."""
    taken_at = np.asarray(taken_at, dtype=np.int64)
    if taken_at.size == 0:
        return Buckets(np.empty(0, dtype=np.int64), [], [], [])
    ids = _bucket_ids(taken_at, granularity)
    first = int(ids.min())
    last = int(ids.max())
    keys: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for bucket_id in range(first, last + 1):
        label, start, end = _bucket_bounds(bucket_id, granularity)
        keys.append(label)
        starts.append(start)
        ends.append(end)
    return Buckets(index=(ids - first).astype(np.int64), keys=keys, starts=starts, ends=ends)


def mood_timeline(
    probs: np.ndarray,
    taken_at: np.ndarray,
    moods: Sequence[str],
    granularity: str = "month",
) -> dict:
    """Aggregate per-photo mood probabilities into a timeline.

    Args:
        probs: ``(n, k)`` per-photo mood distribution from :func:`mood_probabilities`.
        taken_at: ``(n,)`` Unix timestamps aligned with *probs*.
        moods: The *k* mood words.
        granularity: One of :data:`GRANULARITIES`.

    Returns:
        A JSON-ready dict with ``moods``, ``granularity``, ``overall``
        (``count``, ``share``) and ``buckets``; each bucket has ``key``,
        ``start``, ``end``, ``count``, ``share`` and ``lift`` (``None`` values
        for empty buckets).
    """
    k = len(moods)
    buckets = assign_buckets(taken_at, granularity)
    n_buckets = len(buckets.keys)
    counts = np.bincount(buckets.index, minlength=n_buckets) if n_buckets else np.zeros(0)
    sums = np.zeros((n_buckets, k), dtype=np.float64)
    if probs.shape[0]:
        np.add.at(sums, buckets.index, probs.astype(np.float64))
    overall_share = (
        probs.astype(np.float64).mean(axis=0) if probs.shape[0] else np.zeros(k, dtype=np.float64)
    )

    out_buckets = []
    for i in range(n_buckets):
        count = int(counts[i])
        if count:
            share = sums[i] / count
            lift = np.divide(
                share, overall_share, out=np.zeros_like(share), where=overall_share > 0
            )
            share_list: list[float | None] = [round(float(v), 4) for v in share]
            lift_list: list[float | None] = [round(float(v), 3) for v in lift]
        else:
            share_list = [None] * k
            lift_list = [None] * k
        out_buckets.append(
            {
                "key": buckets.keys[i],
                "start": buckets.starts[i],
                "end": buckets.ends[i],
                "count": count,
                "share": share_list,
                "lift": lift_list,
            }
        )
    return {
        "moods": list(moods),
        "granularity": granularity,
        "overall": {
            "count": int(probs.shape[0]),
            "share": [round(float(v), 4) for v in overall_share],
        },
        "buckets": out_buckets,
    }


def top_photos(
    probs: np.ndarray,
    asset_ids: np.ndarray,
    taken_at: np.ndarray,
    mood_index: int,
    *,
    start: int | None = None,
    end: int | None = None,
    limit: int = 12,
) -> list[dict]:
    """Return the photos that lean most towards one mood, optionally within ``[start, end)``."""
    mask = np.ones(asset_ids.shape[0], dtype=bool)
    if start is not None:
        mask &= taken_at >= start
    if end is not None:
        mask &= taken_at < end
    idx = np.flatnonzero(mask)
    if idx.size == 0 or limit <= 0:
        return []
    scores = probs[idx, mood_index]
    part = np.argpartition(-scores, limit - 1)[:limit] if idx.size > limit else np.arange(idx.size)
    order = part[np.argsort(-scores[part], kind="stable")]
    return [
        {
            "asset_id": int(asset_ids[idx[j]]),
            "score": round(float(scores[j]), 4),
            "taken_at": int(taken_at[idx[j]]),
        }
        for j in order
    ]
