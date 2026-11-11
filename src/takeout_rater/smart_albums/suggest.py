"""Cluster CLIP image embeddings into album suggestions and name them.

Everything here is pure NumPy / scikit-learn so it can be tested without the
CLIP model.  :mod:`takeout_rater.smart_albums.service` loads the embeddings,
encodes the vocabulary with the CLIP text encoder and stores the result.

Algorithm
---------
1. **Cluster** the L2-normalised 768-D CLIP embeddings with K-Means
   (MiniBatchKMeans above :data:`_MINIBATCH_THRESHOLD` photos).  Clustering
   happens in the full embedding space, not in the 3-D UMAP projection of the
   embedding map, because the projection distorts distances and would need
   UMAP to run first (minutes on a large library).
2. **Name** each cluster: the centroid is compared with the CLIP text
   embedding of every vocabulary term.  Among the terms that match the
   centroid best (top :data:`_CANDIDATES` by cosine), the ones that are most
   *distinctive* for this cluster win, i.e. those whose similarity is highest
   relative to the average over all clusters.  This keeps generic terms such
   as "person" or "daytime", which match almost every photo, out of names
   unless they really stand out.
3. **Order** members by similarity to the centroid, so an album opens with
   its most typical photos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

_MINIBATCH_THRESHOLD = 20_000
_CANDIDATES = 12  # terms considered per cluster (by raw similarity)
_DUPLICATE_TERM_SIM = 0.9  # text-text cosine above which two terms are near-synonyms
_SEED = 42
_LEAD_MIN_SHARE = 0.6  # a lead term needs >= 60 % of the best term's lift


@dataclass(frozen=True)
class SuggestParams:
    """User-tunable settings for one suggestion run."""

    n_albums: int = 40
    """Number of clusters K-Means looks for (fewer when the library is small)."""
    min_photos: int = 12
    """Clusters with fewer photos are not suggested."""
    terms_per_name: int = 2
    """How many vocabulary terms make up an album name."""


DEFAULT_PARAMS = SuggestParams()


@dataclass
class Suggestion:
    """One suggested album."""

    name: str
    terms: list[tuple[str, float]]  # best first, (term, distinctiveness)
    asset_ids: list[int]  # most typical first
    similarities: list[float]  # cosine to centroid, aligned with asset_ids
    cohesion: float  # mean of similarities
    centroid: Any = field(repr=False, default=None)


def normalise_rows(mat: np.ndarray) -> np.ndarray:
    """Return *mat* with every row scaled to unit L2 norm (zero rows stay zero)."""
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def effective_k(n: int, n_albums: int) -> int:
    """Number of clusters to request for *n* photos.

    Never more than *n_albums*, and on small libraries about one cluster per
    ~20 photos so clusters are big enough to be worth an album.
    """
    if n < 2:
        return 1
    return max(1, min(n_albums, n, max(2, n // 20)))


def cluster_embeddings(mat: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Cluster unit vectors with K-Means.

    Returns:
        ``(labels, centroids)`` where *centroids* are L2-normalised, shape ``(k, d)``.
    """
    n = mat.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int), np.zeros((0, mat.shape[1]), dtype=np.float32)
    k = max(1, min(k, n))
    if k == 1:
        labels = np.zeros(n, dtype=int)
    elif n > _MINIBATCH_THRESHOLD:
        from sklearn.cluster import MiniBatchKMeans

        labels = MiniBatchKMeans(
            n_clusters=k, random_state=_SEED, batch_size=4096, n_init=3
        ).fit_predict(mat)
    else:
        from sklearn.cluster import KMeans

        labels = KMeans(n_clusters=k, random_state=_SEED, n_init=3).fit_predict(mat)

    centroids = np.zeros((k, mat.shape[1]), dtype=np.float32)
    for cid in range(k):
        members = mat[labels == cid]
        if len(members):
            centroids[cid] = members.mean(axis=0)
    return labels.astype(int), normalise_rows(centroids)


def rank_terms(
    centroids: np.ndarray,
    vocab: np.ndarray,
    terms: list[str],
    *,
    max_terms: int = 3,
    lead_terms: set[str] | frozenset[str] | None = None,
) -> list[list[tuple[str, float]]]:
    """Pick the most distinctive vocabulary terms for each centroid.

    Args:
        centroids: ``(k, d)`` unit vectors.
        vocab: ``(V, d)`` unit text embeddings, aligned with *terms*.
        max_terms: Terms returned per cluster.
        lead_terms: When given, the first term is taken from this set if a
            candidate in it is nearly as distinctive as the best term (subjects
            such as "beach" make better album names than styles such as "moody").

    Returns:
        For every centroid, up to *max_terms* ``(term, lift)`` pairs, best
        first.  *lift* is how much more the term matches this cluster than the
        average cluster (in cosine units).
    """
    if len(centroids) == 0 or len(terms) == 0:
        return [[] for _ in range(len(centroids))]
    sims = centroids @ vocab.T  # (k, V)
    lift = sims - sims.mean(axis=0, keepdims=True)
    text_sims = vocab @ vocab.T
    n_cand = min(_CANDIDATES, len(terms))
    out: list[list[tuple[str, float]]] = []
    for row in range(len(centroids)):
        candidates = np.argsort(-sims[row])[:n_cand]
        # Highest lift first; ties (e.g. a single cluster, where lift is 0)
        # fall back to raw similarity because argsort is stable.
        ordered = list(candidates[np.argsort(-lift[row, candidates], kind="stable")])
        if lead_terms:
            lead = next((i for i in ordered if terms[i] in lead_terms), None)
            best = lift[row, ordered[0]]
            # Only promote a subject that is nearly as distinctive as the best term.
            if lead is not None and lift[row, lead] >= _LEAD_MIN_SHARE * best:
                ordered.remove(lead)
                ordered.insert(0, lead)
        chosen: list[int] = []
        for idx in ordered:
            if any(text_sims[idx, c] >= _DUPLICATE_TERM_SIM for c in chosen):
                continue
            chosen.append(int(idx))
            if len(chosen) == max_terms:
                break
        out.append([(terms[i], round(float(lift[row, i]), 4)) for i in chosen])
    return out


def format_name(terms: list[str]) -> str:
    """``["beach", "sunset"]`` → ``"Beach · Sunset"``."""
    if not terms:
        return "Untitled album"
    return " · ".join(t[:1].upper() + t[1:] for t in terms)


def unique_names(ranked: list[list[tuple[str, float]]], terms_per_name: int) -> list[str]:
    """Build one name per cluster, adding a term when two names would collide."""
    names: list[str] = []
    used: set[str] = set()
    for terms in ranked:
        words = [t for t, _ in terms]
        n = max(1, terms_per_name)
        name = format_name(words[:n])
        while name in used and n < len(words):
            n += 1
            name = format_name(words[:n])
        base, i = name, 2
        while name in used:
            name = f"{base} ({i})"
            i += 1
        used.add(name)
        names.append(name)
    return names


def suggest_albums(
    asset_ids: list[int],
    embeddings: np.ndarray,
    vocab: np.ndarray,
    terms: list[str],
    params: SuggestParams = DEFAULT_PARAMS,
    *,
    lead_terms: set[str] | frozenset[str] | None = None,
) -> list[Suggestion]:
    """Cluster *embeddings* and return named album suggestions, biggest first.

    Args:
        asset_ids: Asset id per embedding row.
        embeddings: ``(n, d)`` CLIP image embeddings (normalised here).
        vocab: ``(V, d)`` CLIP text embeddings of *terms* (normalised here).
        terms: Vocabulary terms.
        params: Run settings.
        lead_terms: Terms preferred as the first word of a name (see
            :func:`rank_terms`).
    """
    if len(asset_ids) == 0:
        return []
    mat = normalise_rows(embeddings)
    vocab = normalise_rows(vocab)
    k = effective_k(len(asset_ids), params.n_albums)
    labels, centroids = cluster_embeddings(mat, k)
    ranked = rank_terms(
        centroids,
        vocab,
        terms,
        max_terms=max(3, params.terms_per_name + 1),
        lead_terms=lead_terms,
    )

    ids = np.asarray(asset_ids)
    clusters: list[tuple[int, np.ndarray, np.ndarray]] = []
    for cid in range(len(centroids)):
        idx = np.where(labels == cid)[0]
        if len(idx) < params.min_photos or len(idx) == 0:
            continue
        sims = mat[idx] @ centroids[cid]
        order = np.argsort(-sims, kind="stable")
        clusters.append((cid, idx[order], sims[order]))
    clusters.sort(key=lambda c: (-len(c[1]), c[0]))

    names = unique_names([ranked[cid] for cid, _, _ in clusters], params.terms_per_name)
    return [
        Suggestion(
            name=name,
            terms=ranked[cid],
            asset_ids=[int(a) for a in ids[idx]],
            similarities=[round(float(s), 4) for s in sims],
            cohesion=round(float(sims.mean()), 4),
            centroid=centroids[cid],
        )
        for name, (cid, idx, sims) in zip(names, clusters, strict=True)
    ]
