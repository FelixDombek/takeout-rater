"""3-D embedding map builder for CLIP image embeddings.

Pipeline
--------
1. **StandardScaler** – normalise each of the 768 dimensions to zero mean and
   unit variance so that no single feature dominates the reduction.
2. **PCA** – reduce to 50 components (or fewer when the dataset is small) to
   remove noise before the expensive UMAP step.
3. **UMAP** – project the 50-D output to 3 dimensions using cosine metric for
   the nearest-neighbour graph (CLIP embeddings are already unit-normalised).
4. **Clustering** – cluster the 3-D coordinates with K-Means or HDBSCAN so
   each point can be coloured by group in the frontend.
5. **Representative selection** – for each cluster the asset whose 3-D
   coordinate is closest to the cluster centroid is chosen as the
   representative thumbnail.

The result is returned as a plain ``dict`` that can be serialised directly to
JSON:

.. code-block:: json

    {
        "points": [
            {"asset_id": 123, "x": 1.23, "y": -0.45, "z": 0.67,
             "cluster_id": 2, "relpath": "Photos/img.jpg"}
        ],
        "clusters": [
            {"cluster_id": 0, "representative_asset_id": 456, "size": 10}
        ],
        "total": 500
    }
"""

from __future__ import annotations

import contextlib
import io
import math
import struct
from collections.abc import Callable
from typing import Any

_DIM = 768  # ViT-L/14 embedding dimension
_PCA_COMPONENTS = 50
_UMAP_COMPONENTS = 3
_UMAP_METRIC = "cosine"
_MIN_UMAP_N = 15  # UMAP needs at least this many samples to be meaningful
_MIN_CLUSTERS = 2  # always create at least 2 clusters when n ≥ 2
_DEFAULT_MAX_CLUSTERS = 24  # cap to keep the legend readable
_CLUSTERING_METHODS = {"kmeans", "hdbscan"}

# Progress fractions at the end of each stage (cumulative).
_FRAC_LOADED = 0.03
_FRAC_SCALED = 0.07
_FRAC_PCA = 0.15
_FRAC_UMAP = 0.90  # epoch-level sub-progress fills _FRAC_PCA → _FRAC_UMAP
_FRAC_CLUSTERING = 0.96
_FRAC_DONE = 1.00


def _n_clusters(n: int, max_clusters: int = _DEFAULT_MAX_CLUSTERS) -> int:
    """Return a sensible number of KMeans clusters for *n* points.

    Uses the common heuristic ``k ≈ √n``, capped at ``max_clusters`` to
    keep the legend readable and at ``n`` so KMeans never asks for more
    clusters than there are data points.
    """
    if max_clusters < 1:
        raise ValueError("max_clusters must be at least 1")
    if n < 2:
        return 1
    return min(max_clusters, max(_MIN_CLUSTERS, int(math.sqrt(n))), n)


def _normalise_cluster_labels(labels: Any) -> Any:
    """Renumber non-noise cluster labels to stable ``0..N`` ids.

    HDBSCAN uses ``-1`` for noise; that label is preserved so the frontend can
    display noise separately.
    """
    import numpy as np  # noqa: PLC0415

    normalised = np.full(labels.shape, -1, dtype=int)
    cluster_ids = sorted(int(label) for label in np.unique(labels) if int(label) >= 0)
    for new_id, old_id in enumerate(cluster_ids):
        normalised[labels == old_id] = new_id
    return normalised


def _cap_hdbscan_clusters(coords: Any, labels: Any, max_clusters: int) -> Any:
    """Merge excess HDBSCAN clusters into the nearest retained cluster.

    HDBSCAN does not expose a hard maximum-cluster parameter.  We keep the
    largest clusters up to ``max_clusters`` and reassign points from smaller
    clusters to the nearest retained centroid.  Noise (``-1``) remains noise.
    """
    import numpy as np  # noqa: PLC0415

    cluster_ids = [int(label) for label in np.unique(labels) if int(label) >= 0]
    if len(cluster_ids) <= max_clusters:
        return labels

    sizes = {cid: int((labels == cid).sum()) for cid in cluster_ids}
    retained = sorted(cluster_ids, key=lambda cid: (-sizes[cid], cid))[:max_clusters]
    retained_centroids = {cid: coords[labels == cid].mean(axis=0) for cid in retained}

    capped = labels.copy()
    for cid in cluster_ids:
        if cid in retained:
            continue
        indices = np.where(labels == cid)[0]
        for idx in indices:
            nearest = min(
                retained,
                key=lambda keep_id: float(
                    np.linalg.norm(coords[idx] - retained_centroids[keep_id])
                ),
            )
            capped[idx] = nearest
    return capped


def _cluster_coords(
    coords: Any,
    method: str,
    max_clusters: int,
) -> tuple[Any, dict[int, Any]]:
    """Cluster 3-D coordinates and return labels plus representative centroids."""
    import numpy as np  # noqa: PLC0415

    n = coords.shape[0]
    if max_clusters < 1:
        raise ValueError("max_clusters must be at least 1")
    if method not in _CLUSTERING_METHODS:
        raise ValueError(f"Unsupported clustering method: {method}")
    if n == 0:
        return np.array([], dtype=int), {}
    if n == 1:
        return np.zeros(1, dtype=int), {0: coords[0]}

    if method == "kmeans":
        from sklearn.cluster import KMeans  # noqa: PLC0415

        k = _n_clusters(n, max_clusters=max_clusters)
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(coords)
        centroids = {cid: km.cluster_centers_[cid] for cid in range(k)}
        return labels, centroids

    from sklearn.cluster import HDBSCAN  # noqa: PLC0415

    min_cluster_size = max(2, math.ceil(n / max_clusters))
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, copy=True)
    labels = clusterer.fit_predict(coords)
    labels = _cap_hdbscan_clusters(coords, labels, max_clusters=max_clusters)
    labels = _normalise_cluster_labels(labels)

    centroids: dict[int, Any] = {}
    for cid in sorted(int(label) for label in np.unique(labels)):
        centroids[cid] = coords[labels == cid].mean(axis=0)
    return labels, centroids


@contextlib.contextmanager
def _track_umap_epochs(
    on_progress: Callable[[float, str], None],
    base: float,
    scale: float,
):
    """Temporarily replace ``umap.umap_.trange`` with an epoch-tracking version.

    While the context is active, every tqdm ``update()`` call inside the UMAP
    epoch loop forwards a fractional progress value to *on_progress* so the
    caller can report per-epoch progress to the UI.  Output is redirected to a
    ``StringIO`` sink so nothing is printed to the server console.

    The patch is applied to the ``umap.umap_`` module directly.  If the module
    is not importable the context silently yields without patching.

    .. note::
        This relies on ``umap-learn`` using ``tqdm.trange`` for its epoch loop
        (true as of umap-learn 0.5.x).  If a future release changes its
        progress-reporting mechanism the patch will silently have no effect —
        the computation still completes correctly, but epoch-level progress
        will not be reported.
    """
    try:
        import importlib  # noqa: PLC0415

        umap_impl = importlib.import_module("umap.umap_")
    except ImportError:
        yield
        return

    orig_tqdm = getattr(umap_impl, "tqdm", None)
    orig_trange = getattr(umap_impl, "trange", None)
    if orig_tqdm is None:
        yield
        return

    _sink = io.StringIO()

    class _TrackingTqdm(orig_tqdm):  # type: ignore[valid-type]
        def update(self, n: int = 1) -> bool | None:
            result = super().update(n)
            if self.total and self.total > 0:
                on_progress(
                    base + scale * min(1.0, self.n / self.total),
                    f"UMAP projection – epoch {self.n}/{self.total}",
                )
            return result

    def _tracking_trange(*args: Any, **kwargs: Any) -> _TrackingTqdm:
        kwargs.pop("disable", None)  # force-enable so update() fires
        kwargs["file"] = _sink  # discard bar text output
        return _TrackingTqdm(range(*args), **kwargs)

    def _tracking_tqdm(iterable: Any = None, *args: Any, **kwargs: Any) -> _TrackingTqdm:
        kwargs.pop("disable", None)
        kwargs["file"] = _sink
        return _TrackingTqdm(iterable, *args, **kwargs)

    umap_impl.tqdm = _tracking_tqdm
    if orig_trange is not None:
        umap_impl.trange = _tracking_trange
    try:
        yield
    finally:
        umap_impl.tqdm = orig_tqdm
        if orig_trange is not None:
            umap_impl.trange = orig_trange


def build_embedding_map(
    rows: list[tuple[int, bytes, str]],
    progress_callback: Callable[[float, str], None] | None = None,
    clustering_method: str = "kmeans",
    max_clusters: int = _DEFAULT_MAX_CLUSTERS,
) -> dict[str, Any]:
    """Compute a 3-D embedding map from raw CLIP embedding rows.

    Args:
        rows: List of ``(asset_id, embedding_blob, relpath)`` tuples as
            returned by
            :func:`takeout_rater.db.queries.load_clip_embeddings_with_relpaths`.
        progress_callback: Optional callable ``(fraction, message)`` invoked
            at each pipeline stage.  *fraction* is in ``[0.0, 1.0]``.
            During the UMAP step it is called once per epoch.
        clustering_method: ``"kmeans"`` or ``"hdbscan"``.
        max_clusters: Maximum number of non-noise clusters to show.

    Returns:
        A dict with keys ``"points"``, ``"clusters"``, and ``"total"`` as
        described in the module docstring.  Returns an empty-result dict when
        *rows* is empty.
    """
    if not rows:
        return {"points": [], "clusters": [], "total": 0}
    if clustering_method not in _CLUSTERING_METHODS:
        raise ValueError(f"Unsupported clustering method: {clustering_method}")
    if max_clusters < 1:
        raise ValueError("max_clusters must be at least 1")

    def _cb(frac: float, msg: str) -> None:
        if progress_callback is not None:
            progress_callback(frac, msg)

    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    n = len(rows)
    asset_ids = [r[0] for r in rows]
    relpaths = [r[2] for r in rows]

    # Decode float32 blobs → numpy matrix  (N, 768)
    mat = np.empty((n, _DIM), dtype=np.float32)
    for i, (_, blob, _) in enumerate(rows):
        mat[i] = np.array(struct.unpack(f"{_DIM}f", blob), dtype=np.float32)
    _cb(_FRAC_LOADED, "Scaling features…")

    # 1. StandardScaler
    mat_scaled = StandardScaler().fit_transform(mat)
    _cb(_FRAC_SCALED, f"PCA reduction (768 → {min(_PCA_COMPONENTS, n - 1)})…")

    # 2. PCA
    pca_components = min(_PCA_COMPONENTS, n - 1, mat_scaled.shape[1])
    if pca_components < 1:
        mat_pca = np.zeros((n, 1), dtype=float)
    else:
        mat_pca = PCA(n_components=pca_components, random_state=42).fit_transform(mat_scaled)
    _cb(_FRAC_PCA, "UMAP projection…")

    # 3. UMAP (skip when too few samples — fall back to PCA[:3])
    if n >= _MIN_UMAP_N:
        import umap  # noqa: PLC0415

        n_neighbors = min(15, n - 1)
        reducer = umap.UMAP(
            n_components=_UMAP_COMPONENTS,
            metric=_UMAP_METRIC,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            random_state=42,
        )
        with _track_umap_epochs(_cb, base=_FRAC_PCA, scale=_FRAC_UMAP - _FRAC_PCA):
            coords = reducer.fit_transform(mat_pca).astype(float)
    else:
        # Pad PCA output to 3 columns if needed
        if mat_pca.shape[1] < _UMAP_COMPONENTS:
            pad = np.zeros((n, _UMAP_COMPONENTS - mat_pca.shape[1]))
            coords = np.hstack([mat_pca, pad]).astype(float)
        else:
            coords = mat_pca[:, :_UMAP_COMPONENTS].astype(float)
    _cb(_FRAC_UMAP, "Clustering…")

    # 4. Clustering on 3-D coords
    cluster_labels, centroids = _cluster_coords(
        coords,
        method=clustering_method,
        max_clusters=max_clusters,
    )
    _cb(_FRAC_CLUSTERING, "Finalizing…")

    # 5. Representative per cluster: asset closest to its centroid
    representative: dict[int, int] = {}  # cluster_id → index into rows
    cluster_ids = sorted(int(cid) for cid in centroids)
    for cid in cluster_ids:
        mask = cluster_labels == cid
        if not mask.any():
            continue
        indices = np.where(mask)[0]
        dists = np.linalg.norm(coords[indices] - centroids[cid], axis=1)
        representative[cid] = int(indices[np.argmin(dists)])

    # Build output
    points = [
        {
            "asset_id": int(asset_ids[i]),
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4),
            "z": round(float(coords[i, 2]), 4),
            "cluster_id": int(cluster_labels[i]),
            "relpath": relpaths[i],
        }
        for i in range(n)
    ]

    clusters = [
        {
            "cluster_id": cid,
            "representative_asset_id": int(asset_ids[representative[cid]])
            if cid in representative
            else None,
            "size": int((cluster_labels == cid).sum()),
        }
        for cid in cluster_ids
    ]

    return {
        "points": points,
        "clusters": clusters,
        "total": n,
        "params": {
            "clustering_method": clustering_method,
            "max_clusters": max_clusters,
        },
    }
