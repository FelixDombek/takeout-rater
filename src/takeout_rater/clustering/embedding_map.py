"""3-D embedding map builder for CLIP image embeddings.

Pipeline
--------
1. **StandardScaler** – normalise each of the 768 dimensions to zero mean and
   unit variance so that no single feature dominates the reduction.
2. **PCA** – reduce to 50 components (or fewer when the dataset is small) to
   remove noise before the expensive UMAP step.
3. **UMAP** – project the 50-D output to 3 dimensions using cosine metric for
   the nearest-neighbour graph (CLIP embeddings are already unit-normalised).
4. **KMeans** – cluster the 3-D coordinates so each point can be coloured by
   group in the frontend.
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

import math
import struct
from typing import Any

_DIM = 768  # ViT-L/14 embedding dimension
_PCA_COMPONENTS = 50
_UMAP_COMPONENTS = 3
_UMAP_METRIC = "cosine"
_MIN_UMAP_N = 15  # UMAP needs at least this many samples to be meaningful


def _n_clusters(n: int) -> int:
    """Return a sensible number of KMeans clusters for *n* points."""
    if n < 2:
        return 1
    # Use ~sqrt(n/2) capped at 12, but never more than n itself
    return min(12, max(2, int(math.sqrt(n / 2))), n)


def build_embedding_map(
    rows: list[tuple[int, bytes, str]],
) -> dict[str, Any]:
    """Compute a 3-D embedding map from raw CLIP embedding rows.

    Args:
        rows: List of ``(asset_id, embedding_blob, relpath)`` tuples as
            returned by
            :func:`takeout_rater.db.queries.load_clip_embeddings_with_relpaths`.

    Returns:
        A dict with keys ``"points"``, ``"clusters"``, and ``"total"`` as
        described in the module docstring.  Returns an empty-result dict when
        *rows* is empty.
    """
    if not rows:
        return {"points": [], "clusters": [], "total": 0}

    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    n = len(rows)
    asset_ids = [r[0] for r in rows]
    relpaths = [r[2] for r in rows]

    # Decode float32 blobs → numpy matrix  (N, 768)
    mat = np.empty((n, _DIM), dtype=np.float32)
    for i, (_, blob, _) in enumerate(rows):
        mat[i] = np.array(struct.unpack(f"{_DIM}f", blob), dtype=np.float32)

    # 1. StandardScaler
    mat_scaled = StandardScaler().fit_transform(mat)

    # 2. PCA
    pca_components = min(_PCA_COMPONENTS, n - 1, mat_scaled.shape[1])
    mat_pca = PCA(n_components=pca_components, random_state=42).fit_transform(mat_scaled)

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
        coords = reducer.fit_transform(mat_pca).astype(float)
    else:
        # Pad PCA output to 3 columns if needed
        if mat_pca.shape[1] < _UMAP_COMPONENTS:
            pad = np.zeros((n, _UMAP_COMPONENTS - mat_pca.shape[1]))
            coords = np.hstack([mat_pca, pad]).astype(float)
        else:
            coords = mat_pca[:, :_UMAP_COMPONENTS].astype(float)

    # 4. KMeans clustering on 3-D coords
    k = _n_clusters(n)
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    cluster_labels = km.fit_predict(coords)
    centroids = km.cluster_centers_  # shape (k, 3)

    # 5. Representative per cluster: asset closest to its centroid
    representative: dict[int, int] = {}  # cluster_id → index into rows
    for cid in range(k):
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
        for cid in range(k)
    ]

    return {"points": points, "clusters": clusters, "total": n}
