"""DBSCAN-based clustering of face embeddings into identity groups.

Clusters face embeddings (512-d ArcFace vectors) into person groups using
DBSCAN with cosine distance.  Each cluster represents a likely unique person.

Usage::

    from takeout_rater.faces.clustering import cluster_faces

    n = cluster_faces(conn, detection_run_id=1, eps=0.5, min_samples=2)
"""

from __future__ import annotations

import json
import sqlite3
import struct
import time
from collections.abc import Callable

from takeout_rater.faces.detector import EMBEDDING_DIM

_METHOD = "dbscan"


def cluster_faces(
    conn: sqlite3.Connection,
    *,
    detection_run_id: int | None = None,
    eps: float = 0.5,
    min_samples: int = 2,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Cluster face embeddings into person groups using DBSCAN.

    All face embeddings from the specified detection run (or all runs if
    *detection_run_id* is ``None``) are loaded, clustered via DBSCAN with
    cosine distance, and persisted to the ``face_cluster_runs``,
    ``face_clusters``, and ``face_cluster_members`` tables.

    Args:
        conn: Open library database connection.
        detection_run_id: Restrict to faces from a specific detection run.
            When ``None``, all face embeddings are used.
        eps: DBSCAN neighbourhood radius (cosine distance).  Smaller values
            produce tighter (more precise) clusters.  Default: 0.5.
        min_samples: Minimum number of face embeddings to form a cluster.
            Default: 2.
        on_progress: Optional callback ``(processed, total)``.

    Returns:
        Number of person clusters created.
    """
    import numpy as np  # noqa: PLC0415
    from sklearn.cluster import DBSCAN  # noqa: PLC0415

    # Load face embeddings
    if detection_run_id is not None:
        rows = conn.execute(
            "SELECT id, embedding FROM face_embeddings WHERE run_id = ? ORDER BY id",
            (detection_run_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, embedding FROM face_embeddings ORDER BY id",
        ).fetchall()

    if not rows:
        return 0

    face_ids: list[int] = []
    vectors: list[object] = []
    expected_bytes = EMBEDDING_DIM * 4

    for face_id, blob in rows:
        if len(blob) != expected_bytes:
            continue
        vec = np.array(struct.unpack(f"{EMBEDDING_DIM}f", blob), dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            continue
        vec = vec / norm
        face_ids.append(face_id)
        vectors.append(vec)

    if not face_ids:
        return 0

    if on_progress:
        on_progress(0, len(face_ids))

    matrix = np.stack(vectors)  # (N, 512)

    # Run DBSCAN with cosine distance
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = db.fit_predict(matrix)

    if on_progress:
        on_progress(len(face_ids), len(face_ids))

    # Collect clusters (label -1 = noise)
    cluster_map: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        if label == -1:
            continue
        cluster_map.setdefault(int(label), []).append(i)

    if not cluster_map:
        return 0

    # Persist
    params = {"eps": eps, "min_samples": min_samples}
    if detection_run_id is not None:
        params["detection_run_id"] = detection_run_id
    params_json = json.dumps(params, separators=(",", ":"), sort_keys=True)

    now = int(time.time())
    run_row = conn.execute(
        "INSERT INTO face_cluster_runs (method, params_json, detection_run_id, created_at)"
        " VALUES (?, ?, ?, ?) RETURNING id",
        (_METHOD, params_json, detection_run_id, now),
    ).fetchone()
    run_id = run_row[0]
    conn.commit()

    n_clusters = 0
    for _label, member_indices in sorted(cluster_map.items()):
        # Find representative: face closest to cluster centroid
        member_vecs = matrix[[i for i in member_indices]]
        centroid = member_vecs.mean(axis=0)
        centroid_norm = float(np.linalg.norm(centroid))
        if centroid_norm > 1e-9:
            centroid = centroid / centroid_norm
        sims = member_vecs @ centroid
        rep_local = int(np.argmax(sims))
        rep_face_id = face_ids[member_indices[rep_local]]

        cluster_row = conn.execute(
            "INSERT INTO face_clusters (run_id, label, created_at) VALUES (?, NULL, ?)"
            " RETURNING id",
            (run_id, now),
        ).fetchone()
        cluster_id = cluster_row[0]

        member_rows: list[tuple[int, int, float | None, int]] = []
        for local_idx in member_indices:
            fid = face_ids[local_idx]
            # Cosine distance from centroid
            cos_sim = float(np.dot(matrix[local_idx], centroid))
            dist = 1.0 - cos_sim
            is_rep = 1 if fid == rep_face_id else 0
            member_rows.append((cluster_id, fid, dist, is_rep))

        conn.executemany(
            "INSERT OR IGNORE INTO face_cluster_members"
            " (cluster_id, face_id, distance, is_representative)"
            " VALUES (?, ?, ?, ?)",
            member_rows,
        )
        n_clusters += 1

    conn.commit()
    return n_clusters
