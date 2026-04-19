"""Slow embedding-map tests.

Excluded from the default test run (--ignore=tests/slow) because these tests
exercise the full UMAP pipeline.
"""

from __future__ import annotations

import struct

_DIM = 768


def _make_embedding(seed: int = 0) -> bytes:
    """Create a deterministic normalised 768-dim float32 embedding blob."""
    import numpy as np

    rng = np.random.RandomState(seed)
    vec = rng.randn(_DIM).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return struct.pack(f"{_DIM}f", *vec)


def test_many_points_full_pipeline() -> None:
    """With enough points UMAP runs; output shape and keys are correct."""
    from takeout_rater.clustering.embedding_map import build_embedding_map

    n = 30
    rows = [(i + 1, _make_embedding(i), f"img{i}.jpg") for i in range(n)]
    result = build_embedding_map(rows)
    assert result["total"] == n
    assert len(result["points"]) == n
    for pt in result["points"]:
        assert {"asset_id", "x", "y", "z", "cluster_id", "relpath"} <= pt.keys()
    for cl in result["clusters"]:
        assert cl["representative_asset_id"] is not None
        assert cl["size"] > 0
