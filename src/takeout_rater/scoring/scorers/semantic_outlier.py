"""Semantic outlier scorer: photos unlike anything else in the library.

Uses the image embeddings already stored for search (CLIP ViT-L/14 from
indexing, or SigLIP 2 when that embedding pass has run) and averages each
photo's cosine similarity to its 10 nearest neighbours.  Photos whose nearest
neighbours are all far away are semantic outliers: one-offs, odd subjects,
accidental shots, screenshots or scans mixed into a photo library.  They are
either hidden gems or candidates for culling.

Metrics:

- ``outlier`` — percentile rank 0–1 within the library (1 = more unusual than
  every other photo).  Sort by it descending, or filter ``>= 0.98`` to review
  the 2 % most unusual photos.
- ``knn_similarity`` — the raw mean cosine similarity to the 10 nearest
  neighbours (low = unusual).  Comparable across runs of the same backbone.

The score depends on the whole library, so it is a :class:`LibraryScorer`: the
scoring pipeline computes all photos in one pass, and recomputes everything
when new embeddings have appeared since the last run.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Any

from takeout_rater.scoring.scorers.base import (
    LibraryScorer,
    MetricSpec,
    ScorerSpec,
    VariantSpec,
)

#: variant_id → embedding model key (see ``db.queries.load_embedding_matrix``).
_VARIANT_MODEL_KEYS: dict[str, str] = {
    "clip_l14": "clip-vit-l-14",
    "siglip2": "siglip2-so400m-16-256",
}

_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        key="outlier",
        display_name="Outlier rank",
        description=(
            "How unusual the photo is compared with the rest of the library, as a "
            "percentile (0.99 = more unusual than 99 % of photos). High values are "
            "one-offs: hidden gems or junk worth deleting."
        ),
        higher_is_better=False,
    ),
    MetricSpec(
        key="knn_similarity",
        display_name="Neighbour similarity",
        description=(
            "Average similarity (cosine) to the 10 most similar photos. Low = the "
            "photo looks like nothing else in the library."
        ),
        min_value=-1.0,
        max_value=1.0,
    ),
)


class SemanticOutlierScorer(LibraryScorer):
    """kNN distance in embedding space, scored over the whole library."""

    #: Neighbours averaged per photo.
    k: int = 10

    @classmethod
    def spec(cls) -> ScorerSpec:
        return ScorerSpec(
            scorer_id="semantic_outlier",
            display_name="Semantic outliers",
            description=(
                "Finds photos that look like nothing else in your library: odd one-offs, "
                "accidental shots, screenshots, scans. Uses the search embeddings, so it "
                "needs no download and runs in seconds to minutes. Re-run after indexing "
                "new photos."
            ),
            technical_description=(
                "Exact k-nearest-neighbour search (k=10, cosine) over stored image "
                "embeddings; score = mean similarity to the k neighbours, plus its "
                "percentile rank within the library. Blocked matrix product on GPU "
                "(fp16) or CPU."
            ),
            version="1",
            variants=(
                VariantSpec(
                    variant_id="clip_l14",
                    display_name="CLIP ViT-L/14",
                    description="Uses the CLIP embeddings computed during indexing.",
                    primary_metric_key="outlier",
                    metrics=_METRICS,
                ),
                VariantSpec(
                    variant_id="siglip2",
                    display_name="SigLIP 2",
                    description=(
                        "Uses SigLIP 2 search embeddings (run the SigLIP 2 embedding pass "
                        "on the Search page first)."
                    ),
                    primary_metric_key="outlier",
                    metrics=_METRICS,
                ),
            ),
            default_variant_id="clip_l14",
        )

    @classmethod
    def is_available(cls) -> bool:
        return True  # numpy (and torch, if present) are core dependencies

    @property
    def model_key(self) -> str:
        return _VARIANT_MODEL_KEYS[self.variant_id]

    def _embedding_source(self) -> tuple[str, tuple[Any, ...]]:
        """SQL selecting asset ids that have an embedding for this variant."""
        if self.variant_id == "clip_l14":
            return "SELECT asset_id FROM clip_embeddings", ()
        return "SELECT asset_id FROM image_embeddings WHERE model_key = ?", (self.model_key,)

    def pending_count(self, conn: sqlite3.Connection, *, scorer_version: str) -> int:
        source_sql, params = self._embedding_source()
        row = conn.execute(
            f"SELECT COUNT(*) FROM ({source_sql}) e"
            " WHERE NOT EXISTS ("
            "   SELECT 1 FROM asset_scores s"
            "   WHERE s.asset_id = e.asset_id AND s.scorer_id = ? AND s.variant_id = ?"
            "     AND s.metric_key = 'outlier' AND s.scorer_version = ?"
            " )",
            (*params, self.spec().scorer_id, self.variant_id, scorer_version),
        ).fetchone()
        return int(row[0]) if row else 0

    def score_library(
        self,
        conn: sqlite3.Connection,
        *,
        on_progress: Callable[[int, int], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> list[tuple[int, dict[str, float]]] | None:
        from takeout_rater.clustering.outliers import knn_mean_similarity, outlier_percentiles
        from takeout_rater.db.queries import load_embedding_matrix

        asset_ids, matrix = load_embedding_matrix(conn, self.model_key)
        if len(asset_ids) < 2:
            return []
        sims = knn_mean_similarity(
            matrix, k=self.k, on_progress=on_progress, cancel_check=cancel_check
        )
        if sims is None:
            return None
        ranks = outlier_percentiles(sims)
        return [
            (int(aid), {"outlier": float(r), "knn_similarity": float(s)})
            for aid, r, s in zip(asset_ids, ranks, sims, strict=True)
        ]
