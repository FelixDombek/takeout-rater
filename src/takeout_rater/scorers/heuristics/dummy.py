"""Dummy scorer: trivial heuristic used for scaffolding and tests.

This scorer always returns a fixed score of 0.5 for every image.  It exists
so that the scorer registry and spec types are exercised even when no real
model is installed.

Do **not** use this scorer for actual photo ranking.
"""

from __future__ import annotations

from pathlib import Path

from takeout_rater.scorers.base import BaseScorer, MetricSpec, ScorerSpec, VariantSpec


class DummyScorer(BaseScorer):
    """Always returns 0.5 for every image (scaffolding / testing only)."""

    @classmethod
    def spec(cls) -> ScorerSpec:
        return ScorerSpec(
            scorer_id="dummy",
            display_name="Dummy (test)",
            description="Returns a constant 0.5 score. Used for scaffolding and tests only.",
            version="1",
            metrics=(
                MetricSpec(
                    key="dummy_score",
                    display_name="Dummy score",
                    description="Always 0.5 – not meaningful.",
                    min_value=0.0,
                    max_value=1.0,
                    higher_is_better=True,
                ),
            ),
            variants=(
                VariantSpec(
                    variant_id="default",
                    display_name="Default",
                    description="The only variant.",
                ),
            ),
            default_variant_id="default",
        )

    @classmethod
    def is_available(cls) -> bool:
        return True

    def score_batch(
        self,
        image_paths: list[Path],
        *,
        variant_id: str | None = None,
    ) -> list[dict[str, float]]:
        return [{"dummy_score": 0.5} for _ in image_paths]
