"""Blur / sharpness scorer: measures image focus using edge-filter variance.

The ``sharpness`` metric is derived from the variance of the output of the
Pillow ``FIND_EDGES`` filter applied to the greyscale image.  A higher value
means the image is sharper (more in-focus).
The raw variance is mapped to a 0–100 scale by clamping at ``_MAX_RAW_VAR``
so that the output is bounded and comparable across images.

This scorer requires Pillow (``pip install Pillow``) but has no other
dependencies.  It is always available when Pillow is installed.
"""

from __future__ import annotations

from pathlib import Path

from takeout_rater.scorers.base import BaseScorer, MetricSpec, ScorerSpec, VariantSpec

# Raw edge-filter variance corresponding to a "100" sharpness score.
# Values above this threshold are clamped to 100.
_MAX_RAW_VAR: float = 2000.0


class BlurScorer(BaseScorer):
    """Measures image sharpness via edge-filter variance (PIL-based)."""

    @classmethod
    def spec(cls) -> ScorerSpec:
        return ScorerSpec(
            scorer_id="blur",
            display_name="Blur / Sharpness",
            description=(
                "Estimates image focus using the variance of the FIND_EDGES filter output.  "
                "Higher sharpness means the image is more in-focus."
            ),
            metrics=(
                MetricSpec(
                    key="sharpness",
                    display_name="Sharpness",
                    description=(
                        "Edge-filter variance normalised to 0–100.  "
                        "Higher values indicate sharper, more in-focus images."
                    ),
                    min_value=0.0,
                    max_value=100.0,
                    higher_is_better=True,
                ),
            ),
            variants=(
                VariantSpec(
                    variant_id="default",
                    display_name="Default",
                    description="Edge-filter variance via Pillow (no ML model).",
                ),
            ),
            default_variant_id="default",
            requires_extras=("index",),
        )

    @classmethod
    def is_available(cls) -> bool:
        try:
            from PIL import Image, ImageFilter, ImageStat  # noqa: F401

            return True
        except ImportError:
            return False

    def score_batch(
        self,
        image_paths: list[Path],
        *,
        variant_id: str | None = None,
    ) -> list[dict[str, float]]:
        """Score a batch of images for sharpness.

        Args:
            image_paths: Paths to image files (can be thumbnails).
            variant_id: Ignored; only one variant exists.

        Returns:
            List of dicts with key ``"sharpness"`` → float in ``[0, 100]``.
            If a file cannot be opened, ``sharpness`` is set to ``0.0``.
        """
        from PIL import Image, ImageFilter, ImageStat  # noqa: PLC0415

        results: list[dict[str, float]] = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("L")
                edges = img.filter(ImageFilter.FIND_EDGES)
                stat = ImageStat.Stat(edges)
                raw_var: float = stat.var[0]
                sharpness = min(raw_var / _MAX_RAW_VAR, 1.0) * 100.0
            except (OSError, ValueError):
                sharpness = 0.0
            results.append({"sharpness": sharpness})
        return results
