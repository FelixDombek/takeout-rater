"""BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) scorer.

BRISQUE is a no-reference image quality metric that uses scene statistics of
locally normalised luminance coefficients to quantify distortions.  A score
of 0 indicates a pristine image, while higher scores (up to ~100) indicate
increasing levels of distortion.  **Lower is better.**

This scorer wraps :func:`skimage.metrics.brisque` from `scikit-image`, which
must be installed separately::

    pip install takeout-rater[brisque]

References
----------
A. Mittal, A. K. Moorthy, and A. C. Bovik, "No-Reference Image Quality
Assessment in the Spatial Domain," IEEE Transactions on Image Processing,
vol. 21, no. 12, pp. 4695–4708, Dec. 2012.
"""

from __future__ import annotations

from pathlib import Path

from takeout_rater.scorers.base import BaseScorer, MetricSpec, ScorerSpec, VariantSpec

# BRISQUE scores can technically exceed 100 for heavily distorted images.
# We clamp to this ceiling so the metric stays on its declared 0–100 scale.
_MAX_BRISQUE: float = 100.0


class BRISQUEScorer(BaseScorer):
    """No-reference image quality scorer using BRISQUE (scikit-image)."""

    @classmethod
    def spec(cls) -> ScorerSpec:
        return ScorerSpec(
            scorer_id="brisque",
            display_name="BRISQUE",
            description=(
                "Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE).  "
                "A no-reference metric based on natural scene statistics.  "
                "Scores range from 0 (best) to 100 (worst).  Lower is better."
            ),
            metrics=(
                MetricSpec(
                    key="brisque",
                    display_name="BRISQUE",
                    description=(
                        "BRISQUE distortion score normalised to 0–100.  "
                        "Lower values indicate higher perceptual quality."
                    ),
                    min_value=0.0,
                    max_value=100.0,
                    higher_is_better=False,
                ),
            ),
            variants=(
                VariantSpec(
                    variant_id="default",
                    display_name="Default",
                    description=(
                        "skimage.metrics.brisque — no ML model, "
                        "scene-statistics approach (Mittal et al. 2012)."
                    ),
                ),
            ),
            default_variant_id="default",
            requires_extras=("brisque",),
        )

    @classmethod
    def is_available(cls) -> bool:
        try:
            from skimage.metrics import brisque  # noqa: F401

            return True
        except ImportError:
            return False

    def score_batch(
        self,
        image_paths: list[Path],
        *,
        variant_id: str | None = None,
    ) -> list[dict[str, float]]:
        """Score a batch of images using BRISQUE.

        Args:
            image_paths: Paths to image files (can be thumbnails).
            variant_id: Ignored; only one variant exists.

        Returns:
            List of dicts with key ``"brisque"`` → float in ``[0, 100]``.
            If a file cannot be opened or scored, ``brisque`` is set to
            ``_MAX_BRISQUE`` (worst possible score).
        """
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        from skimage.metrics import brisque  # noqa: PLC0415

        results: list[dict[str, float]] = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                arr = np.asarray(img)
                raw: float = float(brisque(arr))
                score = min(max(raw, 0.0), _MAX_BRISQUE)
            except (OSError, ValueError, Exception):  # noqa: BLE001
                score = _MAX_BRISQUE
            results.append({"brisque": score})
        return results
