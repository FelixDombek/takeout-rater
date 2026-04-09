"""BRISQUE scorer: Blind/Referenceless Image Spatial Quality Evaluator.

BRISQUE (2012, Mittal et al.) computes a no-reference quality score from
local normalised luminance statistics fitted to a multivariate Gaussian model.
Lower raw BRISQUE values indicate higher perceptual quality.

The raw score (0–100, lower = better) is inverted so that the stored metric
``brisque_quality`` is in the range [0, 100] with **higher values being better**,
consistent with the other scorers in this project.

This scorer uses the ``piq`` library (``pip install piq``), which ships the
pre-fitted BRISQUE parameters so no separate model download is needed.
Pillow is required to open images.
"""

from __future__ import annotations

import logging
from pathlib import Path

from takeout_rater.scorers.base import BaseScorer, MetricSpec, ScorerSpec, VariantSpec

_logger = logging.getLogger(__name__)

#: Raw BRISQUE scores are in [0, 100] where 0 is best.
#: We clamp at this upper bound before inverting.
_RAW_MAX: float = 100.0


class BRISQUEScorer(BaseScorer):
    """BRISQUE image quality scorer — CPU-only, no ML model required.

    Scores images using the Blind/Referenceless Image Spatial Quality Evaluator
    algorithm.  The output metric ``brisque_quality`` ranges from 0 (poor) to
    100 (excellent), derived by inverting the raw BRISQUE score so that higher
    values are always better.
    """

    @classmethod
    def spec(cls) -> ScorerSpec:
        return ScorerSpec(
            scorer_id="brisque",
            display_name="BRISQUE Quality",
            description=(
                "Uses a statistical model of what natural, undistorted images look like to "
                "spot distortions such as blur, noise, and compression artefacts — without "
                "needing a reference image. Photos that match natural image statistics score "
                "higher. The algorithm parameters are pre-fitted and built in, so no download "
                "is required."
            ),
            technical_description=(
                "Estimates perceptual image quality using BRISQUE (Blind/Referenceless "
                "Image Spatial Quality Evaluator). Detects compression artefacts, "
                "noise, and blur from local luminance statistics — no ML model required."
            ),
            version="1",
            metrics=(
                MetricSpec(
                    key="brisque_quality",
                    display_name="BRISQUE Quality",
                    description=(
                        "Perceptual quality score (0–100, higher is better). "
                        "Derived by inverting the raw BRISQUE score so that "
                        "distortion-free images score near 100."
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
                    description=(
                        "Standard BRISQUE with pre-fitted MVG parameters "
                        "from the piq library (no model download required)."
                    ),
                ),
            ),
            default_variant_id="default",
            requires_extras=(),
        )

    @classmethod
    def is_available(cls) -> bool:
        try:
            import PIL  # noqa: F401
            import piq  # noqa: F401
            import torch  # noqa: F401

            return True
        except ImportError:
            return False

    def score_batch(
        self,
        image_paths: list[Path],
        *,
        variant_id: str | None = None,
    ) -> list[dict[str, float]]:
        """Score a batch of images with BRISQUE.

        Each image is opened via Pillow, converted to a torch tensor, and
        passed through ``piq.brisque``.  The raw BRISQUE score (lower = better)
        is inverted so that the stored metric is higher-is-better.

        If a file cannot be opened or the BRISQUE computation fails, the score
        defaults to ``0.0`` (worst quality) rather than raising an exception.
        The failure is logged at WARNING level with the scorer name and asset
        path to help diagnose edge-case images (e.g. fully uniform images that
        cause an ``AssertionError`` inside ``piq._aggd_parameters`` when no
        pairwise MSCN products are negative).

        Args:
            image_paths: Absolute paths to image files.
            variant_id: Ignored; only one variant exists.

        Returns:
            List (same length as *image_paths*) of dicts with key
            ``"brisque_quality"`` → float in ``[0.0, 100.0]``.
        """
        import piq  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        from torchvision.transforms.functional import to_tensor  # noqa: PLC0415

        results: list[dict[str, float]] = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                # piq.brisque expects a float32 tensor in [0, 1] with shape (1, C, H, W)
                tensor: torch.Tensor = to_tensor(img).unsqueeze(0)
                raw: float = float(piq.brisque(tensor, data_range=1.0).item())
                # Clamp and invert: 0 raw → 100 quality; 100 raw → 0 quality
                raw_clamped = max(0.0, min(_RAW_MAX, raw))
                quality = _RAW_MAX - raw_clamped
            except (OSError, ValueError, RuntimeError, AssertionError) as exc:  # noqa: BLE001
                _logger.warning(
                    "Scorer %r failed on %s: %s",
                    self.spec().scorer_id,
                    path,
                    exc,
                )
                quality = 0.0
            results.append({"brisque_quality": quality})
        return results
