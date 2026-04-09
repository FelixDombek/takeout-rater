"""Noise scorer: estimates image noise level using a blur-difference method.

The algorithm works by:

1. Converting the image to greyscale.
2. Applying a Gaussian blur (radius 1) to obtain a smoothed reference.
3. Computing the absolute pixel-wise difference between the original and the
   smoothed image using :func:`PIL.ImageChops.difference`.
4. Using the RMS (root-mean-square) of the difference image as a proxy for the
   noise amplitude.  RMS is preferred over plain mean because it is sensitive
   to both large individual-pixel errors and distributed noise.

The raw RMS value is normalised to a 0–100 scale by clamping at
``_MAX_NOISE_RMS``.  **Lower values indicate less noise** (i.e.
``higher_is_better`` is ``False``).

This scorer requires only Pillow and is always available when Pillow is
installed.
"""

from __future__ import annotations

from pathlib import Path

from takeout_rater.scorers.base import BaseScorer, MetricSpec, ScorerSpec, VariantSpec

# RMS difference value that maps to a score of 100 (very noisy).
# Typical clean images have RMS < 5; visibly noisy images are often 10–25.
_MAX_NOISE_RMS: float = 25.0


class NoiseScorer(BaseScorer):
    """Estimates image noise using a Gaussian blur-difference method (Pillow-based)."""

    @classmethod
    def spec(cls) -> ScorerSpec:
        return ScorerSpec(
            scorer_id="noise",
            display_name="Noise Level",
            description=(
                "Detects graininess and digital noise in a photo, which often appears in "
                "low-light or high-sensitivity shots. It works by lightly smoothing the image "
                "and measuring how much detail disappeared — noise shows up as fine random "
                "speckle that a gentle blur removes. Lower scores mean a cleaner image. "
                "No model download required."
            ),
            version="1",
            metrics=(
                MetricSpec(
                    key="noise",
                    display_name="Noise",
                    description=(
                        "RMS of the blur-difference image, normalised to 0–100.  "
                        "Lower values indicate less noise / a cleaner image."
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
                    description="Gaussian blur-difference RMS via Pillow (no ML model).",
                ),
            ),
            default_variant_id="default",
            requires_extras=(),
        )

    @classmethod
    def is_available(cls) -> bool:
        try:
            from PIL import Image, ImageChops, ImageFilter, ImageStat  # noqa: F401

            return True
        except ImportError:
            return False

    def score_batch(
        self,
        image_paths: list[Path],
        *,
        variant_id: str | None = None,
    ) -> list[dict[str, float]]:
        """Score a batch of images for noise level.

        Args:
            image_paths: Paths to image files (can be thumbnails).
            variant_id: Ignored; only one variant exists.

        Returns:
            List of dicts with key ``"noise"`` → float in ``[0, 100]``.
            If a file cannot be opened, ``noise`` is set to ``0.0``.
        """
        from PIL import Image, ImageChops, ImageFilter, ImageStat  # noqa: PLC0415

        results: list[dict[str, float]] = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("L")
                blurred = img.filter(ImageFilter.GaussianBlur(radius=1))
                diff = ImageChops.difference(img, blurred)
                rms: float = ImageStat.Stat(diff).rms[0]
                noise = min(rms / _MAX_NOISE_RMS, 1.0) * 100.0
            except (OSError, ValueError):
                noise = 0.0
            results.append({"noise": noise})
        return results
