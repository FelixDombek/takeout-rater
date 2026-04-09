"""Luminosity scorer: measures image brightness and contrast (Pillow-based).

``brightness`` is the mean greyscale intensity mapped to 0–100.  A value of
50 corresponds to mid-grey; higher values indicate brighter images.

``contrast`` is the standard deviation of greyscale pixel values normalised to
0–100.  A uniform (single-colour) image scores 0; a half-black / half-white
binary image scores 100.

Both metrics require only Pillow and are always available when Pillow is
installed.
"""

from __future__ import annotations

from pathlib import Path

from takeout_rater.scorers.base import BaseScorer, MetricSpec, ScorerSpec, VariantSpec

# Maximum possible greyscale standard deviation (binary image, 50 % each).
_MAX_CONTRAST_STD: float = 127.5


class LuminosityScorer(BaseScorer):
    """Measures image brightness and contrast (Pillow-based, no ML model)."""

    @classmethod
    def spec(cls) -> ScorerSpec:
        return ScorerSpec(
            scorer_id="luminosity",
            display_name="Luminosity / Contrast",
            description=(
                "Measures two basic properties of a photo's lighting: how bright it is overall "
                "(brightness) and how much tonal variety it contains (contrast). A very dark or "
                "washed-out photo scores low on brightness; a flat, single-colour image scores "
                "low on contrast. No model download required."
            ),
            technical_description=(
                "Measures the overall brightness (mean luminance) and contrast "
                "(greyscale standard deviation) of an image. Both metrics are "
                "computed from the greyscale representation using Pillow and "
                "require no ML model."
            ),
            version="1",
            metrics=(
                MetricSpec(
                    key="brightness",
                    display_name="Brightness",
                    description=(
                        "Mean greyscale intensity normalised to 0–100.  "
                        "50 is mid-grey; higher values indicate brighter images."
                    ),
                    min_value=0.0,
                    max_value=100.0,
                    higher_is_better=True,
                ),
                MetricSpec(
                    key="contrast",
                    display_name="Contrast",
                    description=(
                        "Standard deviation of greyscale pixel values normalised to 0–100.  "
                        "Higher values indicate images with a wider tonal range."
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
                    description="Mean and std of greyscale intensity via Pillow (no ML model).",
                ),
            ),
            default_variant_id="default",
            requires_extras=(),
        )

    @classmethod
    def is_available(cls) -> bool:
        try:
            from PIL import Image, ImageStat  # noqa: F401

            return True
        except ImportError:
            return False

    def score_batch(
        self,
        image_paths: list[Path],
        *,
        variant_id: str | None = None,
    ) -> list[dict[str, float]]:
        """Score a batch of images for brightness and contrast.

        Args:
            image_paths: Paths to image files (can be thumbnails).
            variant_id: Ignored; only one variant exists.

        Returns:
            List of dicts with keys ``"brightness"`` and ``"contrast"``,
            each a float in ``[0, 100]``.
            If a file cannot be opened, both metrics are set to ``0.0``.
        """
        from PIL import Image, ImageStat  # noqa: PLC0415

        results: list[dict[str, float]] = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("L")
                stat = ImageStat.Stat(img)
                mean: float = stat.mean[0]
                stddev: float = stat.stddev[0]
                brightness = mean / 255.0 * 100.0
                contrast = min(stddev / _MAX_CONTRAST_STD, 1.0) * 100.0
            except (OSError, ValueError):
                brightness = 0.0
                contrast = 0.0
            results.append({"brightness": brightness, "contrast": contrast})
        return results
