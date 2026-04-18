"""Simple heuristic scorer: blur/sharpness, luminosity/contrast, and noise.

This scorer merges three Pillow-only heuristics as named variants:

* ``blur``       — edge-filter variance → ``sharpness`` (0–100, higher is better)
* ``luminosity`` — mean / std of greyscale  → ``brightness`` and ``contrast`` (0–100)
* ``noise``      — Gaussian blur-difference RMS → ``noise`` (0–100, lower is better)

All variants require only Pillow and are always available when Pillow is
installed.  No ML model is needed.
"""

from __future__ import annotations

from pathlib import Path

from takeout_rater.scorers.base import BaseScorer, MetricSpec, ScorerSpec, VariantSpec

# ---------------------------------------------------------------------------
# Algorithm constants (kept identical to the original individual scorers)
# ---------------------------------------------------------------------------

# Raw edge-filter variance corresponding to a "100" sharpness score.
_MAX_RAW_VAR: float = 2000.0

# Maximum possible greyscale standard deviation (binary image, 50 % each).
_MAX_CONTRAST_STD: float = 127.5

# RMS difference value that maps to a score of 100 (very noisy).
_MAX_NOISE_RMS: float = 25.0


class SimpleScorer(BaseScorer):
    """Pillow-based heuristic scorer with blur, luminosity, and noise variants."""

    @classmethod
    def spec(cls) -> ScorerSpec:
        return ScorerSpec(
            scorer_id="simple",
            display_name="Simple",
            description=(
                "Three fast, Pillow-only image checks that need no model download: "
                "sharpness (how in-focus the image is), luminosity (brightness and tonal "
                "range), and noise (graininess from low-light or high-ISO shots). "
                "Pick a variant to measure the property you care about."
            ),
            technical_description=(
                "Three Pillow-based metrics: "
                "(1) Blur/Sharpness — variance of FIND_EDGES filter output; "
                "(2) Luminosity — mean greyscale intensity and greyscale standard deviation; "
                "(3) Noise — RMS of blur-difference image (Gaussian radius 1). "
                "All metrics are normalised to 0–100."
            ),
            version="1",
            variants=(
                VariantSpec(
                    variant_id="blur",
                    display_name="Blur / Sharpness",
                    description="Edge-filter variance via Pillow — detects how sharp or blurry a photo is.",
                    primary_metric_key="sharpness",
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
                ),
                VariantSpec(
                    variant_id="luminosity",
                    display_name="Luminosity / Contrast",
                    description="Mean and std of greyscale intensity via Pillow — measures brightness and tonal range.",
                    primary_metric_key="brightness",
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
                ),
                VariantSpec(
                    variant_id="noise",
                    display_name="Noise Level",
                    description="Gaussian blur-difference RMS via Pillow — estimates graininess.",
                    primary_metric_key="noise",
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
                ),
            ),
            default_variant_id="blur",
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
        """Score a batch of images using the selected variant.

        Args:
            image_paths: Paths to image files (can be thumbnails).
            variant_id: One of ``"blur"``, ``"luminosity"``, or ``"noise"``.
                Defaults to ``"blur"``.

        Returns:
            List of metric dicts.  Keys depend on the variant:

            * ``blur``       → ``{"sharpness": float}``
            * ``luminosity`` → ``{"brightness": float, "contrast": float}``
            * ``noise``      → ``{"noise": float}``
        """
        vid = variant_id or self.variant_id
        if vid == "blur":
            return self._score_blur(image_paths)
        if vid == "luminosity":
            return self._score_luminosity(image_paths)
        if vid == "noise":
            return self._score_noise(image_paths)
        raise ValueError(f"Unknown SimpleScorer variant: {vid!r}")

    # ------------------------------------------------------------------
    # Private per-variant implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _score_blur(image_paths: list[Path]) -> list[dict[str, float]]:
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

    @staticmethod
    def _score_luminosity(image_paths: list[Path]) -> list[dict[str, float]]:
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

    @staticmethod
    def _score_noise(image_paths: list[Path]) -> list[dict[str, float]]:
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
