"""NIMA scorer: Neural Image Assessment (Google, 2018).

NIMA (Talebi & Milanfar, 2018, IEEE T-IP) trains a CNN to predict the
*distribution* of human aesthetic ratings over the 10-point AVA scale.  The
mean of that distribution gives an overall quality or aesthetic score in [1, 10].

This implementation delegates to the ``pyiqa`` library (IQA-PyTorch), which
ships its own pre-trained NIMA checkpoints and manages weight downloads
automatically via ``~/.cache/pyiqa``.

Two variants are available:

- ``aesthetic`` variant — ``nima`` (inception_resnet_v2, AVA dataset).
  Outputs a score in [0, 10]; clamped to [1, 10] for display.
- ``technical`` variant — ``nima-koniq`` (inception_resnet_v2, KonIQ-10k
  dataset); more sensitive to blur, noise, and compression artefacts.
  Outputs a score in [0, 1] which is linearly rescaled to [1, 10].
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from takeout_rater.scorers.base import BaseScorer, MetricSpec, ScorerSpec, VariantSpec

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

#: pyiqa metric name for each scorer variant.
_VARIANT_PYIQA_METRIC: dict[str, str] = {
    "aesthetic": "nima",           # inception_resnet_v2, AVA dataset, score range 0–10
    "technical": "nima-koniq",     # inception_resnet_v2, KonIQ-10k dataset, score range ~0–1
}

#: Native output range (min, max) per variant.  Used to rescale to the
#: common [1, 10] display range before clamping.
_VARIANT_NATIVE_RANGE: dict[str, tuple[float, float]] = {
    "aesthetic": (0.0, 10.0),
    "technical": (0.0, 1.0),
}

#: Number of images to forward through the metric in a single call.
_SCORE_BATCH_SIZE = 32

#: Rating scale bounds used for clamping output.
_MIN_SCORE = 1.0
_MAX_SCORE = 10.0


# ---------------------------------------------------------------------------
# Scorer class
# ---------------------------------------------------------------------------


class NIMAScorer(BaseScorer):
    """NIMA (Neural Image Assessment) scorer — aesthetic and technical variants.

    Predicts human aesthetic / technical quality ratings on a 1–10 scale using
    pre-trained NIMA models from the ``pyiqa`` library (IQA-PyTorch).

    Two variants are available:

    - ``aesthetic``: ``nima`` (inception_resnet_v2, AVA dataset) — predicts
      perceived aesthetic quality (composition, colour, subject matter).
      Raw output is in [0, 10]; clamped to [1, 10].
    - ``technical``: ``nima-koniq`` (inception_resnet_v2, KonIQ-10k) — more
      sensitive to blur, noise, and compression artefacts.
      Raw output is in [0, 1]; rescaled to [1, 10].

    Both variants output a single float in [1, 10] stored under the metric
    key ``nima_score``.  A separate scorer run is required for each variant.
    Model weights are downloaded and cached by ``pyiqa`` in ``~/.cache/pyiqa``
    on first use.
    """

    def __init__(self, variant_id: str | None = None, **kwargs: Any) -> None:
        super().__init__(variant_id=variant_id, **kwargs)
        # Lazy-loaded pyiqa metric — populated by _ensure_loaded()
        self._model: Any = None

    # ------------------------------------------------------------------
    # Class-level API
    # ------------------------------------------------------------------

    @classmethod
    def spec(cls) -> ScorerSpec:
        return ScorerSpec(
            scorer_id="nima",
            display_name="NIMA Quality",
            description=(
                "Neural Image Assessment (NIMA) predicts how human viewers would rate a photo "
                "on a 1–10 scale, using a neural network trained on professional photo ratings. "
                "Two variants: aesthetic quality (composition, lighting, colour) and technical "
                "quality (sharpness, noise, exposure). "
                "Requires model download on first use (managed by pyiqa)."
            ),
            technical_description=(
                "Neural Image Assessment (NIMA, Google 2018). Predicts the mean "
                "human rating of image quality on a 1–10 scale. Backed by the "
                "pyiqa library: aesthetic variant uses nima (inception_resnet_v2, AVA dataset), "
                "technical variant uses nima-koniq (inception_resnet_v2, KonIQ-10k; rescaled to 1–10)."
            ),
            version="2",
            metrics=(
                MetricSpec(
                    key="nima_score",
                    display_name="NIMA Score",
                    description=(
                        "Mean predicted human rating (1–10, higher is better). "
                        "Represents the expected value of the rating distribution."
                    ),
                    min_value=1.0,
                    max_value=10.0,
                    higher_is_better=True,
                ),
            ),
            variants=(
                VariantSpec(
                    variant_id="aesthetic",
                    display_name="Aesthetic (AVA)",
                    description=(
                        "NIMA trained on the AVA dataset.  Predicts perceived aesthetic "
                        "quality: composition, colour harmony, and subject interest."
                    ),
                ),
                VariantSpec(
                    variant_id="technical",
                    display_name="Technical (KonIQ-10k)",
                    description=(
                        "NIMA trained on the KonIQ-10k dataset.  Predicts technical "
                        "quality: sharpness, noise level, and compression artefacts."
                    ),
                ),
            ),
            default_variant_id="aesthetic",
            requires_extras=(),
        )

    @classmethod
    def is_available(cls) -> bool:
        """Return ``True`` when required dependencies are importable."""
        try:
            import PIL  # noqa: F401
            import pyiqa  # noqa: F401
            import torch  # noqa: F401

            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Instantiate the pyiqa NIMA metric for the active variant (lazy init).

        Downloads model weights to ``~/.cache/pyiqa`` on first use via
        ``pyiqa.create_metric``.
        """
        if self._model is not None:
            return

        metric_name = _VARIANT_PYIQA_METRIC.get(self.variant_id)
        if metric_name is None:
            raise ValueError(
                f"Unknown NIMA variant '{self.variant_id}'. "
                f"Expected one of: {list(_VARIANT_PYIQA_METRIC)}"
            )

        import pyiqa  # noqa: PLC0415
        import torch  # noqa: PLC0415

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = pyiqa.create_metric(metric_name, device=str(device))

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_batch(
        self,
        image_paths: list[Path],
        *,
        variant_id: str | None = None,
    ) -> list[dict[str, float]]:
        """Score a batch of images using NIMA via pyiqa.

        Images are processed in chunks of :data:`_SCORE_BATCH_SIZE`.  Within
        each chunk, valid images are stacked into a single tensor and passed to
        the pyiqa metric.  If the metric returns a scalar (some metrics only
        support N=1), the chunk falls back to per-image processing.

        A failed image (``OSError``, ``ValueError``) defaults to the minimum
        score (``1.0``).

        Args:
            image_paths: Absolute paths to image files.
            variant_id: Ignored at runtime; the active variant is set at
                construction time via :meth:`create`.

        Returns:
            List (same length as *image_paths*) of dicts with key
            ``"nima_score"`` → float in ``[1.0, 10.0]``.
        """
        if not image_paths:
            return []

        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        from torchvision.transforms.functional import to_tensor  # noqa: PLC0415

        self._ensure_loaded()

        # Some variants (e.g. nima-koniq) output scores in a range other than
        # [1, 10].  Linearly rescale raw output to [1, 10] before clamping.
        native_min, native_max = _VARIANT_NATIVE_RANGE.get(self.variant_id, (0.0, 10.0))
        native_span = native_max - native_min
        display_span = _MAX_SCORE - _MIN_SCORE

        def _rescale(raw: float) -> float:
            if native_span == 0:
                return _MIN_SCORE
            scaled = (raw - native_min) / native_span * display_span + _MIN_SCORE
            return max(_MIN_SCORE, min(_MAX_SCORE, scaled))

        scores: list[float] = []

        for batch_start in range(0, len(image_paths), _SCORE_BATCH_SIZE):
            chunk = image_paths[batch_start : batch_start + _SCORE_BATCH_SIZE]
            tensors: list[torch.Tensor] = []
            valid_indices: list[int] = []

            for i, path in enumerate(chunk):
                try:
                    img = Image.open(path).convert("RGB")
                    tensors.append(to_tensor(img))
                    valid_indices.append(i)
                except (OSError, ValueError):
                    pass

            sub_scores = [_MIN_SCORE] * len(chunk)

            if tensors:
                try:
                    batch = torch.stack(tensors)
                    raw_out = self._model(batch)
                    # pyiqa metrics return (N,) or (N, 1); fall back if scalar.
                    if raw_out.dim() == 0:
                        raise RuntimeError("scalar output — fall back to per-image")
                    score_values: list[float] = raw_out.view(-1).tolist()
                    for j, idx in enumerate(valid_indices):
                        sub_scores[idx] = _rescale(float(score_values[j]))
                except RuntimeError:
                    # Fallback: score each valid image individually (e.g. scalar output)
                    for _j, idx in enumerate(valid_indices):
                        path = chunk[idx]
                        try:
                            img = Image.open(path).convert("RGB")
                            tensor = to_tensor(img).unsqueeze(0)
                            raw = float(self._model(tensor).item())
                            sub_scores[idx] = _rescale(raw)
                        except (OSError, ValueError, RuntimeError):
                            pass

            scores.extend(sub_scores)

        return [{"nima_score": s} for s in scores]
