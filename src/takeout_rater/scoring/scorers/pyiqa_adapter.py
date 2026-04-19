"""pyiqa adapter: multi-metric IQA via the IQA-PyTorch library.

This adapter wraps the ``pyiqa`` library (IQA-PyTorch, Chaofeng Chen et al.)
to expose three state-of-the-art no-reference image quality metrics in a
single scorer:

- **MUSIQ** (Multi-Scale Image Quality Transformer, Google 2021): A Vision
  Transformer fine-tuned on multiple IQA datasets (AVA, KonIQ, SPAQ).
  Processes images at native resolution without resizing; ~100 MB weights.

- **TOPIQ** (Top-down Perceptual IQA, 2024): State-of-the-art NR-IQA using
  semantic-aware top-down attention on a CLIP-pretrained backbone.

- **NIQE** (Natural Image Quality Evaluator, 2013): Opinion-free statistical
  quality estimator.  No training data required; uses pre-fitted parameters
  from ``pyiqa``.

Each metric is available as a separate variant.  Scores are normalised to
[0, 1] (higher = better) regardless of each metric's native output range:

- MUSIQ: native [0, 100]; divided by 100.
- TOPIQ: native [0, 1]; used as-is.
- NIQE: native [0, ∞], lower = better; inverted via ``1 / (1 + score)``.

The ``pyiqa`` library manages model weight downloads to its own cache
directory (``~/.cache/pyiqa``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from takeout_rater.scoring.scorers.base import (
    BaseScorer,
    MetricSpec,
    ScorerSpec,
    VariantSpec,
    _run_pipelined_batches,
)

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

#: pyiqa metric name for each variant.
_VARIANT_METRIC: dict[str, str] = {
    "musiq": "musiq",
    "topiq_nr": "topiq_nr",
    "niqe": "niqe",
}


def _to_higher_is_better(variant_id: str, raw: float) -> float:
    """Normalise *raw* to a [0, 1] higher-is-better scale.

    Transformations applied per variant:
    - ``musiq``: divide by 100 (native range 0–100).
    - ``topiq_nr``: no change (native range 0–1).
    - ``niqe``: invert via ``1 / (1 + raw)`` (native range 0–∞, lower = better).
    """
    if variant_id == "musiq":
        return max(0.0, min(1.0, raw / 100.0))
    if variant_id == "topiq_nr":
        return max(0.0, min(1.0, raw))
    if variant_id == "niqe":
        # NIQE ≈ 0 is best; map to [0, 1] via monotone inversion.
        return 1.0 / (1.0 + max(0.0, raw))
    return max(0.0, min(1.0, raw))


#: Number of images to stack into a single metric call.  Most pyiqa metrics
#: accept a ``(N, C, H, W)`` batch tensor.  Tune down if you hit VRAM OOM.
_SCORE_BATCH_SIZE = 32

#: Number of preprocessed batches to keep ready ahead of GPU inference.
_PREFETCH_BATCHES = 2


# ---------------------------------------------------------------------------
# Scorer class
# ---------------------------------------------------------------------------


class PyIQAScorer(BaseScorer):
    """Multi-metric IQA scorer backed by the IQA-PyTorch (pyiqa) library.

    Exposes MUSIQ, TOPIQ-NR, and NIQE as three variants of a single scorer.
    All output a normalised quality score in [0, 1] (higher = better).

    The active variant is selected at construction time via
    :meth:`~takeout_rater.scoring.scorers.base.BaseScorer.create`.

    Example::

        scorer = PyIQAScorer.create(variant_id="musiq")
        results = scorer.score_batch(paths)
    """

    def __init__(self, variant_id: str | None = None, **kwargs: Any) -> None:
        super().__init__(variant_id=variant_id, **kwargs)
        # Lazy-loaded state — populated by _ensure_loaded()
        self._metric: Any = None

    # ------------------------------------------------------------------
    # Class-level API
    # ------------------------------------------------------------------

    @classmethod
    def spec(cls) -> ScorerSpec:
        return ScorerSpec(
            scorer_id="pyiqa",
            display_name="IQA-PyTorch Metrics",
            description=(
                "A family of state-of-the-art no-reference image quality metrics. "
                "Three variants: MUSIQ (a transformer model trained on multiple quality "
                "datasets, ~100 MB), TOPIQ (semantic-aware quality via a CLIP backbone), "
                "and NIQE (a statistics-based approach that needs no training data). "
                "All scores are normalised so higher means better."
            ),
            technical_description=(
                "Collection of state-of-the-art no-reference image quality metrics "
                "from the IQA-PyTorch (pyiqa) library. Includes MUSIQ (Google, 2021), "
                "TOPIQ-NR (2024), and NIQE (2013). All scores are normalised to "
                "[0, 1] with higher values indicating better quality."
            ),
            version="1",
            variants=(
                VariantSpec(
                    variant_id="musiq",
                    display_name="MUSIQ (Google, 2021)",
                    description=(
                        "Multi-Scale Image Quality Transformer.  Vision Transformer "
                        "fine-tuned on AVA, KonIQ, and SPAQ datasets.  Processes "
                        "images at native resolution without distorting crops."
                    ),
                    metrics=(
                        MetricSpec(
                            key="iqa_quality",
                            display_name="IQA Quality",
                            description=(
                                "Normalised image quality score (0–1, higher is better). "
                                "Derived from the selected pyiqa metric variant."
                            ),
                            min_value=0.0,
                            max_value=1.0,
                            higher_is_better=True,
                        ),
                    ),
                ),
                VariantSpec(
                    variant_id="topiq_nr",
                    display_name="TOPIQ-NR (2024)",
                    description=(
                        "Top-down Perceptual IQA.  State-of-the-art no-reference "
                        "scorer using semantic-aware attention on a CLIP backbone."
                    ),
                    metrics=(
                        MetricSpec(
                            key="iqa_quality",
                            display_name="IQA Quality",
                            description=(
                                "Normalised image quality score (0–1, higher is better). "
                                "Derived from the selected pyiqa metric variant."
                            ),
                            min_value=0.0,
                            max_value=1.0,
                            higher_is_better=True,
                        ),
                    ),
                ),
                VariantSpec(
                    variant_id="niqe",
                    display_name="NIQE (2013)",
                    description=(
                        "Natural Image Quality Evaluator.  Opinion-free statistical "
                        "quality estimator using pre-fitted multivariate Gaussian "
                        "patch statistics.  No training data required."
                    ),
                    metrics=(
                        MetricSpec(
                            key="iqa_quality",
                            display_name="IQA Quality",
                            description=(
                                "Normalised image quality score (0–1, higher is better). "
                                "Derived from the selected pyiqa metric variant."
                            ),
                            min_value=0.0,
                            max_value=1.0,
                            higher_is_better=True,
                        ),
                    ),
                ),
            ),
            default_variant_id="musiq",
            requires_extras=(),
        )

    @classmethod
    def is_available(cls) -> bool:
        """Return ``True`` when ``pyiqa`` and its core dependencies are importable."""
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
        """Instantiate the pyiqa metric for the active variant (lazy init).

        Downloads model weights to ``~/.cache/pyiqa`` on first use via
        ``pyiqa.create_metric``.
        """
        if self._metric is not None:
            return

        import pyiqa  # noqa: PLC0415
        import torch  # noqa: PLC0415

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        metric_name = _VARIANT_METRIC.get(self.variant_id, self.variant_id)
        self._metric = pyiqa.create_metric(metric_name, device=str(device))

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_batch(
        self,
        image_paths: list[Path],
        *,
        variant_id: str | None = None,
    ) -> list[dict[str, float]]:
        """Score a batch of images using the selected pyiqa metric.

        Images are processed in chunks of :data:`_SCORE_BATCH_SIZE`.  Within
        each chunk, valid images are stacked into a single ``(N, C, H, W)``
        tensor and passed to the pyiqa metric in one call, amortising model
        overhead across many images.

        CPU preprocessing (PIL decode + ``to_tensor``) and GPU inference are
        **pipelined**: while the GPU processes chunk N, a background thread
        preprocesses chunk N+1 so that neither device sits idle waiting for
        the other.

        If the pyiqa metric returns a per-image tensor (shape ``(N,)`` or
        ``(N, 1)``), each value is normalised individually.  If it returns a
        scalar (some metrics only support N=1), the chunk falls back to
        per-image processing.

        A failed image (``OSError``, ``ValueError``) defaults to ``0.0``.

        Args:
            image_paths: Absolute paths to image files.
            variant_id: Ignored at runtime; the active variant is set at
                construction time via :meth:`create`.

        Returns:
            List (same length as *image_paths*) of dicts with key
            ``"iqa_quality"`` → float in ``[0.0, 1.0]``.
        """
        if not image_paths:
            return []

        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        from torchvision.transforms.functional import to_tensor  # noqa: PLC0415

        self._ensure_loaded()

        chunks = [
            image_paths[start : start + _SCORE_BATCH_SIZE]
            for start in range(0, len(image_paths), _SCORE_BATCH_SIZE)
        ]

        def _preprocess(chunk: list[Path]) -> tuple[list[Any], set[int]]:
            tensors: list[Any] = []
            failed: set[int] = set()
            for i, path in enumerate(chunk):
                try:
                    img = Image.open(path).convert("RGB")
                    tensors.append(to_tensor(img))
                except (OSError, ValueError):
                    failed.add(i)
                    tensors.append(None)
            return tensors, failed

        def _infer(
            tensors: list[Any], failed: set[int], chunk: list[Path]
        ) -> list[dict[str, float]]:
            valid_tensors = [t for t in tensors if t is not None]
            valid_indices = [i for i, t in enumerate(tensors) if t is not None]
            sub_scores = [0.0] * len(chunk)

            if valid_tensors:
                try:
                    batch = torch.stack(valid_tensors)  # (N, C, H, W) — on CPU for pyiqa
                    raw_out = self._metric(batch)
                    # pyiqa metrics return (N, 1) or (N,); fall back if scalar.
                    if raw_out.dim() == 0:
                        raise RuntimeError("scalar output — fall back to per-image")
                    raw_values: list[float] = raw_out.squeeze(-1).tolist()
                    for j, idx in enumerate(valid_indices):
                        sub_scores[idx] = _to_higher_is_better(self.variant_id, raw_values[j])
                except (RuntimeError, IndexError):
                    # Fallback: score each image individually
                    for _j, idx in enumerate(valid_indices):
                        path = chunk[idx]
                        try:
                            img = Image.open(path).convert("RGB")
                            tensor = to_tensor(img).unsqueeze(0)
                            raw = float(self._metric(tensor).item())
                            sub_scores[idx] = _to_higher_is_better(self.variant_id, raw)
                        except (OSError, ValueError, RuntimeError):
                            pass

            return [{"iqa_quality": s} for s in sub_scores]

        return _run_pipelined_batches(chunks, _preprocess, _infer, prefetch=_PREFETCH_BATCHES)
