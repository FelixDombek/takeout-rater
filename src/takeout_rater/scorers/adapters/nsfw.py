"""NSFW image detector scorer.

Uses the ``Falconsai/nsfw_image_detection`` Vision Transformer classifier to
predict the probability that an image is Not Safe For Work (NSFW).

The output metric ``nsfw`` is a probability in [0, 1]:

- Values close to **0** indicate a safe-for-work image.
- Values close to **1** indicate a potentially unsafe image.

The model weights (~330 MB) are downloaded from the HuggingFace Hub on first
use and cached by ``transformers`` (``~/.cache/huggingface``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from takeout_rater.scorers.base import BaseScorer, MetricSpec, ScorerSpec, VariantSpec

# ---------------------------------------------------------------------------
# Model identifier
# ---------------------------------------------------------------------------

#: HuggingFace model identifier for the ViT-based NSFW classifier.
_HF_MODEL = "Falconsai/nsfw_image_detection"


# ---------------------------------------------------------------------------
# Scorer class
# ---------------------------------------------------------------------------


class NSFWScorer(BaseScorer):
    """NSFW image classifier using a Vision Transformer (ViT).

    Predicts the probability that each image is Not Safe For Work (NSFW) on a
    0–1 scale.  Lower scores are safer.

    The classifier is a fine-tuned ViT model hosted at
    ``Falconsai/nsfw_image_detection`` on HuggingFace Hub.
    """

    def __init__(self, variant_id: str | None = None, **kwargs: Any) -> None:
        super().__init__(variant_id=variant_id, **kwargs)
        # Lazy-loaded state — populated by _ensure_loaded()
        self._pipeline: Any = None

    # ------------------------------------------------------------------
    # Class-level API
    # ------------------------------------------------------------------

    @classmethod
    def spec(cls) -> ScorerSpec:
        return ScorerSpec(
            scorer_id="nsfw",
            display_name="NSFW Detector",
            description=(
                "Predicts the probability that an image is Not Safe For Work (NSFW) "
                "using a Vision Transformer (ViT) fine-tuned classifier. "
                "Scores close to 0 are safe; scores close to 1 are likely NSFW."
            ),
            metrics=(
                MetricSpec(
                    key="nsfw",
                    display_name="NSFW Score",
                    description=(
                        "Probability that the image is NSFW (0 = safe, 1 = NSFW). "
                        "Lower is better / safer."
                    ),
                    min_value=0.0,
                    max_value=1.0,
                    higher_is_better=False,
                ),
            ),
            variants=(
                VariantSpec(
                    variant_id="falconsai_vit",
                    display_name="Falconsai ViT",
                    description=(
                        "ViT-based NSFW image classifier "
                        "(Falconsai/nsfw_image_detection on HuggingFace)."
                    ),
                ),
            ),
            default_variant_id="falconsai_vit",
            requires_extras=(),
        )

    @classmethod
    def is_available(cls) -> bool:
        """Return ``True`` when required runtime dependencies are importable."""
        try:
            import PIL  # noqa: F401
            import torch  # noqa: F401
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load the image-classification pipeline on first call (lazy init).

        Downloads model weights to ``~/.cache/huggingface`` if not already
        present.
        """
        if self._pipeline is not None:
            return

        import torch  # noqa: PLC0415
        from transformers import pipeline  # noqa: PLC0415

        device = 0 if torch.cuda.is_available() else -1
        self._pipeline = pipeline("image-classification", model=_HF_MODEL, device=device)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_batch(
        self,
        image_paths: list[Path],
        *,
        variant_id: str | None = None,
    ) -> list[dict[str, float]]:
        """Score a batch of images for NSFW content.

        Each image is opened via Pillow and passed through the ViT classifier.
        If a file cannot be opened or processed the score defaults to ``0.0``
        (treated as safe).

        Args:
            image_paths: Paths to image files.  Thumbnails (512 px) are
                sufficient and load much faster than originals.
            variant_id: Ignored; only one variant exists.

        Returns:
            A list (same length as *image_paths*) of dicts with a single key
            ``"nsfw"`` → float in ``[0.0, 1.0]``.
        """
        if not image_paths:
            return []

        from PIL import Image  # noqa: PLC0415

        self._ensure_loaded()

        results: list[dict[str, float]] = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                preds: list[dict[str, Any]] = self._pipeline(img)
                # preds is a list of {"label": str, "score": float}
                nsfw_score: float = next(
                    (float(p["score"]) for p in preds if p["label"].lower() == "nsfw"),
                    0.0,
                )
            except (OSError, ValueError, RuntimeError):
                nsfw_score = 0.0

            results.append({"nsfw": float(nsfw_score)})

        return results
