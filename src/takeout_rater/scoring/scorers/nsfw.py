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

from takeout_rater.scoring.scorers.base import (
    BaseScorer,
    MetricSpec,
    ScorerSpec,
    VariantSpec,
    _run_pipelined_batches,
)

# ---------------------------------------------------------------------------
# Model identifier
# ---------------------------------------------------------------------------

#: HuggingFace model identifier for the ViT-based NSFW classifier.
_HF_MODEL = "Falconsai/nsfw_image_detection"

#: Number of images to forward through the ViT classifier in a single GPU
#: pass.  The transformers ``pipeline`` handles internal chunking when the
#: input list is longer than this value.
_SCORE_BATCH_SIZE = 32

#: Number of preprocessed batches to keep ready ahead of GPU inference.
_PREFETCH_BATCHES = 2


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
                "Classifies whether a photo is safe for work or contains mature content, "
                "using a neural network fine-tuned specifically for this task. Scores close "
                "to 0 are safe; scores close to 1 suggest the image may be inappropriate. "
                "Requires ~330 MB model download."
            ),
            technical_description=(
                "Predicts the probability that an image is Not Safe For Work (NSFW) "
                "using a Vision Transformer (ViT) fine-tuned classifier. "
                "Scores close to 0 are safe; scores close to 1 are likely NSFW."
            ),
            version="1",
            variants=(
                VariantSpec(
                    variant_id="falconsai_vit",
                    display_name="Falconsai ViT",
                    description=(
                        "ViT-based NSFW image classifier "
                        "(Falconsai/nsfw_image_detection on HuggingFace)."
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

        Images are processed in chunks of :data:`_SCORE_BATCH_SIZE`.  Within
        each chunk, valid images are passed to the ViT classifier as a list so
        the ``transformers`` pipeline can forward them together in one GPU pass,
        amortising pipeline overhead across images.

        CPU image loading and GPU inference are **pipelined**: while the GPU
        processes chunk N, a background thread opens chunk N+1, so that neither
        device sits idle waiting for the other.

        Images that cannot be opened default to a score of ``0.0`` (safe).

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

        chunks = [
            image_paths[start : start + _SCORE_BATCH_SIZE]
            for start in range(0, len(image_paths), _SCORE_BATCH_SIZE)
        ]

        def _preprocess(chunk: list[Path]) -> tuple[list[Any], set[int]]:
            images: list[Any] = []
            failed: set[int] = set()
            for i, path in enumerate(chunk):
                try:
                    images.append(Image.open(path).convert("RGB"))
                except (OSError, ValueError, RuntimeError):
                    failed.add(i)
                    images.append(None)
            return images, failed

        def _infer(
            images: list[Any], failed: set[int], chunk: list[Path]
        ) -> list[dict[str, float]]:
            valid_imgs = [img for img in images if img is not None]
            valid_indices = [i for i, img in enumerate(images) if img is not None]
            nsfw_scores = [0.0] * len(chunk)

            if valid_imgs:
                try:
                    # Pass the full list; the pipeline batches it using batch_size internally.
                    all_preds: list[list[dict[str, Any]]] = self._pipeline(
                        valid_imgs, batch_size=_SCORE_BATCH_SIZE
                    )
                    for j, idx in enumerate(valid_indices):
                        preds = all_preds[j]
                        nsfw_scores[idx] = next(
                            (float(p["score"]) for p in preds if p["label"].lower() == "nsfw"),
                            0.0,
                        )
                except RuntimeError:  # noqa: BLE001
                    # Fallback: score each image individually if bulk call fails.
                    for j, idx in enumerate(valid_indices):
                        try:
                            preds = self._pipeline(valid_imgs[j])
                            nsfw_scores[idx] = next(
                                (float(p["score"]) for p in preds if p["label"].lower() == "nsfw"),
                                0.0,
                            )
                        except (OSError, ValueError, RuntimeError):
                            pass

            return [{"nsfw": s} for s in nsfw_scores]

        return _run_pipelined_batches(chunks, _preprocess, _infer, prefetch=_PREFETCH_BATCHES)
