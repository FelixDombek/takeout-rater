"""Image style classifier using the CafeAI cafe_style model.

``cafeai/cafe_style`` is a BEiT Vision Transformer classifier that assigns
probability scores to five mutually-exclusive style categories:

- **real_life** — a real photograph.
- **anime** — anime or manga artwork.
- **manga_like** — manga/illustration-style artwork.
- **3d** — 3D rendered image.
- **other** — anything outside the preceding categories.

Each category is exposed as a separate metric with a probability in [0, 1].
The five probabilities sum to approximately 1.0.

The model weights (~370 MB) are downloaded from the HuggingFace Hub on first
use and cached by ``transformers`` (``~/.cache/huggingface``).

The primary use-case for Google Photos libraries is the ``real_life`` metric:
filtering for values close to 1.0 lets users see only real photographs and
exclude screenshots, memes, artwork, or other non-photographic content that
may have accumulated in their archive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from takeout_rater.scoring.scorers.base import BaseScorer, MetricSpec, ScorerSpec, VariantSpec

# ---------------------------------------------------------------------------
# Model identifier
# ---------------------------------------------------------------------------

#: HuggingFace model identifier for the CafeAI style classifier.
_HF_MODEL = "cafeai/cafe_style"

#: Number of images to forward through the ViT classifier in a single GPU
#: pass.  The transformers ``pipeline`` handles internal chunking when the
#: input list is longer than this value.
_SCORE_BATCH_SIZE = 32

#: Number of top-k labels to request from the pipeline.  The model has
#: exactly 5 classes; requesting all 5 ensures every category is present.
_TOP_K = 5

#: Source: ``cafeai/cafe_style`` config.json ``id2label``:
#:   {"0": "anime", "1": "real_life", "2": "3d", "3": "manga_like", "4": "other"}
_LABELS: list[str] = ["anime", "real_life", "3d", "manga_like", "other"]

#: Safe default value returned for any metric when scoring fails.
_DEFAULT_SCORE = 0.0


def _empty_result() -> dict[str, float]:
    """Return a zeroed-out result dict for all known metrics."""
    return {metric_key: _DEFAULT_SCORE for metric_key in _LABELS}


# ---------------------------------------------------------------------------
# Scorer class
# ---------------------------------------------------------------------------


class CafeStyleScorer(BaseScorer):
    """Image style classifier using the CafeAI ViT style model.

    Produces five probability metrics — one per style category — that sum to
    approximately 1.0.  The ``real_life`` metric is the most useful for
    Google Photos libraries: values close to 1.0 indicate a real photograph,
    while low values indicate artwork, screenshots, or synthetic imagery.

    The classifier is a fine-tuned ViT hosted at ``cafeai/cafe_style`` on
    HuggingFace Hub.
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
            scorer_id="cafe_style",
            display_name="Style Classifier",
            description=(
                "Classifies each image into one of five style categories using a BEiT Vision Transformer: "
                "real_life (photo), anime, manga_like (illustrations/frames/speech bubbles), 3d, and other. "
                "Each category gets a probability score from 0 to 1, with a sum of about 1.0 for the complete probability distribution. "
                "Requires ~370 MB model download."
            ),
            technical_description=(
                "Applies the cafeai/cafe_style BEiT Vision Transformer classifier to "
                "predict the probability distribution over its native labels: real_life, "
                "anime, 3d, manga_like, and other. These labels are used directly as "
                "metric keys."
            ),
            version="2",
            variants=(
                VariantSpec(
                    variant_id="cafeai_v1",
                    display_name="CafeAI v1",
                    description=(
                        "BEiT-based style classifier (cafeai/cafe_style on HuggingFace). "
                        "Trained to distinguish real-life photos, anime, manga/illustration, "
                        "3D renders, and other CGI styles."
                    ),
                    metrics=(
                        MetricSpec(
                            key="real_life",
                            display_name="Photo",
                            description=(
                                "Probability that the image is a real photograph (0–1). "
                                "Values close to 1 indicate a genuine photo; "
                                "values close to 0 suggest artwork or synthetic imagery."
                            ),
                            min_value=0.0,
                            max_value=1.0,
                            higher_is_better=True,
                        ),
                        MetricSpec(
                            key="anime",
                            display_name="Anime",
                            description=(
                                "Probability that the image is in anime or manga style (0–1)."
                            ),
                            min_value=0.0,
                            max_value=1.0,
                            higher_is_better=True,
                        ),
                        MetricSpec(
                            key="manga_like",
                            display_name="Manga-like",
                            description=(
                                "Probability that the image is an illustration or has comic-like frames and speech bubbles (0-1)."
                            ),
                            min_value=0.0,
                            max_value=1.0,
                            higher_is_better=True,
                        ),
                        MetricSpec(
                            key="3d",
                            display_name="3D Render",
                            description=(
                                "Probability that the image is a 3D rendered image (0–1)."
                            ),
                            min_value=0.0,
                            max_value=1.0,
                            higher_is_better=True,
                        ),
                        MetricSpec(
                            key="other",
                            display_name="Other",
                            description=(
                                "Probability that the image is not one of the above categories (0–1)."
                            ),
                            min_value=0.0,
                            max_value=1.0,
                            higher_is_better=True,
                        ),
                    ),
                ),
            ),
            default_variant_id="cafeai_v1",
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
        self._pipeline = pipeline(
            "image-classification",
            model=_HF_MODEL,
            device=device,
            top_k=_TOP_K,
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_batch(
        self,
        image_paths: list[Path],
        *,
        variant_id: str | None = None,
    ) -> list[dict[str, float]]:
        """Score a batch of images for style category probabilities.

        All images are opened first and then passed to the ViT classifier as a
        single list so the ``transformers`` pipeline can batch them internally
        (controlled by :data:`_SCORE_BATCH_SIZE`).  This amortises pipeline
        overhead and keeps the GPU busy between images.

        Images that cannot be opened default to all-zero scores.

        Args:
            image_paths: Paths to image files.  Thumbnails (512 px) are
                sufficient and load much faster than originals.
            variant_id: Ignored; only one variant exists.

        Returns:
            A list (same length as *image_paths*) of dicts with five keys:
            ``"real_life"``, ``"anime"``, ``"manga_like"``,
            ``"3d"``, ``"other"`` — each a float in ``[0.0, 1.0]``.
        """
        if not image_paths:
            return []

        from PIL import Image  # noqa: PLC0415

        self._ensure_loaded()

        # Pre-load all images; keep input positions so failures can stay zeroed.
        imgs: list[Any] = []
        for path in image_paths:
            try:
                imgs.append(Image.open(path).convert("RGB"))
            except (OSError, ValueError, RuntimeError):
                imgs.append(None)

        valid_imgs = [img for img in imgs if img is not None]
        valid_indices = [i for i, img in enumerate(imgs) if img is not None]

        results: list[dict[str, float]] = [_empty_result() for _ in image_paths]

        if valid_imgs:
            try:
                all_preds: list[list[dict[str, Any]]] = self._pipeline(
                    valid_imgs, batch_size=_SCORE_BATCH_SIZE
                )
                for j, idx in enumerate(valid_indices):
                    results[idx] = _preds_to_scores(all_preds[j])
            except RuntimeError:  # noqa: BLE001
                # Fallback: score each image individually if bulk call fails.
                for _j, idx in enumerate(valid_indices):
                    try:
                        preds = self._pipeline(imgs[idx])
                        results[idx] = _preds_to_scores(preds)
                    except (OSError, ValueError, RuntimeError):
                        pass

        return results


def _preds_to_scores(preds: list[dict[str, Any]]) -> dict[str, float]:
    """Convert pipeline predictions to model-label metric scores.

    Args:
        preds: List of ``{"label": str, "score": float}`` dicts returned by
            the image-classification pipeline.

    Returns:
        Dict mapping each native model label to a probability in [0, 1].
        Unknown labels are silently ignored; missing labels default to 0.0.
    """
    scores = _empty_result()
    for pred in preds:
        label = str(pred.get("label", "")).lower()
        if label in scores:
            scores[label] = float(pred["score"])
    return scores
