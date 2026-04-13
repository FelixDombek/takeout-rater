"""Image style classifier using the CafeAI cafe_style model.

``cafeai/cafe_style`` is a Vision Transformer classifier that assigns
probability scores to five mutually-exclusive style categories:

- **photo** — a real photograph.
- **anime** — anime or manga artwork.
- **illustration** — general digital or hand-drawn illustration.
- **3d** — 3D rendered image.
- **CGI** — computer-generated imagery (catch-all for other CGI styles).

Each category is exposed as a separate metric with a probability in [0, 1].
The five probabilities sum to approximately 1.0.

The model weights (~370 MB) are downloaded from the HuggingFace Hub on first
use and cached by ``transformers`` (``~/.cache/huggingface``).

The primary use-case for Google Photos libraries is the ``style_photo`` metric:
filtering for values close to 1.0 lets users see only real photographs and
exclude screenshots, memes, artwork, or other non-photographic content that
may have accumulated in their archive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from takeout_rater.scorers.base import BaseScorer, MetricSpec, ScorerSpec, VariantSpec

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

#: Mapping from the model's lowercase label strings to our metric keys.
#: Labels returned by the pipeline are lowercased before lookup.
#: The cafeai/cafe_style model (BEiT, id2label from config.json) uses
#: "anime", "real_life", "3d", "manga_like", and "other" as its five class
#: labels.  Generic aliases ("photo", "illustration", "cgi") are kept for
#: test fixtures and any future model revision that uses different names.
_LABEL_TO_METRIC: dict[str, str] = {
    # Actual cafeai/cafe_style model labels (from config.json id2label)
    "anime": "style_anime",
    "real_life": "style_photo",
    "3d": "style_3d",
    "manga_like": "style_illustration",
    "other": "style_cgi",
    # Generic aliases kept for unit-test fixtures and future model variants
    "photo": "style_photo",
    "illustration": "style_illustration",
    "cgi": "style_cgi",
}

#: Keyword patterns for dynamically mapping unrecognised model label strings
#: to metric keys at load time.  Each entry is ``(keywords, metric_key)``
#: where *keywords* is a tuple of substrings that must all appear in the
#: lowercased label string.  The first matching pattern wins.
_LABEL_KEYWORD_PATTERNS: list[tuple[tuple[str, ...], str]] = [
    (("anime",), "style_anime"),
    (("manga",), "style_illustration"),
    (("3d",), "style_3d"),
    (("illust",), "style_illustration"),
    (("illustration",), "style_illustration"),
    (("drawing",), "style_illustration"),
    (("real",), "style_photo"),
    (("photo",), "style_photo"),
    (("photograph",), "style_photo"),
    (("cgi",), "style_cgi"),
    (("computer",), "style_cgi"),
]

#: Safe default value returned for any metric when scoring fails.
_DEFAULT_SCORE = 0.0


def _empty_result() -> dict[str, float]:
    """Return a zeroed-out result dict for all known metrics."""
    return {metric_key: _DEFAULT_SCORE for metric_key in set(_LABEL_TO_METRIC.values())}


# ---------------------------------------------------------------------------
# Scorer class
# ---------------------------------------------------------------------------


class CafeStyleScorer(BaseScorer):
    """Image style classifier using the CafeAI ViT style model.

    Produces five probability metrics — one per style category — that sum to
    approximately 1.0.  The ``style_photo`` metric is the most useful for
    Google Photos libraries: values close to 1.0 indicate a real photograph,
    while low values indicate artwork, screenshots, or synthetic imagery.

    The classifier is a fine-tuned ViT hosted at ``cafeai/cafe_style`` on
    HuggingFace Hub.
    """

    def __init__(self, variant_id: str | None = None, **kwargs: Any) -> None:
        super().__init__(variant_id=variant_id, **kwargs)
        # Lazy-loaded state — populated by _ensure_loaded()
        self._pipeline: Any = None
        # Label→metric mapping resolved from the actual model config at load
        # time; falls back to _LABEL_TO_METRIC for unit tests that skip loading.
        self._label_to_metric: dict[str, str] = _LABEL_TO_METRIC

    # ------------------------------------------------------------------
    # Class-level API
    # ------------------------------------------------------------------

    @classmethod
    def spec(cls) -> ScorerSpec:
        return ScorerSpec(
            scorer_id="cafe_style",
            display_name="Style Classifier",
            description=(
                "Classifies each image into one of five style categories: real photo, "
                "anime, illustration, 3D render, or CGI. Each category gets a probability "
                "score from 0 to 1. Use the 'Photo' score to filter out non-photographic "
                "content (screenshots, artwork, memes) from your library. "
                "Requires ~370 MB model download."
            ),
            technical_description=(
                "Applies the ``cafeai/cafe_style`` Vision Transformer classifier to "
                "predict the probability distribution over five mutually-exclusive style "
                "classes: photo, anime, illustration, 3d, CGI. "
                "Outputs five metrics (``style_photo``, ``style_anime``, "
                "``style_illustration``, ``style_3d``, ``style_cgi``) that sum to ≈ 1.0."
            ),
            version="1",
            metrics=(
                MetricSpec(
                    key="style_photo",
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
                    key="style_anime",
                    display_name="Anime",
                    description=("Probability that the image is in anime or manga style (0–1)."),
                    min_value=0.0,
                    max_value=1.0,
                    higher_is_better=True,
                ),
                MetricSpec(
                    key="style_illustration",
                    display_name="Illustration",
                    description=(
                        "Probability that the image is a digital or hand-drawn illustration (0–1)."
                    ),
                    min_value=0.0,
                    max_value=1.0,
                    higher_is_better=True,
                ),
                MetricSpec(
                    key="style_3d",
                    display_name="3D Render",
                    description=("Probability that the image is a 3D rendered image (0–1)."),
                    min_value=0.0,
                    max_value=1.0,
                    higher_is_better=True,
                ),
                MetricSpec(
                    key="style_cgi",
                    display_name="CGI",
                    description=(
                        "Probability that the image is computer-generated imagery (CGI, 0–1)."
                    ),
                    min_value=0.0,
                    max_value=1.0,
                    higher_is_better=True,
                ),
            ),
            variants=(
                VariantSpec(
                    variant_id="cafeai_v1",
                    display_name="CafeAI v1",
                    description=(
                        "ViT-based style classifier (cafeai/cafe_style on HuggingFace). "
                        "Trained to distinguish photo, anime, illustration, 3D, and CGI styles."
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
        present.  Also resolves the actual label strings used by the model
        into the :data:`_LABEL_TO_METRIC` mapping so that unknown alternate
        label names (e.g. ``"real"`` for *photo*, ``"cg"`` for *cgi*,
        ``"illust"`` for *illustration*) are handled automatically.
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

        # Build an augmented label mapping from the model's actual id2label so
        # that any non-canonical label strings (like "real", "cg", "illust")
        # are correctly routed to the right metric key.
        self._label_to_metric = dict(_LABEL_TO_METRIC)
        try:
            id2label: dict = self._pipeline.model.config.id2label
            for label in id2label.values():
                label_lower = str(label).lower()
                if label_lower in self._label_to_metric:
                    continue  # already known
                for keywords, metric_key in _LABEL_KEYWORD_PATTERNS:
                    if all(kw in label_lower for kw in keywords):
                        self._label_to_metric[label_lower] = metric_key
                        break
        except AttributeError:
            pass  # pipeline without accessible model config — use static map

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
            ``"style_photo"``, ``"style_anime"``, ``"style_illustration"``,
            ``"style_3d"``, ``"style_cgi"`` — each a float in ``[0.0, 1.0]``.
        """
        if not image_paths:
            return []

        from PIL import Image  # noqa: PLC0415

        self._ensure_loaded()

        # Pre-load all images; track which indices failed.
        imgs: list[Any] = []
        failed: set[int] = set()
        for i, path in enumerate(image_paths):
            try:
                imgs.append(Image.open(path).convert("RGB"))
            except (OSError, ValueError, RuntimeError):
                failed.add(i)
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
                    results[idx] = _preds_to_scores(all_preds[j], self._label_to_metric)
            except RuntimeError:  # noqa: BLE001
                # Fallback: score each image individually if bulk call fails.
                for _j, idx in enumerate(valid_indices):
                    try:
                        preds = self._pipeline(imgs[idx])
                        results[idx] = _preds_to_scores(preds, self._label_to_metric)
                    except (OSError, ValueError, RuntimeError):
                        pass

        return results


def _preds_to_scores(
    preds: list[dict[str, Any]],
    label_to_metric: dict[str, str] | None = None,
) -> dict[str, float]:
    """Convert pipeline prediction list to a metrics dict.

    Args:
        preds: List of ``{"label": str, "score": float}`` dicts returned by
            the image-classification pipeline.
        label_to_metric: Mapping from lowercased label strings to metric keys.
            Defaults to :data:`_LABEL_TO_METRIC` when ``None``.

    Returns:
        Dict mapping each of the five metric keys to a probability in [0, 1].
        Unknown labels are silently ignored; missing labels default to 0.0.
    """
    mapping = label_to_metric if label_to_metric is not None else _LABEL_TO_METRIC
    scores = _empty_result()
    for pred in preds:
        label = pred.get("label", "").lower()
        metric_key = mapping.get(label)
        if metric_key is not None:
            scores[metric_key] = float(pred["score"])
    return scores
