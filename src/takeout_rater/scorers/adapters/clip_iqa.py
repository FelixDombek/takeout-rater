"""CLIP-IQA scorer: no-reference image quality using CLIP text-image similarity.

CLIP-IQA measures perceptual image quality by comparing image embeddings against
paired text prompts ("a high quality photo" vs "a low quality photo") using a
CLIP model.  The score is the softmax probability that the image matches the
high-quality prompt, mapped to 0–100.  **Higher is better.**

The scorer leverages the ``open_clip`` library which is already a core
dependency of takeout-rater (used by the Aesthetic scorer).

Two model variants are provided:

* ``vitb32`` — ViT-B/32 (~350 MB download).  Lighter and faster.  Good default
  for machines with limited VRAM.
* ``vitl14`` — ViT-L/14 (~900 MB download).  Same backbone used by the Aesthetic
  scorer; if it is already cached the download is skipped.

References
----------
Wang, J. et al., "Exploring CLIP for Assessing the Look and Feel of Images,"
AAAI 2023. arXiv:2207.12396.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from takeout_rater.scorers.base import BaseScorer, MetricSpec, ScorerSpec, VariantSpec

if TYPE_CHECKING:
    import torch

# ---------------------------------------------------------------------------
# Variant → (model_name, pretrained) mapping
# ---------------------------------------------------------------------------

_VARIANTS: dict[str, tuple[str, str]] = {
    "vitb32": ("ViT-B-32", "openai"),
    "vitl14": ("ViT-L-14", "openai"),
}
_DEFAULT_VARIANT = "vitb32"

# Text prompts used to anchor the quality axis.
_PROMPT_HIGH = "a high quality photo"
_PROMPT_LOW = "a low quality photo"

# How many images to process per forward pass through the CLIP vision encoder.
_SCORE_BATCH_SIZE = 64


class CLIPIQAScorer(BaseScorer):
    """No-reference quality scorer using CLIP text-image similarity (CLIP-IQA)."""

    def __init__(self, variant_id: str | None = None, **kwargs: Any) -> None:
        super().__init__(variant_id=variant_id, **kwargs)
        # Lazy-loaded state
        self._clip_model: Any = None
        self._preprocess: Any = None
        self._text_features: Any = None  # torch.Tensor, shape (2, dim)
        self._device: Any = None

    # ------------------------------------------------------------------
    # Class-level API
    # ------------------------------------------------------------------

    @classmethod
    def spec(cls) -> ScorerSpec:
        return ScorerSpec(
            scorer_id="clip_iqa",
            display_name="CLIP-IQA",
            description=(
                "No-reference image quality assessment using CLIP text-image similarity.  "
                "Images are scored by comparing their CLIP embedding against the prompts "
                "'a high quality photo' and 'a low quality photo'.  "
                "Scores range from 0 (poor) to 100 (excellent).  Higher is better."
            ),
            metrics=(
                MetricSpec(
                    key="clip_iqa",
                    display_name="CLIP-IQA",
                    description=(
                        "Softmax probability of matching 'a high quality photo', "
                        "normalised to 0–100.  Higher values indicate better "
                        "perceived image quality."
                    ),
                    min_value=0.0,
                    max_value=100.0,
                    higher_is_better=True,
                ),
            ),
            variants=(
                VariantSpec(
                    variant_id="vitb32",
                    display_name="ViT-B/32 (lightweight)",
                    description=(
                        "CLIP ViT-B/32 backbone (~350 MB download).  "
                        "Faster and more memory-efficient."
                    ),
                ),
                VariantSpec(
                    variant_id="vitl14",
                    display_name="ViT-L/14 (accurate)",
                    description=(
                        "CLIP ViT-L/14 backbone (~900 MB download).  "
                        "Same model used by the Aesthetic scorer; cached if already downloaded."
                    ),
                ),
            ),
            default_variant_id=_DEFAULT_VARIANT,
            requires_extras=(),
        )

    @classmethod
    def is_available(cls) -> bool:
        try:
            import open_clip  # noqa: F401
            import PIL  # noqa: F401
            import torch  # noqa: F401

            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load the CLIP model and pre-compute text features if not already done."""
        if self._clip_model is not None:
            return

        import open_clip  # noqa: PLC0415
        import torch  # noqa: PLC0415

        vid = self.variant_id or _DEFAULT_VARIANT
        model_name, pretrained = _VARIANTS.get(vid, _VARIANTS[_DEFAULT_VARIANT])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        model = model.to(device).eval()

        tokenizer = open_clip.get_tokenizer(model_name)
        text_tokens = tokenizer([_PROMPT_HIGH, _PROMPT_LOW]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self._clip_model = model
        self._preprocess = preprocess
        self._text_features = text_features
        self._device = device

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_batch(
        self,
        image_paths: list[Path],
        *,
        variant_id: str | None = None,
    ) -> list[dict[str, float]]:
        """Score a batch of images using CLIP-IQA.

        Args:
            image_paths: Paths to image files (can be thumbnails).
            variant_id: Ignored after instantiation; use
                :meth:`~takeout_rater.scorers.base.BaseScorer.create` with
                *variant_id* to select the model variant.

        Returns:
            List of dicts with key ``"clip_iqa"`` → float in ``[0, 100]``.
            If a file cannot be opened, ``clip_iqa`` is set to ``0.0``.
        """
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        self._ensure_loaded()
        assert self._clip_model is not None
        assert self._preprocess is not None
        assert self._text_features is not None
        assert self._device is not None

        scores: list[float] = []

        for batch_start in range(0, len(image_paths), _SCORE_BATCH_SIZE):
            batch_paths = image_paths[batch_start : batch_start + _SCORE_BATCH_SIZE]
            tensors: list[Any] = []
            valid_indices: list[int] = []

            for i, path in enumerate(batch_paths):
                try:
                    img = Image.open(path).convert("RGB")
                    tensors.append(self._preprocess(img))
                    valid_indices.append(i)
                except (OSError, ValueError):
                    pass

            # Prepare output for this sub-batch (default 0.0 for failed images)
            sub_scores = [0.0] * len(batch_paths)

            if tensors:
                img_batch: torch.Tensor = torch.stack(tensors).to(self._device)
                with torch.no_grad():
                    img_features = self._clip_model.encode_image(img_batch)
                    img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                    # logits shape: (N, 2) — columns are [high_quality, low_quality]
                    logits = img_features @ self._text_features.T
                    probs = logits.softmax(dim=-1)
                    high_quality_probs = probs[:, 0].tolist()

                for j, idx in enumerate(valid_indices):
                    sub_scores[idx] = float(high_quality_probs[j]) * 100.0

            scores.extend(sub_scores)

        return [{"clip_iqa": s} for s in scores]
