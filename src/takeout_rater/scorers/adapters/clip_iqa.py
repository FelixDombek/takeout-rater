"""CLIP-IQA scorer: zero-shot perceptual quality via CLIP text–image similarity.

CLIP-IQA (Wang et al., 2022) scores image quality by measuring the cosine
similarity between an image embedding and two antonym text prompts —
``"Good photo."`` and ``"Bad photo."`` — using a pre-trained CLIP backbone.
The softmax probability assigned to the positive prompt is the quality score.

Because ``open-clip-torch`` is already a core dependency of this project (used
by the aesthetic scorer), this scorer incurs **no additional package download**.
The CLIP ViT-L/14 backbone (~900 MB) is shared with the aesthetic scorer and
cached by ``open_clip`` in the standard Torch hub cache (``~/.cache/torch/hub``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from takeout_rater.scorers.base import BaseScorer, MetricSpec, ScorerSpec, VariantSpec

if TYPE_CHECKING:
    import torch

# ---------------------------------------------------------------------------
# Text prompts used as quality anchors
# ---------------------------------------------------------------------------

#: Positive-quality anchor — the probability assigned to this prompt is the score.
_PROMPT_GOOD = "Good photo."
#: Negative-quality anchor — paired with the positive prompt in a softmax.
_PROMPT_BAD = "Bad photo."

# ---------------------------------------------------------------------------
# CLIP backbone identifier (shared with the aesthetic scorer)
# ---------------------------------------------------------------------------

_CLIP_MODEL_NAME = "ViT-L-14"
_CLIP_PRETRAINED = "openai"


class CLIPIQAScorer(BaseScorer):
    """Zero-shot perceptual quality scorer using CLIP text–image similarity.

    Computes the cosine similarity between a CLIP image embedding and two text
    prompts ("Good photo." / "Bad photo."), then applies a softmax to obtain
    a quality probability in [0, 1].

    No separate model download is required beyond the CLIP ViT-L/14 backbone
    that is already used by the aesthetic scorer.
    """

    def __init__(self, variant_id: str | None = None, **kwargs: Any) -> None:
        super().__init__(variant_id=variant_id, **kwargs)
        # Lazy-loaded state — populated by _ensure_loaded()
        self._clip_model: Any = None
        self._preprocess: Any = None
        self._text_features: Any = None  # pre-encoded text prompts, shape (2, 768)
        self._device: Any = None

    # ------------------------------------------------------------------
    # Class-level API
    # ------------------------------------------------------------------

    @classmethod
    def spec(cls) -> ScorerSpec:
        return ScorerSpec(
            scorer_id="clip_iqa",
            display_name="CLIP-IQA Quality",
            description=(
                "Estimates perceptual image quality using zero-shot CLIP text–image "
                "similarity.  The score is the softmax probability that the image "
                "matches 'Good photo.' over 'Bad photo.' using CLIP ViT-L/14."
            ),
            metrics=(
                MetricSpec(
                    key="clip_quality",
                    display_name="CLIP Quality",
                    description=(
                        "Zero-shot perceptual quality score (0–1, higher is better). "
                        "Derived from CLIP ViT-L/14 text–image cosine similarity."
                    ),
                    min_value=0.0,
                    max_value=1.0,
                    higher_is_better=True,
                ),
            ),
            variants=(
                VariantSpec(
                    variant_id="vit_l14_openai",
                    display_name="CLIP ViT-L/14 (OpenAI)",
                    description=(
                        "CLIP ViT-L/14 pre-trained on OpenAI's WIT dataset. "
                        "Shared backbone with the aesthetic scorer."
                    ),
                ),
            ),
            default_variant_id="vit_l14_openai",
            requires_extras=(),
        )

    @classmethod
    def is_available(cls) -> bool:
        """Return ``True`` when CLIP dependencies are importable."""
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
        """Load the CLIP backbone and pre-encode the quality text prompts.

        The CLIP weights (~900 MB) are downloaded once to the standard Torch
        hub cache (``~/.cache/torch/hub``) and reused on subsequent runs.
        The text features for "Good photo." / "Bad photo." are encoded once
        here and cached on ``self._text_features`` for all subsequent calls.
        """
        if self._clip_model is not None:
            return

        import open_clip  # noqa: PLC0415
        import torch  # noqa: PLC0415

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            _CLIP_MODEL_NAME, pretrained=_CLIP_PRETRAINED
        )
        clip_model.eval()
        clip_model.to(device)

        tokenizer = open_clip.get_tokenizer(_CLIP_MODEL_NAME)
        tokens = tokenizer([_PROMPT_GOOD, _PROMPT_BAD]).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self._device = device
        self._clip_model = clip_model
        self._preprocess = preprocess
        self._text_features = text_features  # shape (2, D), float

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_batch(
        self,
        image_paths: list[Path],
        *,
        variant_id: str | None = None,
    ) -> list[dict[str, float]]:
        """Score a batch of images for perceptual quality using CLIP-IQA.

        Each image is encoded by CLIP ViT-L/14 and compared against the
        pre-encoded text features for "Good photo." and "Bad photo." via
        cosine similarity.  A softmax over the two similarities yields a
        quality probability in [0, 1].

        Images are processed individually (not in one GPU batch) to keep
        memory use predictable.  For throughput-critical use, batching can
        be added in a future optimisation pass.

        If a file cannot be opened or the forward pass fails, the score
        defaults to ``0.0`` rather than raising.

        Args:
            image_paths: Absolute paths to image files.
            variant_id: Ignored; only one variant exists.

        Returns:
            List (same length as *image_paths*) of dicts with key
            ``"clip_quality"`` → float in ``[0.0, 1.0]``.
        """
        if not image_paths:
            return []

        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        self._ensure_loaded()

        results: list[dict[str, float]] = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                tensor: torch.Tensor = self._preprocess(img).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    img_features = self._clip_model.encode_image(tensor)
                    img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                    # Cosine similarities with each text prompt: shape (2,)
                    sims = (img_features @ self._text_features.T).squeeze(0)
                    # Softmax → probability that the image is "good"
                    probs = torch.softmax(sims, dim=0)
                    quality = float(probs[0].item())
            except (OSError, ValueError, RuntimeError):
                quality = 0.0
            results.append({"clip_quality": max(0.0, min(1.0, quality))})
        return results
