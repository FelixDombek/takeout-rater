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

#: Number of images to encode in a single GPU forward pass.
#: 64 fits comfortably within the VRAM budget of an 8 GB GPU alongside
#: the ViT-L/14 weights (~900 MB).  Tune down if you hit VRAM OOM errors.
_SCORE_BATCH_SIZE = 64


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

        Images are processed in chunks of :data:`_SCORE_BATCH_SIZE` so that
        each chunk is forwarded through CLIP ViT-L/14 in a single GPU pass,
        making full use of hardware parallelism.

        Within each chunk, images that fail to open are skipped and receive a
        score of ``0.0``.  If the whole-chunk forward pass raises
        ``RuntimeError`` (e.g. VRAM OOM), the chunk falls back to per-image
        scoring so that a single failure never silently zeros an entire batch.

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

        scores: list[float] = []

        for batch_start in range(0, len(image_paths), _SCORE_BATCH_SIZE):
            chunk = image_paths[batch_start : batch_start + _SCORE_BATCH_SIZE]
            tensors: list[torch.Tensor] = []
            valid_indices: list[int] = []

            for i, path in enumerate(chunk):
                try:
                    img = Image.open(path).convert("RGB")
                    tensors.append(self._preprocess(img))
                    valid_indices.append(i)
                except (OSError, ValueError):
                    pass

            sub_scores = [0.0] * len(chunk)

            if tensors:
                try:
                    batch = torch.stack(tensors).to(self._device)
                    with torch.no_grad():
                        img_features = self._clip_model.encode_image(batch)  # (N, D)
                        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                        # sims: (N, 2) — column 0 = "Good photo.", column 1 = "Bad photo."
                        sims = img_features @ self._text_features.T
                        probs = torch.softmax(sims, dim=-1)  # (N, 2)
                        quality_values: list[float] = probs[:, 0].tolist()
                    for j, idx in enumerate(valid_indices):
                        sub_scores[idx] = max(0.0, min(1.0, quality_values[j]))
                except RuntimeError:
                    # Fallback: score each valid image individually (e.g. VRAM OOM)
                    for _j, idx in enumerate(valid_indices):
                        path = chunk[idx]
                        try:
                            img = Image.open(path).convert("RGB")
                            tensor = self._preprocess(img).unsqueeze(0).to(self._device)
                            with torch.no_grad():
                                img_feat = self._clip_model.encode_image(tensor)
                                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                                sim = img_feat @ self._text_features.T  # (1, 2)
                                prob = torch.softmax(sim, dim=-1)
                                sub_scores[idx] = max(0.0, min(1.0, float(prob[0, 0].item())))
                        except (OSError, ValueError, RuntimeError):
                            pass

            scores.extend(sub_scores)

        return [{"clip_quality": s} for s in scores]
