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

from takeout_rater.scoring.scorers.base import (
    BaseScorer,
    MetricSpec,
    ScorerSpec,
    VariantSpec,
    _run_pipelined_batches,
)

if TYPE_CHECKING:
    pass

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

#: Number of preprocessed batches to keep ready ahead of GPU inference.
_PREFETCH_BATCHES = 2


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
        self._logit_scale: Any = None  # scalar temperature for similarity scaling
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
                "Uses a language-vision model (CLIP) to estimate photo quality without any "
                "task-specific training. It compares the image against the text descriptions "
                "'Good photo.' and 'Bad photo.' and reports how strongly the image resembles "
                "the good one. Requires the aesthetic scorer's CLIP model (~900 MB)."
            ),
            technical_description=(
                "Estimates perceptual image quality using zero-shot CLIP text–image "
                "similarity. The score is the softmax probability that the image "
                "matches 'Good photo.' over 'Bad photo.' using CLIP ViT-L/14."
            ),
            version="2",
            variants=(
                VariantSpec(
                    variant_id="vit_l14_openai",
                    display_name="CLIP ViT-L/14 (OpenAI)",
                    description=(
                        "CLIP ViT-L/14 pre-trained on OpenAI's WIT dataset. "
                        "Shared backbone with the aesthetic scorer."
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

        import torch  # noqa: PLC0415

        from takeout_rater.scoring.scorers.clip_backbone import get_clip_model  # noqa: PLC0415

        clip_model, preprocess, tokenizer, device = get_clip_model()

        tokens = tokenizer([_PROMPT_GOOD, _PROMPT_BAD]).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self._device = device
        self._clip_model = clip_model
        self._preprocess = preprocess
        self._text_features = text_features  # shape (2, D), float
        # CLIP's learned temperature: multiplying similarities by logit_scale
        # before softmax separates "Good photo" vs "Bad photo" probabilities
        # into a meaningful range instead of collapsing near 0.5.
        self._logit_scale = clip_model.logit_scale.exp().detach()

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

        CPU preprocessing (PIL decode + CLIP transforms) and GPU inference are
        **pipelined**: while the GPU processes chunk N, a background thread
        preprocesses chunk N+1 so that neither device sits idle waiting for
        the other.

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
                    tensors.append(self._preprocess(img))
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
                    batch = torch.stack(valid_tensors).to(self._device)
                    with torch.no_grad():
                        img_features = self._clip_model.encode_image(batch)  # (N, D)
                        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                        # sims: (N, 2) — column 0 = "Good photo.", column 1 = "Bad photo."
                        # Apply logit_scale (CLIP's learned temperature) so that small
                        # cosine-similarity differences between the two prompts are
                        # amplified into a meaningful probability spread.
                        sims = self._logit_scale * (img_features @ self._text_features.T)
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
                                sim = self._logit_scale * (
                                    img_feat @ self._text_features.T
                                )  # (1, 2)
                                prob = torch.softmax(sim, dim=-1)
                                sub_scores[idx] = max(0.0, min(1.0, float(prob[0, 0].item())))
                        except (OSError, ValueError, RuntimeError):
                            pass

            return [{"clip_quality": s} for s in sub_scores]

        return _run_pipelined_batches(chunks, _preprocess, _infer, prefetch=_PREFETCH_BATCHES)
