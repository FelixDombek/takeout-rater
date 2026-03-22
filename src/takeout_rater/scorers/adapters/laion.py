"""LAION Aesthetic Predictor v2 scorer.

This scorer uses the LAION aesthetic predictor MLP on top of OpenAI CLIP
ViT-L/14 image embeddings to predict a visual-quality score on a 0–10 scale.

Higher scores indicate images that a crowd of annotators would rate as more
aesthetically pleasing.  The model was trained on a combination of LAION,
SAC, and AVA datasets.

Requirements
------------
Install the ``aesthetic`` extra::

    pip install takeout-rater[aesthetic]

This brings in ``torch``, ``torchvision``, ``Pillow``, ``huggingface-hub``,
and ``open-clip-torch``.

The CLIP backbone (~900 MB) is downloaded from OpenAI on first use and cached
by ``open_clip`` in the standard Torch hub cache (``~/.cache/torch/hub``).
The MLP weights (~4 MB) are downloaded from the HuggingFace Hub and cached by
``huggingface_hub`` (``~/.cache/huggingface``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from takeout_rater.scorers.base import BaseScorer, MetricSpec, ScorerSpec, VariantSpec

if TYPE_CHECKING:
    import torch

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------

#: HuggingFace repo containing the MLP weights.
_HF_REPO = "LAION-AI/aesthetic-predictor"
#: Filename of the MLP checkpoint inside the repo (trained with MSE loss on
#: CLIP ViT-L/14 embeddings).
_HF_FILENAME = "sac+logos+ava1-l14-linearMSE.pth"

#: CLIP model architecture used to extract image embeddings.
_CLIP_MODEL_NAME = "ViT-L-14"
#: Which pre-trained weights to use for the CLIP backbone.
_CLIP_PRETRAINED = "openai"
#: Embedding dimension produced by ViT-L/14.
_EMBEDDING_DIM = 768


# ---------------------------------------------------------------------------
# MLP architecture (must match the saved checkpoint)
# ---------------------------------------------------------------------------


def _build_mlp(input_dim: int) -> torch.nn.Module:
    """Return the MLP architecture used by the LAION aesthetic predictor.

    The architecture is a 5-layer MLP with dropout, matching the checkpoint
    saved at ``LAION-AI/aesthetic-predictor`` on HuggingFace.

    Args:
        input_dim: Size of the input embedding (768 for ViT-L/14).

    Returns:
        An uninitialised ``torch.nn.Sequential`` module.
    """
    import torch.nn as nn  # noqa: PLC0415

    return nn.Sequential(
        nn.Linear(input_dim, 1024),
        nn.Dropout(0.2),
        nn.Linear(1024, 128),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.Dropout(0.1),
        nn.Linear(64, 16),
        nn.Dropout(0.1),
        nn.Linear(16, 1),
    )


# ---------------------------------------------------------------------------
# Scorer class
# ---------------------------------------------------------------------------


class AestheticScorer(BaseScorer):
    """LAION Aesthetic Predictor v2 — predicts image aesthetic quality (0–10).

    The scorer is a two-stage pipeline:

    1. **CLIP ViT-L/14** extracts a 768-dimensional image embedding.
    2. A **5-layer MLP** maps that embedding to a single float in [0, 10].

    Both components are loaded lazily on the first call to
    :meth:`score_batch` so that startup cost is zero when the scorer is
    not used.

    This scorer is only available when the ``aesthetic`` extra is installed::

        pip install takeout-rater[aesthetic]
    """

    def __init__(self, variant_id: str | None = None, **kwargs: Any) -> None:
        super().__init__(variant_id=variant_id, **kwargs)
        # Lazy-loaded state — populated by _ensure_loaded()
        self._clip_model: Any = None
        self._preprocess: Any = None
        self._mlp: Any = None
        self._device: Any = None

    # ------------------------------------------------------------------
    # Class-level API
    # ------------------------------------------------------------------

    @classmethod
    def spec(cls) -> ScorerSpec:
        return ScorerSpec(
            scorer_id="aesthetic",
            display_name="Aesthetic Score",
            description=(
                "Predicts the aesthetic quality of an image using the LAION Aesthetic "
                "Predictor v2. The model uses CLIP ViT-L/14 embeddings and a small MLP "
                "trained on LAION, SAC, and AVA datasets. Scores range from 0 (poor) "
                "to 10 (excellent)."
            ),
            metrics=(
                MetricSpec(
                    key="aesthetic",
                    display_name="Aesthetic",
                    description=(
                        "Predicted aesthetic quality score (0–10, higher is better). "
                        "Scores above 6 are generally considered visually pleasing."
                    ),
                    min_value=0.0,
                    max_value=10.0,
                    higher_is_better=True,
                ),
            ),
            variants=(
                VariantSpec(
                    variant_id="laion_v2",
                    display_name="LAION Aesthetic v2",
                    description=(
                        "MLP trained on LAION, SAC, and AVA datasets using CLIP "
                        "ViT-L/14 embeddings (sac+logos+ava1-l14-linearMSE)."
                    ),
                ),
            ),
            default_variant_id="laion_v2",
            requires_extras=("aesthetic",),
        )

    @classmethod
    def is_available(cls) -> bool:
        """Return ``True`` when ``torch`` and ``open_clip`` are importable."""
        try:
            import open_clip  # noqa: F401
            import torch  # noqa: F401

            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load the CLIP backbone and aesthetic MLP on first call (lazy init).

        Downloads model weights to the standard cache directories if not
        already present:
        - CLIP weights → ``~/.cache/torch/hub`` (managed by ``open_clip``)
        - MLP weights  → ``~/.cache/huggingface`` (managed by ``huggingface_hub``)
        """
        if self._clip_model is not None:
            return  # already loaded

        import open_clip  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load CLIP ViT-L/14 backbone
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            _CLIP_MODEL_NAME, pretrained=_CLIP_PRETRAINED
        )
        clip_model.eval()
        clip_model.to(self._device)
        self._clip_model = clip_model
        self._preprocess = preprocess

        # Download and load the aesthetic MLP weights
        weights_path = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILENAME)
        mlp = _build_mlp(_EMBEDDING_DIM)
        state_dict = torch.load(weights_path, map_location=self._device, weights_only=True)
        mlp.load_state_dict(state_dict)
        mlp.eval()
        mlp.to(self._device)
        self._mlp = mlp

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_batch(
        self,
        image_paths: list[Path],
        *,
        variant_id: str | None = None,
    ) -> list[dict[str, float]]:
        """Score a batch of images for aesthetic quality.

        Images are processed one at a time (each PIL open + CLIP forward pass).
        If a file cannot be opened or processed, the score defaults to ``0.0``.

        Args:
            image_paths: Paths to image files.  Thumbnails (512 px) work well
                and are much faster to load than full-resolution originals.
            variant_id: Ignored; only one variant exists.

        Returns:
            A list (same length as *image_paths*) of dicts with a single key
            ``"aesthetic"`` → float in ``[0.0, 10.0]``.
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
                tensor = self._preprocess(img).unsqueeze(0).to(self._device)

                with torch.no_grad():
                    embedding = self._clip_model.encode_image(tensor)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                    raw: float = self._mlp(embedding.float()).item()

                score = max(0.0, min(10.0, float(raw)))
            except (OSError, ValueError, RuntimeError):
                score = 0.0

            results.append({"aesthetic": score})

        return results
