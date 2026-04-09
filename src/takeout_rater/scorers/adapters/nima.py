"""NIMA scorer: Neural Image Assessment (Google, 2018).

NIMA (Talebi & Milanfar, 2018, IEEE T-IP) trains a CNN to predict the
*distribution* of human aesthetic ratings over the 10-point AVA scale.  The
mean of that distribution gives an overall quality or aesthetic score in [1, 10].

This implementation uses two pre-trained MobileNet-V2 checkpoints hosted on
HuggingFace:

- ``aesthetic`` variant — trained on the AVA aesthetic dataset.
- ``technical`` variant — trained on the TID2013 / KADID-10k technical quality
  datasets; more sensitive to blur, noise, and compression artefacts.

Both checkpoints replace the standard MobileNet-V2 classifier head with a
10-neuron softmax head followed by an Earth Mover's Distance (EMD) loss during
training.  At inference we compute the expected value of the rating distribution
(``sum(rating * prob)`` for rating in 1..10).

The MobileNet-V2 backbone and pre-trained NIMA weights are downloaded from
HuggingFace Hub on first use and cached in ``~/.cache/huggingface``.
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

#: HuggingFace repo hosting the NIMA MobileNet-V2 checkpoints.
_HF_REPO = "truskovskiyk/nima-pytorch"

#: Filename for each variant's checkpoint inside the HuggingFace repo.
_VARIANT_FILENAMES: dict[str, str] = {
    "aesthetic": "aesthetic_mobilenetv2.pth",
    "technical": "technical_mobilenetv2.pth",
}

#: Number of rating bins (1–10 scale used in the AVA / TID datasets).
_NUM_CLASSES = 10

#: Rating values [1, 2, …, 10] used to compute the expected score.
_RATING_VALUES = list(range(1, _NUM_CLASSES + 1))

#: Input size expected by the MobileNet-V2 backbone.
_INPUT_SIZE = 224

#: Number of images to forward through MobileNet-V2 in a single GPU pass.
#: MobileNet-V2 is lightweight; 64 images fit easily on an 8 GB GPU.
_SCORE_BATCH_SIZE = 64

# ---------------------------------------------------------------------------
# MobileNet-V2 NIMA head
# ---------------------------------------------------------------------------


def _build_nima_model() -> Any:
    """Build a MobileNet-V2 with a 10-class softmax head for NIMA.

    The standard MobileNet-V2 classifier (1000-class) is replaced with a
    10-neuron linear layer matching the NIMA checkpoint convention.

    Returns:
        An uninitialised ``torch.nn.Module`` ready to load NIMA weights.
    """
    import torch.nn as nn  # noqa: PLC0415
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2  # noqa: PLC0415

    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    # Replace the classification head with a 10-neuron softmax head.
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, _NUM_CLASSES),
        nn.Softmax(dim=1),
    )
    return model


# ---------------------------------------------------------------------------
# Scorer class
# ---------------------------------------------------------------------------


class NIMAScorer(BaseScorer):
    """NIMA (Neural Image Assessment) scorer — aesthetic and technical variants.

    Predicts human aesthetic / technical quality ratings on a 1–10 scale using
    a MobileNet-V2 backbone fine-tuned with Earth Mover's Distance loss on AVA
    (aesthetic) and TID2013 / KADID-10k (technical) datasets.

    Two variants are available:

    - ``aesthetic``: Predicts perceived aesthetic quality (composition, colour,
      subject matter).  Trained on the AVA dataset.
    - ``technical``: Predicts technical quality (sharpness, noise, JPEG
      compression artefacts).

    Both variants output a single float in [1, 10] stored under the metric
    key ``nima_score``.  A separate scorer run is required for each variant.
    """

    def __init__(self, variant_id: str | None = None, **kwargs: Any) -> None:
        super().__init__(variant_id=variant_id, **kwargs)
        # Lazy-loaded state — populated by _ensure_loaded()
        self._model: Any = None
        self._preprocess: Any = None
        self._device: Any = None

    # ------------------------------------------------------------------
    # Class-level API
    # ------------------------------------------------------------------

    @classmethod
    def spec(cls) -> ScorerSpec:
        return ScorerSpec(
            scorer_id="nima",
            display_name="NIMA Quality",
            description=(
                "Neural Image Assessment (NIMA) predicts how human viewers would rate a photo "
                "on a 1–10 scale, using a compact neural network (MobileNet-V2) trained on "
                "professional photo ratings. Two variants: aesthetic quality (composition, "
                "lighting, colour) and technical quality (sharpness, noise, exposure). "
                "Requires ~20 MB model download per variant."
            ),
            version="1",
            metrics=(
                MetricSpec(
                    key="nima_score",
                    display_name="NIMA Score",
                    description=(
                        "Mean predicted human rating (1–10, higher is better). "
                        "Represents the expected value of the rating distribution."
                    ),
                    min_value=1.0,
                    max_value=10.0,
                    higher_is_better=True,
                ),
            ),
            variants=(
                VariantSpec(
                    variant_id="aesthetic",
                    display_name="Aesthetic (AVA)",
                    description=(
                        "NIMA trained on the AVA dataset.  Predicts perceived aesthetic "
                        "quality: composition, colour harmony, and subject interest."
                    ),
                ),
                VariantSpec(
                    variant_id="technical",
                    display_name="Technical (TID2013/KADID)",
                    description=(
                        "NIMA trained on TID2013 / KADID-10k.  Predicts technical "
                        "quality: sharpness, noise level, and compression artefacts."
                    ),
                ),
            ),
            default_variant_id="aesthetic",
            requires_extras=(),
        )

    @classmethod
    def is_available(cls) -> bool:
        """Return ``True`` when required dependencies are importable."""
        try:
            import huggingface_hub  # noqa: F401
            import PIL  # noqa: F401
            import torch  # noqa: F401
            import torchvision  # noqa: F401

            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    @classmethod
    def _download_weights(cls, variant_id: str) -> Path:
        """Download the NIMA checkpoint for *variant_id* from HuggingFace Hub.

        Args:
            variant_id: Either ``"aesthetic"`` or ``"technical"``.

        Returns:
            Local path to the downloaded ``.pth`` file.

        Raises:
            ValueError: If *variant_id* is not recognised.
            Exception: Propagated from ``hf_hub_download`` on network errors.
        """
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        filename = _VARIANT_FILENAMES.get(variant_id)
        if filename is None:
            raise ValueError(
                f"Unknown NIMA variant '{variant_id}'. Expected one of: {list(_VARIANT_FILENAMES)}"
            )
        return Path(hf_hub_download(repo_id=_HF_REPO, filename=filename))

    def _ensure_loaded(self) -> None:
        """Load the NIMA model for the active variant on first call (lazy init).

        Downloads the MobileNet-V2 backbone (torchvision) and the NIMA
        checkpoint from HuggingFace Hub on first use.
        """
        if self._model is not None:
            return

        import torch  # noqa: PLC0415
        import torchvision.transforms as T  # noqa: PLC0415

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        weights_path = self._download_weights(self.variant_id)
        model = _build_nima_model()
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)

        preprocess = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(_INPUT_SIZE),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self._device = device
        self._model = model
        self._preprocess = preprocess

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_batch(
        self,
        image_paths: list[Path],
        *,
        variant_id: str | None = None,
    ) -> list[dict[str, float]]:
        """Score a batch of images using NIMA.

        Images are processed in chunks of :data:`_SCORE_BATCH_SIZE` so that
        each chunk is forwarded through the MobileNet-V2 backbone in a single
        GPU pass.  For each image the model outputs a 10-bin probability
        distribution over rating values 1–10; the expected rating (mean) of
        that distribution is the ``nima_score``.

        Within each chunk, images that fail to open are skipped and receive the
        minimum score (``1.0``).  If the whole-chunk forward pass raises
        ``RuntimeError`` (e.g. VRAM OOM), the chunk falls back to per-image
        scoring.

        Args:
            image_paths: Absolute paths to image files.
            variant_id: Ignored at runtime; the active variant is set at
                construction time via :meth:`create`.

        Returns:
            List (same length as *image_paths*) of dicts with key
            ``"nima_score"`` → float in ``[1.0, 10.0]``.
        """
        if not image_paths:
            return []

        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        self._ensure_loaded()

        rating_tensor = torch.tensor(_RATING_VALUES, dtype=torch.float32, device=self._device)
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

            sub_scores = [1.0] * len(chunk)  # default: minimum score

            if tensors:
                try:
                    batch = torch.stack(tensors).to(self._device)
                    with torch.no_grad():
                        probs = self._model(batch)  # (N, 10)
                        # Expected rating: sum(rating * prob) per image
                        raw_scores = (probs * rating_tensor).sum(dim=-1)  # (N,)
                        score_values: list[float] = raw_scores.tolist()
                    for j, idx in enumerate(valid_indices):
                        sub_scores[idx] = max(1.0, min(10.0, float(score_values[j])))
                except RuntimeError:
                    # Fallback: score each valid image individually (e.g. VRAM OOM)
                    for _j, idx in enumerate(valid_indices):
                        path = chunk[idx]
                        try:
                            img = Image.open(path).convert("RGB")
                            tensor = self._preprocess(img).unsqueeze(0).to(self._device)
                            with torch.no_grad():
                                probs_single = self._model(tensor).squeeze(0)  # (10,)
                                score_single = float((probs_single * rating_tensor).sum().item())
                            sub_scores[idx] = max(1.0, min(10.0, score_single))
                        except (OSError, ValueError, RuntimeError):
                            pass

            scores.extend(sub_scores)

        return [{"nima_score": s} for s in scores]
