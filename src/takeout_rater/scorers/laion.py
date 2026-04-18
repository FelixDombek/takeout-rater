"""LAION Aesthetic Predictor v2 scorer.

This scorer uses the LAION aesthetic predictor MLP on top of OpenAI CLIP
ViT-L/14 image embeddings to predict a visual-quality score on a 0–10 scale.

Higher scores indicate images that a crowd of annotators would rate as more
aesthetically pleasing.  The model was trained on a combination of LAION,
SAC, and AVA datasets.

The CLIP backbone (~900 MB) is downloaded from OpenAI on first use and cached
by ``open_clip`` in the standard Torch hub cache (``~/.cache/torch/hub``).
The MLP weights (~4 MB) are downloaded from the HuggingFace Hub and cached by
``huggingface_hub`` (``~/.cache/huggingface``).
"""

from __future__ import annotations

import os
import queue
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from takeout_rater.scorers.base import BaseScorer, MetricSpec, ScorerSpec, VariantSpec

if TYPE_CHECKING:
    import torch

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------

#: HuggingFace repos containing the MLP weights (first accessible one wins).
#:
#: The original LAION repo was removed, so we try open mirrors first and fall
#: back to the historic location for anyone who still has access.
_HF_REPO_FALLBACKS = (
    "camenduru/improved-aesthetic-predictor",
    "ttj/sac-logos-ava1-l14-linearMSE",
    "LAION-AI/aesthetic-predictor",
)
#: Filename of the MLP checkpoint inside the repo (trained with MSE loss on
#: CLIP ViT-L/14 embeddings).
_HF_FILENAME = "sac+logos+ava1-l14-linearMSE.pth"

#: CLIP model architecture used to extract image embeddings.
_CLIP_MODEL_NAME = "ViT-L-14"
#: Which pre-trained weights to use for the CLIP backbone.
_CLIP_PRETRAINED = "openai"
#: Embedding dimension produced by ViT-L/14.
_EMBEDDING_DIM = 768
#: Number of preprocessed batches to keep ready ahead of GPU inference.
#: Higher values use more CPU memory but keep the GPU better fed.
_PREFETCH_BATCHES = 2
#: 64 is well within the VRAM budget of an 8 GB GPU (CLIP ViT-L/14 uses ~1.8 GB,
#: leaving ample headroom).  Tune down if you hit VRAM OOM errors.
_SCORE_BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# MLP architecture (must match the saved checkpoint)
# ---------------------------------------------------------------------------


def _build_mlp(input_dim: int) -> torch.nn.Module:
    """Return the MLP architecture used by the LAION aesthetic predictor.

    The architecture is a 5-layer MLP with dropout, matching the checkpoint
    published as ``sac+logos+ava1-l14-linearMSE.pth`` on HuggingFace mirrors.

    The checkpoint stores weights under a ``layers`` submodule (e.g.
    ``layers.0.weight``), so the returned module wraps the ``nn.Sequential``
    in a container class that exposes it as ``self.layers``.

    Args:
        input_dim: Size of the input embedding (768 for ViT-L/14).

    Returns:
        An uninitialised ``MLP`` module whose ``layers`` attribute holds the
        ``nn.Sequential`` stack.
    """
    import torch.nn as nn  # noqa: PLC0415

    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.Dropout(0.2),
                nn.Linear(1024, 128),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.Dropout(0.1),
                nn.Linear(64, 16),
                nn.Linear(16, 1),
            )

        def forward(self, x):  # type: ignore[override]
            return self.layers(x)

    return MLP()


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
                "Predicts how aesthetically pleasing a photo looks to human viewers, "
                "using a neural network trained on crowd-sourced ratings from the AVA photo "
                "competition. Higher scores indicate compositions, lighting, and colour "
                "balance that people tend to rate highly. Requires ~900 MB CLIP model download."
            ),
            technical_description=(
                "Predicts the aesthetic quality of an image using the LAION Aesthetic "
                "Predictor v2. The model uses CLIP ViT-L/14 embeddings and a small MLP "
                "trained on LAION, SAC, and AVA datasets. Scores range from 0 (poor) "
                "to 10 (excellent)."
            ),
            version="1",
            variants=(
                VariantSpec(
                    variant_id="laion_v2",
                    display_name="LAION Aesthetic v2",
                    description=(
                        "MLP trained on LAION, SAC, and AVA datasets using CLIP "
                        "ViT-L/14 embeddings (sac+logos+ava1-l14-linearMSE)."
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
                ),
            ),
            default_variant_id="laion_v2",
            requires_extras=(),
        )

    @classmethod
    def is_available(cls) -> bool:
        """Return ``True`` when required runtime dependencies are importable."""
        try:
            import huggingface_hub  # noqa: F401
            import open_clip  # noqa: F401
            import PIL  # noqa: F401
            import torch  # noqa: F401

            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    @staticmethod
    def _hf_repo_candidates() -> tuple[str, ...]:
        """Return ordered HuggingFace repo candidates for the aesthetic MLP.

        Users can override the first candidate via ``TAKEOUT_RATER_AESTHETIC_REPO``.
        """
        env_repo = os.getenv("TAKEOUT_RATER_AESTHETIC_REPO")
        repos = [env_repo] if env_repo else []
        repos.extend(_HF_REPO_FALLBACKS)
        # Preserve order while removing duplicates
        seen: set[str] = set()
        ordered: list[str] = []
        for repo in repos:
            if repo not in seen:
                seen.add(repo)
                ordered.append(repo)
        return tuple(ordered)

    @classmethod
    def _download_mlp_weights(cls) -> Path:
        """Download the aesthetic MLP weights from the first reachable repo."""
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        errors: list[Exception] = []
        for repo_id in cls._hf_repo_candidates():
            try:
                return Path(hf_hub_download(repo_id=repo_id, filename=_HF_FILENAME))
            except Exception as exc:  # pragma: no cover - repr is surfaced below
                errors.append(exc)
                continue

        if errors:
            # Raise the last error to preserve the most relevant traceback while
            # still surfacing that all candidates were attempted.
            raise errors[-1]

        raise RuntimeError("No HuggingFace repo candidates configured for aesthetic MLP download.")

    def _ensure_loaded(self) -> None:
        """Load the CLIP backbone and aesthetic MLP on first call (lazy init).

        Downloads model weights to the standard cache directories if not
        already present:
        - CLIP weights → ``~/.cache/torch/hub`` (managed by ``open_clip``)
        - MLP weights  → ``~/.cache/huggingface`` (managed by ``huggingface_hub``)
        """
        if (
            self._clip_model is not None
            and self._preprocess is not None
            and self._mlp is not None
            and self._device is not None
        ):
            return  # already loaded

        import torch  # noqa: PLC0415

        from takeout_rater.scorers.clip_backbone import get_clip_model  # noqa: PLC0415

        clip_model, preprocess, _tokenizer, device = get_clip_model()

        # Download and load the aesthetic MLP weights
        weights_path = self._download_mlp_weights()
        mlp = _build_mlp(_EMBEDDING_DIM)
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        mlp.load_state_dict(state_dict)
        mlp.eval()
        mlp.to(device)

        self._device = device
        self._clip_model = clip_model
        self._preprocess = preprocess
        self._mlp = mlp

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _preprocess_batch(self, chunk: list[Path]) -> tuple[Any, set[int]]:
        """Load and preprocess a chunk of images on the CPU.

        Args:
            chunk: Paths to the images in this chunk.

        Returns:
            A tuple ``(tensor, failed_indices)`` where *tensor* is a stacked
            ``torch.Tensor`` of shape ``(n, 3, 224, 224)`` on CPU and
            *failed_indices* is a ``set[int]`` of positions within *chunk*
            where image loading failed (``OSError``, ``ValueError``).
        """
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        tensors: list[Any] = []
        failed: set[int] = set()
        for i, path in enumerate(chunk):
            try:
                img = Image.open(path).convert("RGB")
                tensors.append(self._preprocess(img))
            except (OSError, ValueError):
                failed.add(i)
                tensors.append(None)

        valid_shape = next((t.shape for t in tensors if t is not None), None)
        if valid_shape is None:
            # Every image in this chunk failed — return a dummy tensor.
            return torch.zeros(len(chunk), 3, 224, 224), failed

        padded: list[Any] = [t if t is not None else torch.zeros(valid_shape) for t in tensors]
        return torch.stack(padded), failed

    def score_batch(
        self,
        image_paths: list[Path],
        *,
        variant_id: str | None = None,
    ) -> list[dict[str, float]]:
        """Score a batch of images for aesthetic quality.

        Images are processed in chunks of :data:`_SCORE_BATCH_SIZE` so that
        each chunk is forwarded through CLIP and the aesthetic MLP in a single
        GPU pass, making full use of hardware parallelism.

        CPU preprocessing (PIL decode + CLIP transforms) and GPU inference are
        **pipelined**: while the GPU processes chunk N, a background thread
        preprocesses chunk N+1 so that neither device sits idle waiting for
        the other.

        If a file cannot be opened (``OSError``, ``ValueError``), a zero tensor
        placeholder keeps the batch shape intact and the score defaults to
        ``0.0``.  If a whole-chunk forward pass raises ``RuntimeError`` (e.g.
        VRAM OOM), the chunk falls back to per-image scoring so that a single
        failure never silently zeros out an entire batch.

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

        chunks = [
            image_paths[start : start + _SCORE_BATCH_SIZE]
            for start in range(0, len(image_paths), _SCORE_BATCH_SIZE)
        ]

        # Use a cancel event so the background thread exits cleanly if the
        # main thread raises before draining the queue.
        cancel = threading.Event()
        prefetch_queue: queue.Queue[tuple[Any, set[int], list[Path]] | None] = queue.Queue(
            maxsize=_PREFETCH_BATCHES
        )

        def _producer() -> None:
            try:
                for chunk in chunks:
                    if cancel.is_set():
                        return
                    tensor, failed = self._preprocess_batch(chunk)
                    # Block on a full queue, but bail out if cancelled.
                    while True:
                        try:
                            prefetch_queue.put((tensor, failed, chunk), timeout=0.1)
                            break
                        except queue.Full:
                            if cancel.is_set():
                                return
            finally:
                prefetch_queue.put(None)  # sentinel — always placed so consumer unblocks

        producer = threading.Thread(target=_producer, daemon=True)
        producer.start()

        results: list[dict[str, float]] = []
        try:
            while True:
                item = prefetch_queue.get()
                if item is None:
                    break
                tensor, failed, chunk = item

                # All-failed shortcut — no forward pass needed.
                if len(failed) == len(chunk):
                    results.extend({"aesthetic": 0.0} for _ in chunk)
                    continue

                try:
                    batch = tensor.to(self._device)
                    with torch.no_grad():
                        embedding = self._clip_model.encode_image(batch)
                        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                        raw_scores: list[float] = self._mlp(embedding.float()).reshape(-1).tolist()

                    for i, raw in enumerate(raw_scores):
                        if i in failed:
                            results.append({"aesthetic": 0.0})
                        else:
                            results.append({"aesthetic": max(0.0, min(10.0, float(raw)))})

                except RuntimeError:
                    # Fallback: score each image individually (e.g. on VRAM OOM).
                    for i, path in enumerate(chunk):
                        if i in failed:
                            results.append({"aesthetic": 0.0})
                            continue
                        try:
                            img = Image.open(path).convert("RGB")
                            tensor = self._preprocess(img).unsqueeze(0).to(self._device)
                            with torch.no_grad():
                                embedding = self._clip_model.encode_image(tensor)
                                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                                raw_single: float = self._mlp(embedding.float()).item()
                            score = max(0.0, min(10.0, float(raw_single)))
                        except (OSError, ValueError, RuntimeError):
                            score = 0.0
                        results.append({"aesthetic": score})
        except Exception:
            cancel.set()
            # Drain the queue so the producer thread isn't blocked on a put(),
            # allowing it to observe the cancel event and exit promptly.
            try:
                while True:
                    prefetch_queue.get_nowait()
            except queue.Empty:
                pass
            raise
        finally:
            producer.join()

        return results
