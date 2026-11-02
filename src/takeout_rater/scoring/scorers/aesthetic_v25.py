"""Aesthetic Predictor V2.5 scorer (SigLIP so400m + MLP head).

Aesthetic Predictor V2.5 (https://github.com/discus0434/aesthetic-predictor-v2-5)
is the successor of the LAION Aesthetic Predictor v2.  Instead of CLIP
ViT-L/14 it uses Google's ``siglip-so400m-patch14-384`` vision tower and a
small MLP head trained on a larger and more diverse aesthetic dataset, so it
rates real-world photos (not only AI art) more reliably.

The output metric ``aesthetic`` is on a roughly 1–10 scale; the authors
consider 5.5+ "good".

Model weights are downloaded on first use: the SigLIP vision tower (~1.7 GB in
safetensors) from the HuggingFace Hub into ``~/.cache/huggingface``, and the
MLP head (~4 MB) from the project's GitHub repository into the torch hub cache.

License note: the ``aesthetic-predictor-v2-5`` package and its head weights are
AGPL-3.0.  takeout-rater is GPL-3.0; GPL-3.0 §13 explicitly permits combining
the two, with the AGPL part keeping its own terms.

Performance notes (see the 2026-11 performance audit):

- Each thumbnail is decoded and resized exactly **once**, in a small thread
  pool, straight to the 384×384 model input.  The HuggingFace image processor
  is not used on the hot path; :func:`preprocess_image` reproduces its
  resize/rescale/normalise steps, with the normalisation done on the device.
- Images are sent to the device as ``uint8`` (4× less transfer than float32).
- CPU decoding of sub-batch N+1 overlaps device inference of sub-batch N.
- On CUDA the model runs in bfloat16 (as recommended upstream) when the GPU
  supports it; a CUDA out-of-memory error halves the sub-batch instead of
  dropping to one image at a time.
- The loaded model is cached per process, so starting a new scoring job does
  not reload 1.7 GB of weights.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from takeout_rater.scoring.scorers.base import (
    BaseScorer,
    MetricSpec,
    ScorerSpec,
    VariantSpec,
    _run_pipelined_batches,
)

#: HuggingFace identifier of the SigLIP vision tower the head was trained on.
_ENCODER_MODEL = "google/siglip-so400m-patch14-384"

#: Square input resolution of the SigLIP so400m-patch14-384 tower.
INPUT_SIZE = 384

#: Images per forward pass.  SO400M at 384 px is ~4× the compute of
#: ViT-L/14 at 224 px, so this is smaller than the CLIP scorers' batch.
#: Override with ``TAKEOUT_RATER_AESTHETIC_V25_BATCH_SIZE``.
_DEFAULT_INFER_BATCH_SIZE = 16

#: Number of decoded sub-batches buffered ahead of inference.
_PREFETCH_BATCHES = 2

#: Value written for images that cannot be decoded.
_FAILED_SCORE = 0.0

# Process-wide model cache: (model, device, dtype).
_model_lock = threading.Lock()
_cached_model: tuple[Any, Any, Any] | None = None


def _infer_batch_size() -> int:
    raw = os.environ.get("TAKEOUT_RATER_AESTHETIC_V25_BATCH_SIZE")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return _DEFAULT_INFER_BATCH_SIZE


def _decode_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def preprocess_image(path: Path) -> Any:
    """Decode *path* and return a ``(384, 384, 3)`` ``uint8`` array.

    Matches ``SiglipImageProcessor``'s resize step for this model (direct
    resize to 384×384 with bicubic resampling, no crop).  Rescaling and
    normalisation (``x / 255``, mean 0.5, std 0.5) are applied later on the
    device by :func:`normalize_pixels`.

    Raises:
        OSError, ValueError: When the image cannot be decoded.
    """
    import numpy as np
    from PIL import Image

    with Image.open(path) as img:
        rgb = img.convert("RGB")
    resized = rgb.resize((INPUT_SIZE, INPUT_SIZE), resample=Image.Resampling.BICUBIC)
    return np.array(resized, dtype=np.uint8)


def normalize_pixels(batch_uint8: Any, dtype: Any) -> Any:
    """Convert an ``(N, H, W, 3)`` uint8 tensor to normalised ``(N, 3, H, W)`` *dtype*.

    Equivalent to SigLIP's ``rescale(1/255)`` followed by
    ``normalize(mean=0.5, std=0.5)``, i.e. ``x / 127.5 - 1``.
    """
    x = batch_uint8.permute(0, 3, 1, 2).float()
    return (x / 127.5 - 1.0).to(dtype)


def _load_model() -> tuple[Any, Any, Any]:
    """Load (or return the cached) predictor model, device and dtype."""
    global _cached_model  # noqa: PLW0603
    with _model_lock:
        if _cached_model is not None:
            return _cached_model

        import torch
        from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

        model, _processor = convert_v2_5_from_siglip(
            encoder_model_name=_ENCODER_MODEL,
            low_cpu_mem_usage=True,
        )
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        else:
            device = torch.device("cpu")
            dtype = torch.float32
        model = model.to(device=device, dtype=dtype).eval()
        _cached_model = (model, device, dtype)
        return _cached_model


class AestheticV25Scorer(BaseScorer):
    """Aesthetic Predictor V2.5: SigLIP so400m embeddings + MLP head, ~1–10."""

    def __init__(self, variant_id: str | None = None, **kwargs: Any) -> None:
        super().__init__(variant_id=variant_id, **kwargs)
        self._model: Any = None
        self._device: Any = None
        self._dtype: Any = None
        self._pool: ThreadPoolExecutor | None = None

    # ------------------------------------------------------------------
    # Class-level API
    # ------------------------------------------------------------------

    @classmethod
    def spec(cls) -> ScorerSpec:
        return ScorerSpec(
            scorer_id="aesthetic_v25",
            display_name="Aesthetic Predictor V2.5",
            description=(
                "Rates how beautiful or visually appealing a photo is on a 1–10 scale, "
                "like the LAION aesthetic score but with a newer, stronger model that "
                "handles everyday photos better. Around 5.5 and up is generally good. "
                "Slower than LAION; requires a ~1.7 GB model download. "
                "Model license: AGPL-3.0."
            ),
            technical_description=(
                "Aesthetic Predictor V2.5 (discus0434): MLP head on L2-normalised "
                "pooled embeddings of google/siglip-so400m-patch14-384 (384 px input, "
                "bicubic resize, no crop). Successor to LAION Aesthetic Predictor v2. "
                "bfloat16 on CUDA when supported, float32 otherwise."
            ),
            version="1",
            variants=(
                VariantSpec(
                    variant_id="siglip_so400m",
                    display_name="SigLIP so400m (V2.5)",
                    description=(
                        "Aesthetic Predictor V2.5 head on google/siglip-so400m-patch14-384."
                    ),
                    metrics=(
                        MetricSpec(
                            key="aesthetic",
                            display_name="Aesthetic V2.5",
                            description=(
                                "Predicted aesthetic quality, roughly 1–10. "
                                "5.5+ is considered good. Higher is better."
                            ),
                            min_value=1.0,
                            max_value=10.0,
                            higher_is_better=True,
                        ),
                    ),
                ),
            ),
            default_variant_id="siglip_so400m",
            requires_extras=(),
        )

    @classmethod
    def is_available(cls) -> bool:
        """Return ``True`` when torch, transformers and the predictor package import."""
        try:
            import aesthetic_predictor_v2_5  # noqa: F401
            import numpy  # noqa: F401
            import PIL  # noqa: F401
            import torch  # noqa: F401
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._model, self._device, self._dtype = _load_model()
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=_decode_workers(), thread_name_prefix="aesthetic-v25-decode"
            )

    def close(self) -> None:
        """Shut down the decode thread pool (the model stays cached)."""
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _forward(self, pixels_uint8: Any) -> list[float]:
        """Run the model on an ``(N, H, W, 3)`` uint8 CPU tensor.

        Halves the batch on CUDA out-of-memory instead of failing.
        """
        import torch

        try:
            with torch.inference_mode():
                batch = pixels_uint8.to(self._device, non_blocking=True)
                pixel_values = normalize_pixels(batch, self._dtype)
                logits = self._model(pixel_values=pixel_values, return_dict=True).logits
                return logits.float().reshape(-1).cpu().tolist()
        except torch.cuda.OutOfMemoryError:
            if pixels_uint8.shape[0] <= 1:
                raise
            torch.cuda.empty_cache()
            half = pixels_uint8.shape[0] // 2
            return self._forward(pixels_uint8[:half]) + self._forward(pixels_uint8[half:])

    def score_batch(
        self,
        image_paths: list[Path],
        *,
        variant_id: str | None = None,
    ) -> list[dict[str, float]]:
        """Score a batch of images.

        Args:
            image_paths: Paths to image files (512 px thumbnails are ideal).
            variant_id: Ignored; only one variant exists.

        Returns:
            A list (same length as *image_paths*) of ``{"aesthetic": float}``.
            Images that cannot be decoded get ``0.0``.
        """
        if not image_paths:
            return []

        import numpy as np
        import torch

        self._ensure_loaded()
        assert self._pool is not None
        pool = self._pool
        use_pinned = self._device is not None and self._device.type == "cuda"

        step = _infer_batch_size()
        chunks = [image_paths[i : i + step] for i in range(0, len(image_paths), step)]

        def _decode_one(path: Path) -> Any:
            try:
                return preprocess_image(path)
            except (OSError, ValueError, RuntimeError, SyntaxError):
                return None

        def _preprocess(chunk: list[Path]) -> tuple[Any, set[int]]:
            arrays = list(pool.map(_decode_one, chunk))
            failed = {i for i, a in enumerate(arrays) if a is None}
            valid = [a for a in arrays if a is not None]
            if not valid:
                return None, failed
            tensor = torch.from_numpy(np.stack(valid))
            if use_pinned:
                tensor = tensor.pin_memory()
            return tensor, failed

        def _infer(tensor: Any, failed: set[int], chunk: list[Path]) -> list[dict[str, float]]:
            scores = [_FAILED_SCORE] * len(chunk)
            if tensor is not None:
                valid_scores = self._forward(tensor)
                valid_idx = [i for i in range(len(chunk)) if i not in failed]
                for idx, value in zip(valid_idx, valid_scores, strict=True):
                    scores[idx] = float(value)
            return [{"aesthetic": s} for s in scores]

        return _run_pipelined_batches(chunks, _preprocess, _infer, prefetch=_PREFETCH_BATCHES)
