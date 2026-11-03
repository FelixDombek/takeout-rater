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
the two, with the AGPL part keeping its own terms.  The package is installed
from PyPI and the weights are downloaded from upstream; nothing AGPL is
vendored in this repository.

Performance notes (see the 2026-11 performance audit):

- Each thumbnail is decoded and resized exactly **once**, in a small thread
  pool, straight to the 384×384 model input.  The HuggingFace image processor
  is not used on the hot path; :func:`preprocess_image` reproduces its
  resize/rescale/normalise steps, with the normalisation done on the device.
- Images are sent to the device as ``uint8`` (4× less transfer than float32).
- CPU decoding of sub-batch N+1 overlaps device inference of sub-batch N.
- A CUDA out-of-memory error halves the sub-batch instead of dropping to one
  image at a time.
- The vision tower runs in bfloat16 on CUDA, but the normalisation and MLP
  head stay in float32, so scores are not quantised and match the CPU path.
- The model is released (and the CUDA cache emptied) when the scorer is
  closed at the end of a scoring run, so it does not keep ~1.7 GB of VRAM.
"""

from __future__ import annotations

import importlib.util
import os
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

#: Clamp range of the ``aesthetic`` metric (matches the declared MetricSpec).
_MIN_SCORE = 1.0
_MAX_SCORE = 10.0


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

    Matches the PIL-based ``SiglipImageProcessor`` the head was trained with
    (direct resize to 384×384 with PIL bicubic resampling, no crop).  Note that
    transformers 5 defaults to a torchvision backend whose resize differs
    slightly; this function deliberately keeps the PIL behaviour.  Rescaling and
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


def pooled_embeddings(model: Any, pixel_values: Any) -> Any:
    """Return the SigLIP tower's pooled output for *pixel_values*.

    Calls ``SiglipVisionModel.forward`` directly, skipping the predictor's own
    forward (which would run the head in the tower's dtype).  This works with
    both transformers 4 (tower under ``.vision_model``) and 5 (flattened).
    """
    from transformers import SiglipVisionModel

    return SiglipVisionModel.forward(model, pixel_values=pixel_values).pooler_output


def _load_model() -> tuple[Any, Any, Any]:
    """Load the predictor and return ``(model, device, tower_dtype)``.

    The SigLIP vision tower runs in bfloat16 on CUDA when supported (as
    recommended upstream); the L2-normalisation and the MLP head always run in
    float32 so scores are not quantised to bf16 steps and match across devices.
    The model is owned by the scorer instance and freed by
    :meth:`AestheticV25Scorer.close`, so it does not hold ~1.7 GB of VRAM for
    the rest of the process.
    """
    import torch
    from aesthetic_predictor_v2_5 import AestheticPredictorV2_5Model
    from aesthetic_predictor_v2_5.siglip_v2_5 import URL as HEAD_URL
    from transformers import SiglipVisionConfig

    # The checkpoint is a full SigLIP (text + vision) model.  transformers 5
    # no longer extracts the vision part on its own and fails with
    # "'SiglipConfig' object has no attribute 'hidden_size'", so pass it.
    vision_config = SiglipVisionConfig.from_pretrained(_ENCODER_MODEL)
    model = AestheticPredictorV2_5Model.from_pretrained(
        _ENCODER_MODEL, config=vision_config, low_cpu_mem_usage=True
    )
    head_state = torch.hub.load_state_dict_from_url(HEAD_URL, map_location="cpu")
    model.layers.load_state_dict(head_state)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        tower_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    else:
        device = torch.device("cpu")
        tower_dtype = torch.float32
    model = model.to(device=device, dtype=tower_dtype)
    model.layers.to(dtype=torch.float32)
    model.eval()
    return model, device, tower_dtype


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
                "Vision tower in bfloat16 on CUDA when supported; head always float32; "
                "clamped to 1–10."
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
        """Return ``True`` when the runtime dependencies are installed (no imports)."""
        modules = ("aesthetic_predictor_v2_5", "numpy", "PIL", "torch", "transformers")
        try:
            return all(importlib.util.find_spec(m) is not None for m in modules)
        except (ImportError, ValueError):
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
        """Shut down the decode pool and release the model (and its VRAM)."""
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
        if self._model is not None:
            on_cuda = self._device is not None and self._device.type == "cuda"
            self._model = None
            if on_cuda:
                import torch

                torch.cuda.empty_cache()

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        # Plain try/except: at interpreter shutdown module globals such as
        # ``contextlib`` may already be None.
        try:  # noqa: SIM105
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _forward(self, pixels_uint8: Any) -> list[float]:
        """Run the model on an ``(N, H, W, 3)`` uint8 CPU tensor.

        Halves the batch on CUDA out-of-memory instead of failing.
        """
        import torch

        try:
            return self._forward_once(pixels_uint8)
        except torch.cuda.OutOfMemoryError:
            if pixels_uint8.shape[0] <= 1:
                raise
        # Retry outside the ``except`` block: the handled exception's traceback
        # (and the failed attempt's device tensors in its frames) is released
        # once the handler exits, so empty_cache() can actually free them.
        torch.cuda.empty_cache()
        half = pixels_uint8.shape[0] // 2
        return self._forward(pixels_uint8[:half]) + self._forward(pixels_uint8[half:])

    def _forward_once(self, pixels_uint8: Any) -> list[float]:
        import torch

        with torch.inference_mode():
            batch = pixels_uint8.to(self._device, non_blocking=True)
            pixel_values = normalize_pixels(batch, self._dtype)
            # Same computation as AestheticPredictorV2_5Model.forward, but the
            # normalisation and head run in float32 (see _load_model).
            pooled = pooled_embeddings(self._model, pixel_values).float()
            embeds = pooled / pooled.norm(dim=-1, keepdim=True)
            scores = self._model.layers(embeds).reshape(-1)
            return scores.clamp(_MIN_SCORE, _MAX_SCORE).cpu().tolist()

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
            A list (same length as *image_paths*) of ``{"aesthetic": float}``
            in ``[1, 10]``.  Images that cannot be decoded get an **empty**
            dict: no score row is written, so they stay unscored (and are
            retried on the next run) instead of sorting as the worst photos.
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
            except Exception:  # noqa: BLE001 - e.g. DecompressionBombError, MemoryError
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
            results: list[dict[str, float]] = [{} for _ in chunk]
            if tensor is not None:
                valid_scores = self._forward(tensor)
                valid_idx = [i for i in range(len(chunk)) if i not in failed]
                for idx, value in zip(valid_idx, valid_scores, strict=True):
                    results[idx] = {"aesthetic": float(value)}
            return results

        return _run_pipelined_batches(chunks, _preprocess, _infer, prefetch=_PREFETCH_BATCHES)
