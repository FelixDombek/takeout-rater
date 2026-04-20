"""Shared CLIP ViT-L/14 backbone singleton.

Both :class:`AestheticScorer` and :class:`CLIPIQAScorer` use the same CLIP
ViT-L/14 backbone (``open_clip`` with ``openai`` weights).  Loading it
costs ~900 MB of RAM and takes a few seconds on CPU.  This module provides
a lazy singleton so the backbone is loaded at most once per process,
regardless of how many consumers need it.

Usage::

    model, preprocess, tokenizer, device = get_clip_model()
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Model identifiers — must match the values used by the scorers.
# ---------------------------------------------------------------------------

CLIP_MODEL_NAME = "ViT-L-14"
CLIP_PRETRAINED = "openai"
EMBEDDING_DIM = 768
CLIP_IMAGE_INPUT_NAME = "image"
CLIP_IMAGE_OUTPUT_NAME = "embedding"
CLIP_IMAGE_SIZE = 224
CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

# ---------------------------------------------------------------------------
# Module-level lazy singleton
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_clip_model: Any = None
_preprocess: Any = None
_tokenizer: Any = None
_device: Any = None


def get_clip_model() -> tuple[Any, Any, Any, Any]:
    """Return ``(model, preprocess, tokenizer, device)`` for CLIP ViT-L/14.

    The model is loaded lazily on the first call and cached for the lifetime
    of the process.  Subsequent calls return the cached objects immediately.

    Returns:
        A 4-tuple of ``(clip_model, preprocess, tokenizer, device)`` where

        - *clip_model* is the ``open_clip`` model in eval mode,
        - *preprocess* is the image-preprocessing transform,
        - *tokenizer* is the text tokenizer callable, and
        - *device* is the :class:`torch.device` the model lives on.
    """
    global _clip_model, _preprocess, _tokenizer, _device  # noqa: PLW0603

    if _clip_model is not None:
        return _clip_model, _preprocess, _tokenizer, _device

    with _lock:
        # Double-check after acquiring the lock.
        if _clip_model is not None:
            return _clip_model, _preprocess, _tokenizer, _device

        import open_clip
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, force_quick_gelu=True
        )
        model.eval()
        model.to(device)

        tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)

        _clip_model = model
        _preprocess = preprocess
        _tokenizer = tokenizer
        _device = device

    return _clip_model, _preprocess, _tokenizer, _device


def export_clip_image_onnx(cache_dir: Path) -> Path:
    """Export the shared CLIP image encoder to ONNX and return its path.

    The exported graph accepts a preprocessed image tensor shaped
    ``[batch, 3, 224, 224]`` and returns unnormalised image embeddings.  The
    caller is responsible for L2-normalising embeddings to match the Torch
    indexing path.
    """

    import contextlib
    import io
    import logging
    import warnings

    import torch

    model, _preprocess, _tokenizer, device = get_clip_model()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "clip_vit_l_14_openai_image_encoder_opset18.onnx"
    if path.exists():
        return path
    tmp_path = path.with_suffix(".tmp.onnx")

    class _ImageEncoderWrapper(torch.nn.Module):
        def __init__(self, clip_model: Any) -> None:
            super().__init__()
            self.clip_model = clip_model

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return self.clip_model.encode_image(image)

    wrapper = _ImageEncoderWrapper(model).eval().to(device)
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    export_output = io.StringIO()
    mha_backend = getattr(getattr(torch.backends, "mha", None), "set_fastpath_enabled", None)
    old_mha_fastpath = (
        torch.backends.mha.get_fastpath_enabled() if mha_backend is not None else None
    )
    try:
        if mha_backend is not None:
            mha_backend(False)
        with (
            torch.inference_mode(),
            warnings.catch_warnings(),
            contextlib.redirect_stdout(export_output),
            contextlib.redirect_stderr(export_output),
        ):
            warnings.simplefilter("ignore")
            torch.onnx.export(
                wrapper,
                dummy,
                tmp_path,
                input_names=[CLIP_IMAGE_INPUT_NAME],
                output_names=[CLIP_IMAGE_OUTPUT_NAME],
                dynamic_axes={
                    CLIP_IMAGE_INPUT_NAME: {0: "batch"},
                    CLIP_IMAGE_OUTPUT_NAME: {0: "batch"},
                },
                opset_version=18,
                dynamo=False,
            )
    except Exception:
        tmp_path.unlink(missing_ok=True)
        captured = export_output.getvalue().strip()
        if captured:
            logging.getLogger(__name__).debug(
                "Captured ONNX exporter output before failure:\n%s",
                captured[-8000:],
            )
        raise
    finally:
        if mha_backend is not None and old_mha_fastpath is not None:
            mha_backend(old_mha_fastpath)
    tmp_path.replace(path)
    return path


def preprocess_clip_image_fast(image: object) -> Any:
    """Preprocess a PIL image for the hard-coded OpenAI CLIP ViT-L/14 model.

    The indexing pipeline already works with small thumbnail images, so using
    the full OpenCLIP torchvision transform adds avoidable overhead.  This
    helper preserves the same input contract: RGB image, bicubic center crop to
    224 px, convert to CHW float tensor in [0, 1], then apply OpenAI CLIP
    normalization.
    """

    import numpy as np
    import torch
    from PIL import Image, ImageOps

    resampling = getattr(Image, "Resampling", Image).BICUBIC
    if image.mode != "RGB":  # type: ignore[union-attr]
        image = image.convert("RGB")  # type: ignore[union-attr]
    image = ImageOps.fit(
        image,  # type: ignore[arg-type]
        (CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE),
        method=resampling,
        centering=(0.5, 0.5),
    )
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(CLIP_IMAGE_MEAN, dtype=np.float32)) / np.asarray(
        CLIP_IMAGE_STD, dtype=np.float32
    )
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def get_clip_onnx_session(
    cache_dir: Path, accelerator: str = "auto"
) -> tuple[Any, Any, list[str]] | None:
    """Return ``(session, preprocess, active_providers)`` for ONNX CLIP image inference.

    ``accelerator`` accepts ``"auto"``, ``"tensorrt"``, ``"onnx"``/``"cuda"``,
    or ``"torch"``.  ``"auto"`` uses ONNX Runtime CUDA when available but does
    not try TensorRT, because TensorRT engine builds are hardware/install
    sensitive and can emit noisy provider logs.  ``None`` is returned when ONNX
    Runtime is unavailable or when ``accelerator`` explicitly selects Torch.
    """

    if accelerator == "torch":
        return None

    try:
        import onnxruntime as ort
    except ImportError:
        return None

    available = set(ort.get_available_providers())
    if accelerator == "auto" and "CUDAExecutionProvider" not in available:
        return None

    _model, preprocess, _tokenizer, _device = get_clip_model()
    onnx_path = export_clip_image_onnx(cache_dir)
    providers: list[Any] = []

    trt_cache_dir = cache_dir / "tensorrt"
    if accelerator == "tensorrt" and "TensorrtExecutionProvider" in available:
        providers.append(
            (
                "TensorrtExecutionProvider",
                {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(trt_cache_dir),
                    "trt_fp16_enable": True,
                },
            )
        )
    if accelerator in {"auto", "tensorrt", "onnx", "cuda"} and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    try:
        session = ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception:
        return None
    return session, preprocess, session.get_providers()


def is_available() -> bool:
    """Return ``True`` when required CLIP dependencies are importable."""
    try:
        import open_clip  # noqa: F401
        import PIL  # noqa: F401
        import torch  # noqa: F401

        return True
    except ImportError:
        return False
