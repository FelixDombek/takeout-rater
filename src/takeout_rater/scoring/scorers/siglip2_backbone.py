"""Shared SigLIP 2 backbone singleton (search only).

SigLIP 2 So400m/16 at 256 px (``open_clip`` with ``webli`` weights) is a much
stronger text-to-image retriever than OpenAI CLIP ViT-L/14 (about 56 vs 37
COCO text→image R@1 in the SigLIP 2 paper) and is multilingual, so queries in
German and other languages work.  It is used only for semantic search: the
LAION, cafe_style, clip_iqa and clip_vocab scorers stay on CLIP L/14 (see
:mod:`takeout_rater.scoring.scorers.clip_backbone`).

Usage::

    model, preprocess, tokenizer, device = get_siglip2_model()
"""

from __future__ import annotations

import functools
import threading
from typing import Any

SIGLIP2_MODEL_NAME = "ViT-SO400M-16-SigLIP2-256"
SIGLIP2_PRETRAINED = "webli"
SIGLIP2_EMBEDDING_DIM = 1152

_lock = threading.Lock()
_model: Any = None
_preprocess: Any = None
_tokenizer: Any = None
_device: Any = None


def get_siglip2_model() -> tuple[Any, Any, Any, Any]:
    """Return ``(model, preprocess, tokenizer, device)`` for SigLIP 2.

    Loaded lazily on first call (weights are downloaded from the Hugging Face
    hub on first use, about 4.5 GB) and cached for the life of the process.
    On CUDA the model runs in float16 to halve memory use.
    """
    global _model, _preprocess, _tokenizer, _device  # noqa: PLW0603

    if _model is not None:
        return _model, _preprocess, _tokenizer, _device

    with _lock:
        if _model is not None:
            return _model, _preprocess, _tokenizer, _device

        import open_clip
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _, preprocess = open_clip.create_model_and_transforms(
            SIGLIP2_MODEL_NAME, pretrained=SIGLIP2_PRETRAINED
        )
        model.eval()
        if device.type == "cuda":
            model = model.half()
        model.to(device)
        tokenizer = open_clip.get_tokenizer(SIGLIP2_MODEL_NAME)

        _model, _preprocess, _tokenizer, _device = model, preprocess, tokenizer, device

    return _model, _preprocess, _tokenizer, _device


@functools.cache
def is_available() -> bool:
    """Return ``True`` when open_clip is installed and knows the SigLIP 2 model."""
    try:
        import open_clip
        import torch  # noqa: F401
    except ImportError:
        return False
    try:
        return SIGLIP2_MODEL_NAME in open_clip.list_models()
    except Exception:  # noqa: BLE001
        return False
