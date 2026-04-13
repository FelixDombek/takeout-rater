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
from typing import Any

# ---------------------------------------------------------------------------
# Model identifiers — must match the values used by the scorers.
# ---------------------------------------------------------------------------

CLIP_MODEL_NAME = "ViT-L-14"
CLIP_PRETRAINED = "openai"
EMBEDDING_DIM = 768

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

        import open_clip  # noqa: PLC0415
        import torch  # noqa: PLC0415

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, quick_gelu=True
        )
        model.eval()
        model.to(device)

        tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)

        _clip_model = model
        _preprocess = preprocess
        _tokenizer = tokenizer
        _device = device

    return _clip_model, _preprocess, _tokenizer, _device


def is_available() -> bool:
    """Return ``True`` when required CLIP dependencies are importable."""
    try:
        import open_clip  # noqa: F401
        import PIL  # noqa: F401
        import torch  # noqa: F401

        return True
    except ImportError:
        return False
