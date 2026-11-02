"""Registry of the image/text backbones that semantic search can use.

Each backbone has a stable ``key`` that is also its storage key:

- ``clip-vit-l-14`` lives in the ``clip_embeddings`` table.  It is computed
  during indexing and shared with the CLIP-space scorers, clustering, the
  embedding map and image similarity.
- Every other backbone (currently SigLIP 2) lives in the model-keyed
  ``image_embeddings`` table and is computed on demand by the ``embed`` job.

Search with ``model=auto`` picks the first backbone in :data:`AUTO_ORDER` that
has embeddings stored.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from takeout_rater.db.queries import CLIP_L14_MODEL_KEY

SIGLIP2_MODEL_KEY = "siglip2-so400m-16-256"


@dataclass(frozen=True)
class SearchBackbone:
    """Static description of one search backbone."""

    key: str
    label: str
    description: str
    dim: int
    loader: Callable[[], tuple[Any, Any, Any, Any]]
    available: Callable[[], bool]


def _load_clip() -> tuple[Any, Any, Any, Any]:
    from takeout_rater.scoring.scorers.clip_backbone import get_clip_model

    return get_clip_model()


def _clip_available() -> bool:
    from takeout_rater.scoring.scorers.clip_backbone import is_available

    return is_available()


def _load_siglip2() -> tuple[Any, Any, Any, Any]:
    from takeout_rater.scoring.scorers.siglip2_backbone import get_siglip2_model

    return get_siglip2_model()


def _siglip2_available() -> bool:
    from takeout_rater.scoring.scorers.siglip2_backbone import is_available

    return is_available()


BACKBONES: dict[str, SearchBackbone] = {
    SIGLIP2_MODEL_KEY: SearchBackbone(
        key=SIGLIP2_MODEL_KEY,
        label="SigLIP 2 (So400m/16, 256 px)",
        description=(
            "Much better text-to-image matching than CLIP, and understands queries "
            "in many languages. Needs a one-off embedding pass (about 4.5 GB download)."
        ),
        dim=1152,
        loader=_load_siglip2,
        available=_siglip2_available,
    ),
    CLIP_L14_MODEL_KEY: SearchBackbone(
        key=CLIP_L14_MODEL_KEY,
        label="CLIP ViT-L/14 (OpenAI)",
        description="Computed during indexing. English queries only.",
        dim=768,
        loader=_load_clip,
        available=_clip_available,
    ),
}

#: Preference order for ``model=auto``.
AUTO_ORDER: tuple[str, ...] = (SIGLIP2_MODEL_KEY, CLIP_L14_MODEL_KEY)


def get_backbone(key: str) -> SearchBackbone:
    """Return the backbone for *key*; raise :class:`KeyError` if unknown."""
    return BACKBONES[key]


def resolve_backbone_key(requested: str, counts: dict[str, int]) -> str:
    """Resolve ``requested`` (a key or ``"auto"``) against embedding counts.

    ``"auto"`` (or an unknown key) returns the first backbone in
    :data:`AUTO_ORDER` with at least one stored embedding, falling back to
    CLIP L/14 when nothing is stored yet.
    """
    if requested in BACKBONES:
        return requested
    for key in AUTO_ORDER:
        if counts.get(key, 0) > 0:
            return key
    return CLIP_L14_MODEL_KEY


def _model_dtype(model: Any) -> Any:
    try:
        return next(model.parameters()).dtype
    except (StopIteration, AttributeError):
        return None


def encode_text(key: str, text: str) -> np.ndarray:
    """Encode *text* with backbone *key*; return an L2-normalised float32 vector."""
    import torch

    model, _preprocess, tokenizer, device = get_backbone(key).loader()
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.float().cpu().numpy().reshape(-1)


def encode_images(key: str, images: list[Any]) -> np.ndarray:
    """Encode PIL *images* with backbone *key*.

    Returns an ``(len(images), dim)`` float32 matrix of L2-normalised rows.
    """
    import torch

    model, preprocess, _tokenizer, device = get_backbone(key).loader()
    batch = torch.stack([preprocess(img) for img in images]).to(device)
    dtype = _model_dtype(model)
    if dtype is not None:
        batch = batch.to(dtype)
    with torch.no_grad():
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.float().cpu().numpy()
