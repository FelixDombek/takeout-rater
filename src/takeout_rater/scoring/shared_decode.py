"""Decode each thumbnail once when several scorers run over the same assets.

A "score all" run used to open and decode every thumbnail once per scorer
variant (about 15 times per photo).  :func:`run_scorer_group` in
:mod:`takeout_rater.scoring.pipeline` now decodes a batch of thumbnails once,
installs the decoded RGB images here with :func:`shared_images`, and then
calls every scorer of the group on that batch.

Scorers load images through :func:`open_rgb` instead of
``Image.open(path).convert("RGB")``.  Outside a shared pass (a single scorer
run, tests) :func:`open_rgb` simply decodes the file, so behaviour and pixels
are unchanged; inside a pass it returns a copy of the already decoded image.

Only one scoring job runs at a time, and scorers read images from their own
worker threads, so the active cache is a module-level dict guarded by a lock
rather than a context variable (which would not follow the scorers' threads).
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

_lock = threading.Lock()
_images: Mapping[str, Any] | None = None


def decode_rgb(path: Path) -> Any:
    """Decode *path* into a loaded RGB :class:`PIL.Image.Image`.

    Raises the same ``OSError`` / ``ValueError`` as ``Image.open`` does for
    unreadable files.
    """
    from PIL import Image

    with Image.open(path) as img:
        return img.convert("RGB")


def open_rgb(path: Path) -> Any:
    """Return the RGB image for *path*, from the shared cache when available.

    The returned image is always the caller's own object (a copy when it came
    from the cache), so scorers may modify it freely.
    """
    with _lock:
        cached = _images.get(str(path)) if _images is not None else None
    if cached is not None:
        return cached.copy()
    return decode_rgb(path)


@contextlib.contextmanager
def shared_images(images: Mapping[str, Any]) -> Iterator[None]:
    """Make *images* (``str(path)`` → RGB image) visible to :func:`open_rgb`.

    Paths that are missing from *images* (for example because they failed to
    decode) fall back to decoding the file, so error handling inside the
    scorers stays exactly as it was.
    """
    global _images  # noqa: PLW0603
    with _lock:
        previous = _images
        _images = images
    try:
        yield
    finally:
        with _lock:
            _images = previous
