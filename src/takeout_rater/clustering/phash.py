"""Perceptual hash (pHash) computation for near-duplicate detection.

The hash is computed using the *difference hash* (dhash) algorithm, which
requires only Pillow.  The result is a 256-bit integer encoded as a 64-character
hexadecimal string, stored in the ``phash`` table under ``algo = "dhash16"``.

Hamming distance between two hex hashes can be computed with
:func:`hamming_distance`.  A distance of ≤ 20 is a common threshold for
near-duplicate detection (out of 256 total bits) at this hash size.

Usage::

    from takeout_rater.clustering.phash import compute_dhash, compute_phash_all

    hex_hash = compute_dhash(Path("thumbnail.jpg"))
    count = compute_phash_all(conn, thumbs_dir, on_progress=print)
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

from takeout_rater.db.queries import list_asset_ids_without_phash, upsert_phash
from takeout_rater.indexing.thumbnailer import thumb_path_for_id

#: Algorithm name stored in the ``phash.algo`` column for the 256-bit dhash.
DHASH_ALGO = "dhash16"
_HASH_SIZE = 16  # produces a 256-bit hash (16 × 16 grid)


def compute_dhash_from_image(img: object, *, hash_size: int = _HASH_SIZE) -> str:
    """Compute the difference hash (dhash) from a PIL Image object.

    The image is resized to ``(hash_size + 1) × hash_size`` pixels, converted
    to greyscale, and the horizontal gradient is encoded as a ``hash_size²``-bit
    integer.

    Args:
        img: A PIL ``Image`` object (already opened).
        hash_size: Side length of the hash grid (default 16 → 256-bit hash).

    Returns:
        Hexadecimal string of length ``hash_size² / 4`` (64 chars for 256-bit).

    Raises:
        ImportError: If Pillow is not installed.
    """
    # img is already a PIL Image, so we just process it directly
    # (caller must import PIL.Image)
    img_gray = img.convert("L").resize((hash_size + 1, hash_size))  # type: ignore[union-attr]
    px = img_gray.load()
    bits = 0
    for idx in range(hash_size * hash_size):
        row = idx // hash_size
        col = idx % hash_size
        left = px[col, row]  # type: ignore[index]
        right = px[col + 1, row]  # type: ignore[index]
        if left > right:
            bits |= 1 << idx
    hex_chars = hash_size * hash_size // 4
    return f"{bits:0{hex_chars}x}"


def compute_dhash(image_path: Path, *, hash_size: int = _HASH_SIZE) -> str:
    """Compute the difference hash (dhash) of an image file.

    The image is resized to ``(hash_size + 1) × hash_size`` pixels, converted
    to greyscale, and the horizontal gradient is encoded as a ``hash_size²``-bit
    integer.

    Args:
        image_path: Path to the image file.
        hash_size: Side length of the hash grid (default 16 → 256-bit hash).

    Returns:
        Hexadecimal string of length ``hash_size² / 4`` (64 chars for 256-bit).

    Raises:
        OSError: If the image file cannot be opened.
        ImportError: If Pillow is not installed.
    """
    from PIL import Image

    with Image.open(image_path) as img:
        return compute_dhash_from_image(img, hash_size=hash_size)


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Return the Hamming distance between two hex-encoded dhash strings.

    Args:
        hash_a: First hex hash string.
        hash_b: Second hex hash string.

    Returns:
        Number of differing bits (0 = identical, 64 = completely different
        for 64-bit hashes).
    """
    diff = int(hash_a, 16) ^ int(hash_b, 16)
    return bin(diff).count("1")


def compute_phash_all(
    conn: sqlite3.Connection,
    thumbs_dir: Path,
    *,
    asset_ids: list[int] | None = None,
    batch_size: int = 64,
    on_progress: Callable[[int, int], None] | None = None,
    on_item: Callable[[int, int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> int:
    """Compute and persist dhash values for assets that lack one.

    Args:
        conn: Open library database connection.
        thumbs_dir: Directory containing thumbnail files.
        asset_ids: Explicit list of asset IDs to process.  When ``None``
            (default), all assets without a stored hash are processed.
        batch_size: Number of images to process before committing (default 64).
            No longer affects the frequency of ``on_progress`` calls (which
            now fire once per item), but retained for backward compatibility.
        on_progress: Optional callback invoked after **each item** with
            ``(processed_so_far, total)`` integers.
        on_item: Optional callback invoked before processing each item with
            ``(asset_id, processed_so_far, total)`` so callers can display the
            current item name.  Receives the raw DB asset ID which can be
            mapped to a filename by the caller.
        cancel_check: Optional callable that returns ``True`` when the run
            should be aborted.  Checked before each item; when it returns
            ``True`` the loop exits early.

    Returns:
        Number of hashes successfully written.

    Raises:
        ImportError: If Pillow is not installed.
    """
    # Verify Pillow is available up-front so callers get a clear error message
    # rather than silently writing 0 hashes.
    try:
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for perceptual hash computation. "
            "Install it with: pip install Pillow"
        ) from exc

    if asset_ids is None:
        asset_ids = list_asset_ids_without_phash(conn, algo=DHASH_ALGO)

    import logging

    logger = logging.getLogger(__name__)

    total = len(asset_ids)
    written = 0

    for i, aid in enumerate(asset_ids):
        if cancel_check is not None and cancel_check():
            break
        processed = i + 1
        if on_item is not None:
            on_item(aid, processed, total)
        thumb = thumb_path_for_id(thumbs_dir, aid)
        if thumb.exists():
            try:
                phash_hex = compute_dhash(thumb)
                upsert_phash(conn, aid, phash_hex, algo=DHASH_ALGO)
                written += 1
            except OSError as exc:
                logger.warning("Could not compute dhash for asset %d (%s): %s", aid, thumb, exc)
        if on_progress is not None:
            on_progress(processed, total)

    return written
