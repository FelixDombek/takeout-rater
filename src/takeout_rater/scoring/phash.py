"""Perceptual hash (pHash) computation for near-duplicate detection.

The hash is computed using the *difference hash* (dhash) algorithm, which
requires only Pillow.  The result is a 64-bit integer encoded as a 16-character
hexadecimal string, stored in the ``phash`` table under ``algo = "dhash"``.

Hamming distance between two hex hashes can be computed with
:func:`hamming_distance`.  A distance of ≤ 10 is a common threshold for
near-duplicate detection (out of 64 total bits).

Usage::

    from takeout_rater.scoring.phash import compute_dhash, compute_phash_all

    hex_hash = compute_dhash(Path("thumbnail.jpg"))
    count = compute_phash_all(conn, thumbs_dir, on_progress=print)
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

from takeout_rater.db.queries import list_asset_ids_without_phash, upsert_phash
from takeout_rater.indexing.thumbnailer import thumb_path_for_id

_DHASH_ALGO = "dhash"
_HASH_SIZE = 8  # produces a 64-bit hash (8 × 8 grid)


def compute_dhash(image_path: Path, *, hash_size: int = _HASH_SIZE) -> str:
    """Compute the difference hash (dhash) of an image file.

    The image is resized to ``(hash_size + 1) × hash_size`` pixels, converted
    to greyscale, and the horizontal gradient is encoded as a ``hash_size²``-bit
    integer.

    Args:
        image_path: Path to the image file.
        hash_size: Side length of the hash grid (default 8 → 64-bit hash).

    Returns:
        Hexadecimal string of length ``hash_size² / 4`` (16 chars for 64-bit).

    Raises:
        OSError: If the image file cannot be opened.
        ImportError: If Pillow is not installed.
    """
    from PIL import Image  # noqa: PLC0415

    with Image.open(image_path) as img:
        img = img.convert("L").resize((hash_size + 1, hash_size))
        px = img.load()
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
) -> int:
    """Compute and persist dhash values for assets that lack one.

    Args:
        conn: Open library database connection.
        thumbs_dir: Directory containing thumbnail files.
        asset_ids: Explicit list of asset IDs to process.  When ``None``
            (default), all assets without a stored hash are processed.
        batch_size: Number of images to process before committing (default 64).
        on_progress: Optional callback called after each batch with
            ``(processed_so_far, total)`` integers.

    Returns:
        Number of hashes successfully written.

    Raises:
        ImportError: If Pillow is not installed.
    """
    # Verify Pillow is available up-front so callers get a clear error message
    # rather than silently writing 0 hashes.
    try:
        from PIL import Image  # noqa: F401, PLC0415
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for perceptual hash computation. "
            "Install it with: pip install Pillow"
        ) from exc

    if asset_ids is None:
        asset_ids = list_asset_ids_without_phash(conn)

    import logging  # noqa: PLC0415

    logger = logging.getLogger(__name__)

    total = len(asset_ids)
    written = 0

    for batch_start in range(0, total, batch_size):
        batch = asset_ids[batch_start : batch_start + batch_size]
        for aid in batch:
            thumb = thumb_path_for_id(thumbs_dir, aid)
            if not thumb.exists():
                continue
            try:
                phash_hex = compute_dhash(thumb)
                upsert_phash(conn, aid, phash_hex, algo=_DHASH_ALGO)
                written += 1
            except OSError as exc:
                logger.warning("Could not compute dhash for asset %d (%s): %s", aid, thumb, exc)

        if on_progress is not None:
            on_progress(min(batch_start + batch_size, total), total)

    return written
