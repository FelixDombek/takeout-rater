"""Generate thumbnail images for indexed assets.

Thumbnails are stored as JPEG files in a bucketed directory structure
under ``<db_root>/takeout-rater/thumbs/``.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass

THUMB_MAX_PX: int = 512
THUMB_FORMAT: str = "JPEG"
THUMB_QUALITY: int = 85


def _empty_timings() -> dict[str, float]:
    return {"decode": 0.0, "resize": 0.0, "write": 0.0}


def _request_decoder_downscale(img: object) -> None:
    """Ask Pillow-backed decoders to reduce at load time when supported."""
    draft = getattr(img, "draft", None)
    if callable(draft):
        try:
            draft("RGB", (THUMB_MAX_PX, THUMB_MAX_PX))
        except Exception:  # noqa: BLE001
            return


def thumb_path_for_id(thumbs_dir: Path, asset_id: int) -> Path:
    """Return the path where a thumbnail for *asset_id* should be stored.

    Assets are bucketed by thousands to keep directory sizes manageable.
    For example, asset 42 → ``<thumbs_dir>/0000/42.jpg``.

    Args:
        thumbs_dir: Base thumbs directory (``<library>/takeout-rater/thumbs/``).
        asset_id: The integer asset ID from the database.

    Returns:
        The destination JPEG path (may not exist yet).
    """
    bucket = f"{asset_id // 1000:04d}"
    return thumbs_dir / bucket / f"{asset_id}.jpg"


def generate_thumbnail_from_image(img: object, output_path: Path) -> object:
    """Generate a ≤512 px JPEG thumbnail from an already-open PIL Image.

    Applies EXIF orientation, converts to RGB, resizes to at most
    :data:`THUMB_MAX_PX` in each dimension, saves the result to
    *output_path*, and returns the thumbnail as a new PIL Image object.
    The caller can then use the returned image for further processing
    (e.g. phash or CLIP embedding) without re-reading the file.

    Args:
        img: An already-open ``PIL.Image.Image`` instance.
        output_path: Destination path for the thumbnail JPEG.
            Parent directories are created automatically.

    Returns:
        The thumbnail as a ``PIL.Image.Image`` (RGB mode, ≤512 px).

    Raises:
        ImportError: If Pillow is not installed.
        OSError: If the thumbnail cannot be written.
    """
    try:
        from PIL import ImageOps
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for thumbnail generation. Install it with: pip install Pillow>=10.0"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Apply EXIF orientation on the source image (mutates a copy).
    _request_decoder_downscale(img)
    img = ImageOps.exif_transpose(img) or img  # type: ignore[arg-type]
    if img.mode not in ("RGB", "L"):  # type: ignore[union-attr]
        img = img.convert("RGB")  # type: ignore[union-attr]
    thumb = img.copy()  # type: ignore[union-attr]
    thumb.thumbnail((THUMB_MAX_PX, THUMB_MAX_PX))
    # Ensure RGB before saving as JPEG (L mode is fine, but be safe).
    if thumb.mode not in ("RGB", "L"):
        thumb = thumb.convert("RGB")
    thumb.save(output_path, format=THUMB_FORMAT, quality=THUMB_QUALITY, optimize=True)
    return thumb


def generate_thumbnail_from_image_timed(
    img: object, output_path: Path
) -> tuple[object, dict[str, float]]:
    """Generate a thumbnail from an open image and return stage timings."""
    try:
        from PIL import ImageOps
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for thumbnail generation. Install it with: pip install Pillow>=10.0"
        ) from exc

    timings = _empty_timings()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start = perf_counter()
    _request_decoder_downscale(img)
    img = ImageOps.exif_transpose(img) or img  # type: ignore[arg-type]
    if img.mode not in ("RGB", "L"):  # type: ignore[union-attr]
        img = img.convert("RGB")  # type: ignore[union-attr]
    thumb = img.copy()  # type: ignore[union-attr]
    timings["decode"] += perf_counter() - start

    start = perf_counter()
    thumb.thumbnail((THUMB_MAX_PX, THUMB_MAX_PX))
    if thumb.mode not in ("RGB", "L"):
        thumb = thumb.convert("RGB")
    timings["resize"] += perf_counter() - start

    start = perf_counter()
    thumb.save(output_path, format=THUMB_FORMAT, quality=THUMB_QUALITY, optimize=True)
    timings["write"] += perf_counter() - start
    return thumb, timings


def generate_thumbnail(image_path: Path, output_path: Path) -> None:
    """Generate a ≤512 px JPEG thumbnail and write it to *output_path*.

    EXIF orientation metadata is applied (via :func:`PIL.ImageOps.exif_transpose`)
    before resizing, so the thumbnail always reflects the correct visual orientation.
    The image is resized so that neither dimension exceeds :data:`THUMB_MAX_PX`
    while preserving the aspect ratio.  The output is always a JPEG regardless
    of the source format.

    Args:
        image_path: Absolute path to the source image file.
        output_path: Destination path for the thumbnail JPEG.
            Parent directories are created automatically.

    Raises:
        ImportError: If Pillow is not installed.
        OSError: If the image cannot be read or the thumbnail cannot be written.
    """
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for thumbnail generation. Install it with: pip install Pillow>=10.0"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as img:
        _request_decoder_downscale(img)
        # Apply EXIF orientation so thumbnails are always visually correct.
        img = ImageOps.exif_transpose(img) or img
        # Convert to RGB so JPEG output always succeeds (handles RGBA, P, etc.)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.thumbnail((THUMB_MAX_PX, THUMB_MAX_PX))
        img.save(output_path, format=THUMB_FORMAT, quality=THUMB_QUALITY, optimize=True)


def generate_thumbnail_timed(image_path: Path, output_path: Path) -> dict[str, float]:
    """Generate a thumbnail from a path and return decode/resize/write timings."""
    _thumb, timings = generate_thumbnail_from_path_timed(image_path, output_path)
    return timings


def generate_thumbnail_from_path_timed(
    image_path: Path, output_path: Path
) -> tuple[object, dict[str, float]]:
    """Generate a thumbnail from a path and return the image plus timings."""
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for thumbnail generation. Install it with: pip install Pillow>=10.0"
        ) from exc

    timings = _empty_timings()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as img:
        start = perf_counter()
        _request_decoder_downscale(img)
        img = ImageOps.exif_transpose(img) or img
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        timings["decode"] += perf_counter() - start

        start = perf_counter()
        img.thumbnail((THUMB_MAX_PX, THUMB_MAX_PX))
        timings["resize"] += perf_counter() - start

        start = perf_counter()
        img.save(output_path, format=THUMB_FORMAT, quality=THUMB_QUALITY, optimize=True)
        timings["write"] += perf_counter() - start
        thumb = img.copy()

    return thumb, timings
