"""Thin wrapper around RapidOCR (PaddleOCR models on ONNX Runtime).

RapidOCR runs a text detector, an angle classifier and a recogniser, all as
ONNX models.  It reuses the ``onnxruntime-gpu`` install that InsightFace
already needs, so no extra runtime is pulled in.  The default PP-OCR models
(~20 MB) ship inside the ``rapidocr`` wheel, so no download is needed.

The wrapper is deliberately small: it turns one BGR ``numpy`` image into an
:class:`OcrResult` (recognised lines + the joined text that is stored in the
``ocr_text`` table and indexed by FTS5).
"""

from __future__ import annotations

import contextlib
import functools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

#: Longest side, in pixels, that original images are downscaled to before OCR.
#: RapidOCR's detector works at ~736 px anyway; the extra resolution only helps
#: the recogniser read small text inside detected boxes.
ORIGINAL_MAX_SIDE = 2048

#: Lines recognised with a confidence below this are dropped by default.
DEFAULT_MIN_SCORE = 0.6


@dataclass
class OcrLine:
    """One recognised line of text."""

    text: str
    score: float


@dataclass
class OcrResult:
    """OCR output for one image."""

    lines: list[OcrLine] = field(default_factory=list)

    @property
    def text(self) -> str:
        """All kept lines joined by newlines (``''`` when nothing was found)."""
        return "\n".join(line.text for line in self.lines)

    @property
    def mean_score(self) -> float | None:
        if not self.lines:
            return None
        return sum(line.score for line in self.lines) / len(self.lines)


@functools.cache
def rapidocr_available() -> bool:
    """Return ``True`` if the ``rapidocr`` package can be imported (cached)."""
    try:
        import rapidocr  # noqa: F401
    except ImportError:
        return False
    return True


@functools.cache
def cuda_available() -> bool:
    """Return ``True`` if ONNX Runtime exposes the CUDA execution provider (cached)."""
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    return "CUDAExecutionProvider" in ort.get_available_providers()


def _load_cuda_dlls() -> None:
    """Make cuDNN/cuBLAS loadable for ONNX Runtime's CUDA provider.

    Without them a GPU session crashes the process ("Cannot load symbol
    cudnnCreate") instead of falling back.  Importing torch first loads the
    CUDA libraries it bundles, which also measured fastest for OCR;
    ``onnxruntime.preload_dlls`` covers installs where torch is missing.
    """
    with contextlib.suppress(Exception):
        import torch  # noqa: F401
    try:
        import onnxruntime as ort

        preload_dlls = getattr(ort, "preload_dlls", None)
        if preload_dlls is not None:
            preload_dlls()
    except Exception:  # noqa: BLE001
        _log.debug("Could not preload ONNX Runtime CUDA DLLs", exc_info=True)


class OcrEngine:
    """Run RapidOCR on images.

    Args:
        accelerator: ``"gpu"`` to use ONNX Runtime CUDA (falls back to CPU when
            CUDA is unavailable) or ``"cpu"``.
        min_score: Drop recognised lines below this confidence.
        model_dir: Optional directory for RapidOCR models.  ``None`` (default)
            uses the models bundled with the package; a custom directory makes
            RapidOCR download them there on first use.
    """

    def __init__(
        self,
        accelerator: str = "gpu",
        min_score: float = DEFAULT_MIN_SCORE,
        model_dir: Path | None = None,
    ) -> None:
        if accelerator not in ("gpu", "cpu"):
            raise ValueError("accelerator must be 'gpu' or 'cpu'")
        self.min_score = float(min_score)
        self.requested_accelerator = accelerator
        self.active_accelerator = "gpu" if accelerator == "gpu" and cuda_available() else "cpu"
        if accelerator == "gpu" and self.active_accelerator != "gpu":
            _log.warning("CUDAExecutionProvider not available; running OCR on CPU.")

        try:
            from rapidocr import RapidOCR
        except ImportError as exc:  # pragma: no cover - depends on install
            raise RuntimeError(
                "RapidOCR is not installed. Install it with `pip install rapidocr`."
            ) from exc

        params: dict[str, Any] = {
            "Global.log_level": "warning",
            # Keep low-confidence lines here; we filter with our own threshold
            # so the stored mean_score reflects what was kept.
            "Global.text_score": min(0.5, self.min_score),
        }
        if model_dir is not None:
            model_dir.mkdir(parents=True, exist_ok=True)
            params["Global.model_root_dir"] = str(model_dir)
        if self.active_accelerator == "gpu":
            _load_cuda_dlls()
            params["EngineConfig.onnxruntime.use_cuda"] = True
            # Every photo has a different detector input shape, so EXHAUSTIVE
            # search would re-run per shape.  HEURISTIC is no option either: on
            # Turing (RTX 20xx) cuDNN's heuristic query fails for the
            # recogniser's asymmetric-padding conv and ORT silently moves that
            # model to the CPU.  DEFAULT keeps all three models on CUDA.
            params["EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search"] = "DEFAULT"
        self._engine = RapidOCR(params=params)
        self.engine_id = self._describe()

    def _describe(self) -> str:
        try:
            from importlib.metadata import version

            ver = version("rapidocr")
        except Exception:  # noqa: BLE001
            ver = "unknown"
        return f"rapidocr-{ver}"

    def recognize(self, img_bgr: Any) -> OcrResult:
        """Run OCR on a BGR ``uint8`` image of shape ``(H, W, 3)``."""
        output = self._engine(img_bgr)
        txts = getattr(output, "txts", None) or ()
        scores = getattr(output, "scores", None) or ()
        lines: list[OcrLine] = []
        for text, score in zip(txts, scores, strict=False):
            text = (text or "").strip()
            if not text or float(score) < self.min_score:
                continue
            lines.append(OcrLine(text=text, score=float(score)))
        return OcrResult(lines=lines)


def load_image_bgr(path: Path, max_side: int | None = None) -> Any | None:
    """Load an image file as a BGR ``numpy`` array, honouring EXIF rotation.

    Uses Pillow (with pillow-heif for HEIC when installed).  Returns ``None``
    if the file cannot be decoded (e.g. a video).
    """
    import numpy as np
    from PIL import Image, ImageOps

    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:
        pass

    try:
        with Image.open(path) as img:
            if max_side is not None:
                img.draft("RGB", (max_side, max_side))
            img = ImageOps.exif_transpose(img) or img
            img = img.convert("RGB")
            if max_side is not None and max(img.size) > max_side:
                img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            arr = np.asarray(img)
    except Exception:  # noqa: BLE001
        _log.debug("Could not decode %s for OCR", path, exc_info=True)
        return None
    return np.ascontiguousarray(arr[:, :, ::-1])
