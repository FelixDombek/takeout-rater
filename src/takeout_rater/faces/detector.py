"""InsightFace wrapper for face detection and 512-d ArcFace embedding extraction.

Loads the InsightFace ``buffalo_l`` (default) or ``buffalo_sc`` model pack
via ONNX Runtime.  All inference is local — no images or embeddings leave the
user's machine.

Usage::

    from takeout_rater.faces.detector import FaceDetector

    detector = FaceDetector(model_pack="buffalo_l")
    faces = detector.detect(image_path)
    # faces: list[DetectedFace]
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_SUPPORTED_PACKS = ("buffalo_l", "buffalo_sc")
_SUPPORTED_ACCELERATORS = ("gpu", "tensorrt")

# Embedding dimensionality for ArcFace models in the buffalo packs.
EMBEDDING_DIM = 512


@dataclass
class DetectedFace:
    """A single detected face with bounding box and embedding."""

    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    det_score: float  # detection confidence [0, 1]
    embedding: list[float]  # 512-d ArcFace L2-normalised embedding
    face_index: int  # 0-based index within the image


class FaceDetector:
    """InsightFace face detector and recogniser.

    Args:
        model_pack: InsightFace model pack name.  ``"buffalo_l"`` (default,
            best accuracy, ~350 MB download) or ``"buffalo_sc"`` (smaller,
            faster).
        det_size: Detection input size as ``(width, height)``.  Larger values
            find smaller faces at the cost of speed.
        det_thresh: Minimum detection confidence threshold.
        accelerator: ONNX Runtime GPU backend. ``"gpu"`` uses CUDA directly;
            ``"tensorrt"`` tries TensorRT first, with CUDA and CPU fallbacks.
        trt_cache_dir: Optional directory for TensorRT engine cache files.
    """

    def __init__(
        self,
        model_pack: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
        accelerator: str = "gpu",
        trt_cache_dir: Path | None = None,
    ) -> None:
        if model_pack not in _SUPPORTED_PACKS:
            raise ValueError(
                f"Unsupported model pack {model_pack!r}. Supported: {', '.join(_SUPPORTED_PACKS)}"
            )
        if accelerator not in _SUPPORTED_ACCELERATORS:
            raise ValueError(
                f"Unsupported accelerator {accelerator!r}. "
                f"Supported: {', '.join(_SUPPORTED_ACCELERATORS)}"
            )
        self._model_pack = model_pack
        self._det_size = det_size
        self._det_thresh = det_thresh
        self._accelerator = accelerator
        self._trt_cache_dir = trt_cache_dir
        self._app: object | None = None

    @property
    def model_pack(self) -> str:
        return self._model_pack

    @property
    def accelerator(self) -> str:
        return self._accelerator

    def _providers(self) -> list[object]:
        """Return ONNX Runtime providers in the requested priority order."""
        providers: list[object] = []
        if self._accelerator == "tensorrt":
            options: dict[str, object] = {
                "device_id": 0,
                "trt_fp16_enable": True,
            }
            cache_dir = self._trt_cache_dir or (
                Path(tempfile.gettempdir()) / "takeout-rater" / "onnxruntime-trt"
            )
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning(
                    "TensorRT engine cache disabled; cannot create %s: %s", cache_dir, exc
                )
            else:
                options.update(
                    {
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": str(cache_dir),
                    }
                )
            providers.append(
                (
                    "TensorrtExecutionProvider",
                    options,
                )
            )
        providers.extend(["CUDAExecutionProvider", "CPUExecutionProvider"])
        return providers

    @staticmethod
    def _preload_onnxruntime_cuda_dlls() -> None:
        """Let ONNX Runtime find CUDA/cuDNN DLLs shipped by GPU packages."""
        try:
            import onnxruntime as ort
        except Exception:  # noqa: BLE001
            return

        preload_dlls = getattr(ort, "preload_dlls", None)
        if preload_dlls is None:
            return
        try:
            preload_dlls()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not preload ONNX Runtime CUDA DLLs: %s", exc)

    def _ensure_loaded(self) -> object:
        """Lazily initialise the InsightFace FaceAnalysis app."""
        if self._app is not None:
            return self._app

        self._preload_onnxruntime_cuda_dlls()
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name=self._model_pack,
            providers=self._providers(),
        )
        app.prepare(ctx_id=0, det_size=self._det_size, det_thresh=self._det_thresh)
        self._app = app
        logger.info(
            "InsightFace loaded: pack=%s  accelerator=%s  det_size=%s  det_thresh=%.2f",
            self._model_pack,
            self._accelerator,
            self._det_size,
            self._det_thresh,
        )
        return self._app

    def detect(self, image_path: Path) -> list[DetectedFace]:
        """Detect faces in a single image and return embeddings.

        Args:
            image_path: Path to the image file (JPEG, PNG, etc.).

        Returns:
            List of :class:`DetectedFace` objects, one per detected face.
            Returns an empty list if no faces are found or the image cannot
            be read.
        """
        import cv2
        import numpy as np

        app = self._ensure_loaded()

        img = cv2.imread(str(image_path))
        if img is None:
            logger.debug("Could not read image: %s", image_path)
            return []

        faces_raw = app.get(img)
        results: list[DetectedFace] = []

        for idx, face in enumerate(faces_raw):
            bbox = face.bbox.astype(float).tolist()
            det_score = float(face.det_score)
            embedding = face.normed_embedding
            if embedding is None:
                continue
            # Ensure L2 normalisation
            norm = float(np.linalg.norm(embedding))
            if norm > 1e-9:
                embedding = (embedding / norm).tolist()
            else:
                continue

            results.append(
                DetectedFace(
                    bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    det_score=det_score,
                    embedding=embedding,
                    face_index=idx,
                )
            )

        return results

    def detect_from_array(self, img_array: object) -> list[DetectedFace]:
        """Detect faces from a numpy BGR array (e.g. already-loaded thumbnail).

        Args:
            img_array: NumPy array in BGR format (as returned by cv2.imread).

        Returns:
            List of :class:`DetectedFace` objects.
        """
        import numpy as np

        app = self._ensure_loaded()

        faces_raw = app.get(img_array)
        results: list[DetectedFace] = []

        for idx, face in enumerate(faces_raw):
            bbox = face.bbox.astype(float).tolist()
            det_score = float(face.det_score)
            embedding = face.normed_embedding
            if embedding is None:
                continue
            norm = float(np.linalg.norm(embedding))
            if norm > 1e-9:
                embedding = (embedding / norm).tolist()
            else:
                continue

            results.append(
                DetectedFace(
                    bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    det_score=det_score,
                    embedding=embedding,
                    face_index=idx,
                )
            )

        return results
