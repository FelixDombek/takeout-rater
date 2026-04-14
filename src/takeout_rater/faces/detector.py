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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_SUPPORTED_PACKS = ("buffalo_l", "buffalo_sc")

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
    """

    def __init__(
        self,
        model_pack: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
    ) -> None:
        if model_pack not in _SUPPORTED_PACKS:
            raise ValueError(
                f"Unsupported model pack {model_pack!r}. Supported: {', '.join(_SUPPORTED_PACKS)}"
            )
        self._model_pack = model_pack
        self._det_size = det_size
        self._det_thresh = det_thresh
        self._app: object | None = None

    @property
    def model_pack(self) -> str:
        return self._model_pack

    def _ensure_loaded(self) -> object:
        """Lazily initialise the InsightFace FaceAnalysis app."""
        if self._app is not None:
            return self._app

        from insightface.app import FaceAnalysis  # noqa: PLC0415

        app = FaceAnalysis(
            name=self._model_pack,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=self._det_size, det_thresh=self._det_thresh)
        self._app = app
        logger.info(
            "InsightFace loaded: pack=%s  det_size=%s  det_thresh=%.2f",
            self._model_pack,
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
        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

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
        import numpy as np  # noqa: PLC0415

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
