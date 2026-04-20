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
from typing import TYPE_CHECKING, Any

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
        recognition_batching: Whether to batch recognition for all faces in one
            image.  Defaults to enabled for CUDA and disabled for TensorRT,
            because variable recognition batch shapes can force expensive
            TensorRT engine/profile churn.
    """

    def __init__(
        self,
        model_pack: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
        accelerator: str = "gpu",
        trt_cache_dir: Path | None = None,
        recognition_batching: bool | None = None,
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
        self._recognition_batching = (
            accelerator != "tensorrt" if recognition_batching is None else recognition_batching
        )
        self._app: object | None = None
        self._rec_model: object | None = None

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
        app_models = getattr(app, "models", None)
        if isinstance(app_models, dict):
            self._rec_model = app_models.get("recognition")
        logger.info(
            "InsightFace loaded: pack=%s  accelerator=%s  det_size=%s  det_thresh=%.2f",
            self._model_pack,
            self._accelerator,
            self._det_size,
            self._det_thresh,
        )
        return self._app

    def active_providers(self) -> list[str]:
        """Return ONNX Runtime providers currently used by loaded InsightFace models."""
        app = self._ensure_loaded()
        providers: list[str] = []
        seen: set[str] = set()

        for model in self._iter_insightface_models(app):
            session = getattr(model, "session", None)
            get_providers = getattr(session, "get_providers", None)
            if get_providers is None:
                continue
            try:
                model_providers = get_providers()
            except Exception:  # noqa: BLE001
                logger.debug("Could not inspect InsightFace ONNX providers", exc_info=True)
                continue
            for provider in model_providers:
                if provider not in seen:
                    seen.add(provider)
                    providers.append(provider)

        return providers

    @staticmethod
    def _iter_insightface_models(app: object) -> list[Any]:
        """Best-effort extraction of model objects from an InsightFace app."""
        models: list[Any] = []
        app_models = getattr(app, "models", None)
        if isinstance(app_models, dict):
            models.extend(app_models.values())
        elif isinstance(app_models, (list, tuple)):
            models.extend(app_models)

        det_model = getattr(app, "det_model", None)
        if det_model is not None and all(model is not det_model for model in models):
            models.append(det_model)
        return models

    def verify_tensorrt(self, probe_image: Path | None = None) -> list[str]:
        """Load TensorRT models, optionally run one inference, and verify TRT is active.

        TensorRT engine files are usually created during model preparation or the
        first inference.  Running a probe image here keeps backend validation out
        of the main face-detection loop, so a TensorRT failure can fall back to a
        clean CUDA-only run before any results are written.
        """
        if self._accelerator != "tensorrt":
            raise RuntimeError("TensorRT verification requested for a non-TensorRT detector.")

        self._ensure_loaded()
        if probe_image is not None:
            self.detect(probe_image)

        active_providers = self.active_providers()
        if active_providers and "TensorrtExecutionProvider" not in active_providers:
            raise RuntimeError(
                "TensorRT provider was requested but is not active. "
                f"Active providers: {active_providers}"
            )
        return active_providers

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

        self._ensure_loaded()

        img = cv2.imread(str(image_path))
        if img is None:
            logger.debug("Could not read image: %s", image_path)
            return []

        return (
            self.detect_batched(img)
            if self._recognition_batching
            else self._detect_via_face_analysis(img)
        )

    def detect_from_array(self, img_array: object) -> list[DetectedFace]:
        """Detect faces from a numpy BGR array (e.g. already-loaded thumbnail).

        Args:
            img_array: NumPy array in BGR format (as returned by cv2.imread).

        Returns:
            List of :class:`DetectedFace` objects.
        """
        return (
            self.detect_batched(img_array)
            if self._recognition_batching
            else self._detect_via_face_analysis(img_array)
        )

    def _detect_via_face_analysis(self, img_array: object) -> list[DetectedFace]:
        """Fallback implementation using InsightFace's standard single-image API."""
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

    def detect_batched(self, img_array: object) -> list[DetectedFace]:
        """Detect faces and batch recognition for all faces in one image.

        InsightFace's public ``FaceAnalysis.get()`` recognises each detected
        face one at a time.  The recognition model supports a list of aligned
        crops, so this method keeps detection per image but runs ArcFace
        embedding extraction for all faces in that image as one ONNX batch.
        """
        import cv2
        import numpy as np
        from insightface.utils import face_align

        app = self._ensure_loaded()
        rec_model = self._rec_model
        det_model = getattr(app, "det_model", None)
        if rec_model is None or det_model is None:
            return self._detect_via_face_analysis(img_array)

        bboxes, kpss = det_model.detect(img_array, max_num=0, metric="default")
        if bboxes.shape[0] == 0:
            return []
        if kpss is None:
            return self._detect_via_face_analysis(img_array)

        input_size = getattr(rec_model, "input_size", (112, 112))
        image_size = int(input_size[0])

        aligned_faces = []
        aligned_indices: list[int] = []
        for idx in range(bboxes.shape[0]):
            kps = kpss[idx]
            try:
                transform = face_align.estimate_norm(kps, image_size=image_size)
            except Exception:  # noqa: BLE001
                logger.debug("Could not align detected face %s", idx, exc_info=True)
                continue
            aligned = cv2.warpAffine(
                img_array,
                transform,
                tuple(input_size),
                borderMode=cv2.BORDER_CONSTANT,
            )
            aligned_faces.append(aligned)
            aligned_indices.append(idx)

        if not aligned_faces:
            return []

        try:
            embeddings = rec_model.get_feat(aligned_faces)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            logger.debug("Batched face recognition failed; falling back", exc_info=True)
            return self._detect_via_face_analysis(img_array)

        results: list[DetectedFace] = []
        for emb_idx, face_idx in enumerate(aligned_indices):
            embedding = embeddings[emb_idx]
            norm = float(np.linalg.norm(embedding))
            if norm <= 1e-9:
                continue
            embedding = (embedding / norm).tolist()
            bbox = bboxes[face_idx, 0:4].astype(float).tolist()
            results.append(
                DetectedFace(
                    bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    det_score=float(bboxes[face_idx, 4]),
                    embedding=embedding,
                    face_index=face_idx,
                )
            )

        return results
