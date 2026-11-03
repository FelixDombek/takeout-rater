"""Face quality scorer: eyes open, smile, facing the camera, sharpness, cut-off.

Rates the *people* in a photo rather than the photo as a whole, for picking the
best frame of a burst and for filtering and sorting portraits and group shots.

Per face it runs, on the thumbnail:

- InsightFace ``buffalo_l`` SCRFD detection (bounding box, 5 keypoints),
- the 106-point 2D landmark model (eye opening, in-frame check),
- the 68-point 3D landmark model (head pose: yaw / pitch),
- the FER+ expression model (smile = "happiness" probability),

and derives 0–1 dimensions with :mod:`takeout_rater.faces.quality`.  Faces
far smaller than the largest face are treated as background and ignored.

Per-photo metrics (all 0–1 except ``face_count``):

- ``face_quality`` — mean and worst face quality averaged.
- ``face_quality_worst`` — the worst face; one blink sinks a group shot.
- ``eyes_open`` — worst face.
- ``smile`` / ``facing_camera`` / ``face_sharpness`` — mean over faces.
- ``face_visible`` — worst face; low when a face is cut off by the frame edge
  or partly hidden.
- ``face_count`` — number of faces rated.  Photos without faces store only
  ``face_count = 0``, so they are not re-analysed and drop out of
  face-metric filters.

Models: InsightFace ``buffalo_l`` (already used by face detection, cached in
``~/.insightface``) and FER+ ``emotion-ferplus-8.onnx`` (35 MB, MIT, from the
ONNX model zoo; downloaded on first use via the HuggingFace Hub cache).  All
inference runs through ONNX Runtime on CUDA when available, else CPU.
"""

from __future__ import annotations

import importlib.util
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from takeout_rater.faces.quality import (
    PHOTO_METRIC_KEYS,
    FaceQuality,
    aggregate_faces,
    build_face_quality,
    sharpness_raw,
)
from takeout_rater.scoring.scorers.base import (
    BaseScorer,
    MetricSpec,
    ScorerSpec,
    VariantSpec,
)

logger = logging.getLogger(__name__)

#: InsightFace model pack (detection + landmark models).
_MODEL_PACK = "buffalo_l"
#: InsightFace modules needed; recognition and gender/age are skipped.
_ALLOWED_MODULES = ["detection", "landmark_2d_106", "landmark_3d_68"]
#: Detector input size; thumbnails are at most 512 px so 640 covers them.
_DET_SIZE = (640, 640)
_DET_THRESH = 0.5

#: FER+ expression model (ONNX model zoo, MIT licence).
_FER_HF_REPO = "onnxmodelzoo/emotion-ferplus-8"
_FER_FILENAME = "emotion-ferplus-8.onnx"
_FER_FALLBACK_URL = (
    "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/"
    "body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
)
_FER_SHA256 = "a2a2ba6a335a3b29c21acb6272f962bd3d47f84952aaffa03b60986e04efa61c"
#: FER+ input side (grayscale).
_FER_INPUT_PX = 64

#: Thumbnail decode threads.
_DECODE_WORKERS = 4


_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        key="face_quality",
        display_name="Face quality",
        description=(
            "Overall rating of the people in the photo (0–1): eyes open, facing the "
            "camera, sharp, fully in frame, smiling. Average of the mean face and the "
            "worst face, so one bad face pulls a group shot down."
        ),
    ),
    MetricSpec(
        key="face_quality_worst",
        display_name="Worst face",
        description="Quality of the worst-rated face in the photo (0–1).",
    ),
    MetricSpec(
        key="eyes_open",
        display_name="Eyes open",
        description=(
            "How open the eyes are, for the face with the most closed eyes (0 = a blink, "
            "1 = everyone's eyes open)."
        ),
    ),
    MetricSpec(
        key="smile",
        display_name="Smile",
        description="Average smile probability across faces (0–1).",
    ),
    MetricSpec(
        key="facing_camera",
        display_name="Facing camera",
        description=(
            "How directly the faces point at the camera, from head pose (1 = straight "
            "on, ~0.4 at a 40° head turn)."
        ),
    ),
    MetricSpec(
        key="face_sharpness",
        display_name="Face sharpness",
        description="Average focus / motion-blur sharpness of the faces (0–1).",
    ),
    MetricSpec(
        key="face_visible",
        display_name="Face fully visible",
        description=(
            "Lowest across faces: 1 when the face is fully inside the frame and "
            "unobstructed, lower when cut off at the edge or partly hidden."
        ),
    ),
    MetricSpec(
        key="face_count",
        display_name="Faces rated",
        description=(
            "Number of faces rated (tiny background faces are ignored). 0 for photos without faces."
        ),
        min_value=0.0,
        max_value=20.0,
    ),
)

assert tuple(m.key for m in _METRICS) == PHOTO_METRIC_KEYS


def _onnx_providers() -> list[str]:
    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except Exception:  # noqa: BLE001
        return ["CPUExecutionProvider"]
    providers = [p for p in ("CUDAExecutionProvider",) if p in available]
    return [*providers, "CPUExecutionProvider"]


def _decode(path: Path) -> Any:
    """Decode an image as a BGR array (Unicode-path safe), or ``None``."""
    import cv2
    import numpy as np

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


class FaceQualityScorer(BaseScorer):
    """Per-face eyes / smile / pose / sharpness / visibility, aggregated per photo."""

    def __init__(self, variant_id: str | None = None, **kwargs: Any) -> None:
        super().__init__(variant_id=variant_id, **kwargs)
        self._app: Any = None
        self._fer: Any = None
        self._fer_input: str = ""

    @classmethod
    def spec(cls) -> ScorerSpec:
        return ScorerSpec(
            scorer_id="face_quality",
            display_name="Face Quality",
            description=(
                "Rates the people in each photo: are the eyes open, are they smiling, "
                "facing the camera, in focus, and fully in the frame. Group shots are "
                "judged by their worst face too, so one blink sinks the frame — handy "
                "for picking the best shot of a burst."
            ),
            technical_description=(
                "InsightFace buffalo_l SCRFD detection + 2d106det landmarks (eye aspect "
                "ratio) + 1k3d68 (head yaw/pitch); FER+ (emotion-ferplus-8) happiness "
                "probability for smiles; contrast-normalised Laplacian variance on a "
                "48 px aligned face crop for sharpness; in-frame landmark/box fraction "
                "and detection score for cut-off/occlusion. Faces < 35 % of the largest "
                "face's height are ignored. Per photo: worst face for eyes and "
                "visibility, mean for the rest."
            ),
            version="1",
            variants=(
                VariantSpec(
                    variant_id="buffalo_l",
                    display_name="InsightFace buffalo_l + FER+",
                    description=(
                        "InsightFace detection and landmarks with the FER+ expression model."
                    ),
                    primary_metric_key="face_count",
                    metrics=_METRICS,
                ),
            ),
            default_variant_id="buffalo_l",
            requires_extras=(),
        )

    @classmethod
    def is_available(cls) -> bool:
        return all(
            importlib.util.find_spec(mod) is not None
            for mod in ("insightface", "cv2", "onnxruntime", "numpy")
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @staticmethod
    def _download_fer_model() -> Path:
        """Return the local path of the FER+ ONNX model, downloading it once."""
        import hashlib

        try:
            from huggingface_hub import hf_hub_download

            return Path(hf_hub_download(repo_id=_FER_HF_REPO, filename=_FER_FILENAME))
        except Exception as exc:  # noqa: BLE001
            logger.warning("FER+ download from HuggingFace failed (%s); trying GitHub", exc)

        import urllib.request

        cache_dir = Path.home() / ".cache" / "takeout-rater" / "models"
        target = cache_dir / _FER_FILENAME
        if not target.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".part")
            urllib.request.urlretrieve(_FER_FALLBACK_URL, tmp)  # noqa: S310
            digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
            if digest != _FER_SHA256:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"FER+ model checksum mismatch ({digest})")
            tmp.replace(target)
        return target

    def _ensure_loaded(self) -> None:
        if self._app is not None:
            return
        from takeout_rater.faces.detector import FaceDetector

        FaceDetector._preload_onnxruntime_cuda_dlls()  # noqa: SLF001
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        providers = _onnx_providers()
        app = FaceAnalysis(name=_MODEL_PACK, allowed_modules=_ALLOWED_MODULES, providers=providers)
        app.prepare(
            ctx_id=0 if "CUDAExecutionProvider" in providers else -1,
            det_thresh=_DET_THRESH,
            det_size=_DET_SIZE,
        )
        fer = ort.InferenceSession(str(self._download_fer_model()), providers=providers)
        self._fer_input = fer.get_inputs()[0].name
        self._fer = fer
        self._app = app
        logger.info("Face quality scorer loaded (providers=%s)", providers)

    def close(self) -> None:
        """Release the ONNX sessions."""
        self._app = None
        self._fer = None

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _expression_probs(self, image_bgr: Any, kps: Any) -> list[float] | None:
        import cv2
        import numpy as np
        from insightface.utils import face_align

        try:
            matrix = face_align.estimate_norm(np.asarray(kps, dtype=np.float32), image_size=112)
            matrix = matrix * (_FER_INPUT_PX / 112.0)
            crop = cv2.warpAffine(
                image_bgr,
                matrix,
                (_FER_INPUT_PX, _FER_INPUT_PX),
                flags=cv2.INTER_AREA,
                borderMode=cv2.BORDER_REPLICATE,
            )
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)[None, None]
            logits = self._fer.run(None, {self._fer_input: gray})[0][0]
        except Exception:  # noqa: BLE001
            logger.debug("FER+ inference failed", exc_info=True)
            return None
        exp = np.exp(logits - logits.max())
        return (exp / exp.sum()).tolist()

    def analyze_image(self, image_bgr: Any) -> list[FaceQuality]:
        """Detect and rate every face in a BGR image."""
        self._ensure_loaded()
        height, width = image_bgr.shape[:2]
        results: list[FaceQuality] = []
        for face in self._app.get(image_bgr):
            landmarks = getattr(face, "landmark_2d_106", None)
            kps = getattr(face, "kps", None)
            if landmarks is None or kps is None:
                continue
            try:
                sharp = sharpness_raw(image_bgr, kps)
            except Exception:  # noqa: BLE001
                logger.debug("Sharpness measurement failed", exc_info=True)
                sharp = 0.0
            results.append(
                build_face_quality(
                    bbox=face.bbox.tolist(),
                    det_score=float(face.det_score),
                    landmarks_106=landmarks,
                    pose=getattr(face, "pose", None),
                    expression_probs=self._expression_probs(image_bgr, kps),
                    sharpness_raw_value=sharp,
                    width=width,
                    height=height,
                )
            )
        return results

    def score_batch(
        self,
        image_paths: list[Path],
        *,
        variant_id: str | None = None,
    ) -> list[dict[str, float]]:
        """Score each image; undecodable images return ``{}`` and are retried later."""
        if not image_paths:
            return []
        self._ensure_loaded()

        results: list[dict[str, float]] = []
        with ThreadPoolExecutor(max_workers=_DECODE_WORKERS) as pool:
            for path, image in zip(image_paths, pool.map(_decode, image_paths), strict=True):
                if image is None:
                    logger.debug("Could not decode %s", path)
                    results.append({})
                    continue
                try:
                    faces = self.analyze_image(image)
                except Exception:  # noqa: BLE001
                    logger.warning("Face quality analysis failed for %s", path, exc_info=True)
                    results.append({})
                    continue
                results.append(aggregate_faces(faces))
        return results
