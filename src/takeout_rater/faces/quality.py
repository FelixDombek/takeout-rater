"""Per-face quality metrics and per-photo aggregation.

This module turns the raw output of InsightFace (bounding box, detection
score, 106-point 2D landmarks, 68-point 3D landmarks with head pose) and of the
FER+ expression model into a handful of 0–1 *face quality* dimensions:

``eyes_open``
    Eye aspect ratio (EAR) from the 106-point landmarks: the mean height of the
    eye opening divided by its width.  The 106-point model follows blinks
    closely (ROC AUC ≈ 0.99 on a labelled open/closed eyes set, also at
    ~40 px face width); the 3D 68-point model regresses towards an open-eyed
    mean shape and is not used for this.  The worse of the two eyes counts,
    unless the head is turned so far that one eye is hidden.
``smile``
    Probability of the FER+ "happiness" class (ROC AUC ≈ 0.98 on CelebA's
    *Smiling* attribute, versus ≈ 0.92 for mouth-geometry features).
``facing_camera``
    Head pose (yaw / pitch from the 3D landmarks): 1 when the face points at
    the camera, falling off smoothly as the head turns away.  This is head
    orientation, not eye gaze.
``sharpness``
    Contrast-normalised Laplacian variance of the face on an aligned crop of
    at most 48×48 px (smaller faces at about their own size), so the value
    depends little on how large the face is in the frame.
``visible``
    How much of the face (brows to chin) is inside the frame, with a mild
    penalty for a low detection score, which is what partly hidden faces
    (hands, hair, another head) typically produce.

Known limits, checked on a real library: eyes looking down (phone, mirror
selfie) narrow the lid gap and rate as half-open; eyes behind sunglasses are
not seen and count as open; masks are not detected as such; statues,
paintings and posters are rated like faces.

``quality`` combines them per face; closed eyes multiply the quality down so
that a blink cannot be made up for by a smile.  :func:`aggregate_faces` then
summarises all *subject* faces of a photo (tiny background faces are ignored)
into per-photo metrics, including the worst face so that one blink in a group
shot sinks the frame.

Everything here is plain NumPy (plus OpenCV for the sharpness crop) so it can
be unit-tested without the models.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Landmark indices (InsightFace 2d106det)
# ---------------------------------------------------------------------------

#: Eye contour indices in the 106-point layout as
#: ``((upper, lower), (upper, lower), (upper, lower), (corner, corner))``.
#: "Left"/"right" refer to the image, not to the person.
EYE_106_IMAGE_LEFT: tuple[tuple[int, int], ...] = ((41, 36), (40, 33), (42, 37), (35, 39))
EYE_106_IMAGE_RIGHT: tuple[tuple[int, int], ...] = ((95, 90), (94, 87), (96, 91), (93, 89))

# ---------------------------------------------------------------------------
# Calibration constants
# ---------------------------------------------------------------------------

#: EAR at or below which an eye counts as fully closed.  Blinks measure
#: ≈ 0.06–0.10; open eyes ≈ 0.20–0.35 (5th percentile of open, smiling
#: CelebA faces ≈ 0.20).  Eyes looking down (phone, mirror selfie) narrow the
#: lid gap to ≈ 0.13–0.16, so the ramp is kept low to rate those half-open
#: rather than closed.
EAR_CLOSED: float = 0.10
#: EAR at or above which an eye counts as fully open.
EAR_OPEN: float = 0.18
#: A broad smile squints the eyes; at smile = 1 both thresholds drop by this.
EAR_SMILE_RELIEF: float = 0.03

#: Yaw beyond which only the eye nearer to the camera is used.
YAW_ONE_EYE_DEG: float = 40.0

#: InsightFace's 3D-landmark pitch reads a few degrees negative for faces that
#: look straight into the camera; this offset re-centres it.
PITCH_OFFSET_DEG: float = -4.0
#: Gaussian fall-off widths of the facing-camera score.  Yaw 20° → ≈ 0.85,
#: 40° → ≈ 0.5, 60° → ≈ 0.23.
YAW_SIGMA_DEG: float = 35.0
PITCH_SIGMA_DEG: float = 25.0

#: Largest side of the aligned crop the sharpness is measured on.  Smaller
#: faces use a crop of about their own size (down to
#: :data:`SHARPNESS_MIN_CROP_PX`) so that upsampling a small, but sharp, face
#: does not make it look blurry.
SHARPNESS_CROP_PX: int = 48
SHARPNESS_MIN_CROP_PX: int = 24
#: Raw sharpness values mapped to 0 and 1 (log scale in between).  Sharp faces
#: measure ≈ 0.6–1.5, a blur of ~5 % of the face width ≈ 0.15, heavy motion or
#: focus blur ≤ 0.05.
SHARPNESS_RAW_LOW: float = 0.05
SHARPNESS_RAW_HIGH: float = 0.8

#: In-frame fraction (share of the 106 landmarks inside the image) mapped to
#: 0 (and 1 at 1.0).  Faces cut in half at the frame edge measure ≈ 0.6.
IN_FRAME_LOW: float = 0.6
#: Detection scores mapped to the occlusion factor 0 and 1.  Clear faces score
#: ≈ 0.8–0.9; partly hidden ones drop towards 0.6.
DET_SCORE_LOW: float = 0.55
DET_SCORE_HIGH: float = 0.8

#: Faces smaller than this fraction of the largest face's height are treated
#: as background faces and ignored.
SUBJECT_RELATIVE_SIZE: float = 0.35
#: Faces smaller than this (in pixels of the analysed image) are ignored —
#: landmarks on them are too unreliable to rate.
MIN_FACE_PX: float = 20.0
#: Faces detected with less confidence are ignored.  Real faces score
#: ≈ 0.75–0.9; pets, food, flowers and patterns mistaken for faces mostly
#: score lower.
MIN_SUBJECT_DET_SCORE: float = 0.6

#: Weights of the dimensions that make up the base of a face's quality.
QUALITY_WEIGHTS: dict[str, float] = {
    "facing_camera": 0.35,
    "sharpness": 0.30,
    "visible": 0.20,
    "smile": 0.10,
}
#: Quality multiplier for fully closed eyes (fully open = 1).
CLOSED_EYES_FACTOR: float = 0.3


def _clip01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


def _ramp(value: float, low: float, high: float) -> float:
    """Linear 0→1 ramp between *low* and *high*, clipped."""
    return _clip01((value - low) / (high - low))


# ---------------------------------------------------------------------------
# Individual dimensions
# ---------------------------------------------------------------------------


def eye_aspect_ratio(landmarks_106: np.ndarray, eye: Sequence[tuple[int, int]]) -> float:
    """Return the eye aspect ratio of one eye from 106-point landmarks.

    EAR = mean vertical lid distance / horizontal corner distance.
    """
    pts = np.asarray(landmarks_106, dtype=np.float64)
    vertical = [float(np.linalg.norm(pts[a] - pts[b])) for a, b in eye[:3]]
    c1, c2 = eye[3]
    horizontal = float(np.linalg.norm(pts[c1] - pts[c2]))
    if horizontal <= 1e-6:
        return 0.0
    return float(np.mean(vertical)) / horizontal


def eye_open_score(ear: float, smile: float = 0.0) -> float:
    """Map an eye aspect ratio to 0 (closed) … 1 (open).

    A broad *smile* lowers the thresholds by up to :data:`EAR_SMILE_RELIEF`,
    because laughing narrows the eyes without closing them.
    """
    relief = EAR_SMILE_RELIEF * _clip01(smile)
    return _ramp(ear, EAR_CLOSED - relief, EAR_OPEN - relief)


def eyes_open_score(
    landmarks_106: np.ndarray, yaw_deg: float = 0.0, smile: float = 0.0
) -> tuple[float, float, float]:
    """Return ``(score, ear_image_left, ear_image_right)`` for a face.

    The score is the *worse* eye, so a wink or a half blink still counts.
    When the head is turned by more than :data:`YAW_ONE_EYE_DEG` the far eye is
    foreshortened or hidden and its landmarks are guesses, so the better eye is
    used instead.
    """
    ear_l = eye_aspect_ratio(landmarks_106, EYE_106_IMAGE_LEFT)
    ear_r = eye_aspect_ratio(landmarks_106, EYE_106_IMAGE_RIGHT)
    s_l, s_r = eye_open_score(ear_l, smile), eye_open_score(ear_r, smile)
    score = max(s_l, s_r) if abs(yaw_deg) > YAW_ONE_EYE_DEG else min(s_l, s_r)
    return score, ear_l, ear_r


def facing_camera_score(pitch_deg: float, yaw_deg: float) -> float:
    """Return 1 for a face pointing at the camera, falling off with head turn."""
    pitch = pitch_deg - PITCH_OFFSET_DEG
    exponent = (yaw_deg / YAW_SIGMA_DEG) ** 2 + (pitch / PITCH_SIGMA_DEG) ** 2
    return _clip01(math.exp(-0.5 * exponent))


def smile_score(expression_probs: Sequence[float] | None) -> float:
    """Return the FER+ happiness probability (class index 1), or 0 when unknown."""
    if expression_probs is None or len(expression_probs) < 2:
        return 0.0
    return _clip01(float(expression_probs[1]))


def sharpness_raw(image_bgr: np.ndarray, kps: np.ndarray) -> float:
    """Return a size-independent raw sharpness value for one face.

    The face is aligned with the 5 detection keypoints onto a
    :data:`SHARPNESS_CROP_PX` square (area-downsampled for large faces); the
    Laplacian variance of the central part is divided by the intensity variance
    so that low-contrast faces are not penalised for being dim.
    """
    import cv2
    from insightface.utils import face_align

    kps_arr = np.asarray(kps, dtype=np.float32)
    eye_dist = float(np.linalg.norm(kps_arr[1] - kps_arr[0]))
    # The 112 px ArcFace template has ~38 px between the eyes, ~0.75 of a face.
    native = eye_dist / 38.0 * 112.0 * 0.75
    size = int(np.clip(round(native), SHARPNESS_MIN_CROP_PX, SHARPNESS_CROP_PX))
    matrix = face_align.estimate_norm(kps_arr, image_size=112)
    matrix = matrix * (size / 112.0)
    crop = cv2.warpAffine(image_bgr, matrix, (size, size), flags=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    margin = size // 8
    centre = gray[margin : size - margin, margin : size - margin]
    lap = cv2.Laplacian(centre, cv2.CV_32F)
    return float(lap.var() / (centre.var() + 25.0))


def sharpness_score(raw: float) -> float:
    """Map :func:`sharpness_raw` output to 0 (blurry) … 1 (sharp) on a log scale."""
    if not math.isfinite(raw) or raw <= 0:
        return 0.0
    low, high = math.log(SHARPNESS_RAW_LOW), math.log(SHARPNESS_RAW_HIGH)
    return _ramp(math.log(raw), low, high)


def in_frame_fraction(
    bbox: Sequence[float],
    landmarks_106: np.ndarray | None,
    width: int,
    height: int,
) -> float:
    """Return the fraction of the face inside the image.

    The share of 106-point landmarks inside the frame; the detector
    extrapolates landmarks past the image edge for faces that are cut off.
    The landmarks span brows to chin, so a close-up whose forehead or hair
    touches the edge still counts as fully in frame.  Without landmarks the
    bounding-box area inside the frame is used.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox)
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if area <= 0:
        return 0.0
    inside_w = max(0.0, min(x2, width) - max(x1, 0.0))
    inside_h = max(0.0, min(y2, height) - max(y1, 0.0))
    box_frac = inside_w * inside_h / area
    if landmarks_106 is None:
        return _clip01(box_frac)
    pts = np.asarray(landmarks_106, dtype=np.float64)
    lm_frac = float(
        np.mean(
            (pts[:, 0] >= 0)
            & (pts[:, 0] <= width - 1)
            & (pts[:, 1] >= 0)
            & (pts[:, 1] <= height - 1)
        )
    )
    return _clip01(lm_frac)


def visibility_score(in_frame: float, det_score: float) -> float:
    """Combine in-frame fraction and detection confidence into 0 … 1."""
    frame = _ramp(in_frame, IN_FRAME_LOW, 1.0)
    unoccluded = _ramp(det_score, DET_SCORE_LOW, DET_SCORE_HIGH)
    return _clip01(frame * (0.6 + 0.4 * unoccluded))


def face_quality_score(
    eyes_open: float,
    smile: float,
    facing_camera: float,
    sharpness: float,
    visible: float,
) -> float:
    """Combine the dimensions into one 0 … 1 quality for a single face.

    A weighted mean of facing, sharpness, visibility and smile, multiplied by
    an eyes factor that drops to :data:`CLOSED_EYES_FACTOR` for closed eyes.
    """
    w = QUALITY_WEIGHTS
    base = (
        w["facing_camera"] * facing_camera
        + w["sharpness"] * sharpness
        + w["visible"] * visible
        + w["smile"] * smile
    ) / sum(w.values())
    eyes_factor = CLOSED_EYES_FACTOR + (1.0 - CLOSED_EYES_FACTOR) * eyes_open
    return _clip01(base * eyes_factor)


# ---------------------------------------------------------------------------
# Per-face record and per-photo aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaceQuality:
    """All quality dimensions of one detected face (each 0 … 1 unless noted)."""

    bbox: tuple[float, float, float, float]
    det_score: float
    eyes_open: float
    smile: float
    facing_camera: float
    sharpness: float
    visible: float
    quality: float
    #: Raw values kept for debugging and calibration.
    ear_left: float = 0.0
    ear_right: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    sharpness_raw: float = 0.0
    in_frame: float = 1.0

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])


def build_face_quality(
    *,
    bbox: Sequence[float],
    det_score: float,
    landmarks_106: np.ndarray,
    pose: Sequence[float] | None,
    expression_probs: Sequence[float] | None,
    sharpness_raw_value: float,
    width: int,
    height: int,
) -> FaceQuality:
    """Compute every dimension of one face from model outputs.

    Args:
        bbox: ``(x1, y1, x2, y2)`` in image pixels.
        det_score: Detection confidence.
        landmarks_106: ``(106, 2)`` landmark array.
        pose: InsightFace ``face.pose``: ``(pitch, yaw, roll)`` in degrees, or
            ``None`` when the 3D landmark model did not run.
        expression_probs: FER+ class probabilities (8 values) or ``None``.
        sharpness_raw_value: Output of :func:`sharpness_raw`.
        width, height: Image size in pixels.
    """
    pitch, yaw = (float(pose[0]), float(pose[1])) if pose is not None else (PITCH_OFFSET_DEG, 0.0)
    smile = smile_score(expression_probs)
    eyes, ear_l, ear_r = eyes_open_score(landmarks_106, yaw, smile)
    facing = facing_camera_score(pitch, yaw)
    sharp = sharpness_score(sharpness_raw_value)
    frame = in_frame_fraction(bbox, landmarks_106, width, height)
    visible = visibility_score(frame, det_score)
    return FaceQuality(
        bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
        det_score=float(det_score),
        eyes_open=eyes,
        smile=smile,
        facing_camera=facing,
        sharpness=sharp,
        visible=visible,
        quality=face_quality_score(eyes, smile, facing, sharp, visible),
        ear_left=ear_l,
        ear_right=ear_r,
        yaw=yaw,
        pitch=pitch,
        sharpness_raw=float(sharpness_raw_value),
        in_frame=frame,
    )


def select_subject_faces(faces: Sequence[FaceQuality]) -> list[FaceQuality]:
    """Drop tiny and background faces, keeping the photo's subjects.

    A face is a subject when it is at least :data:`MIN_FACE_PX` tall, was
    detected with at least :data:`MIN_SUBJECT_DET_SCORE` confidence and is at
    least :data:`SUBJECT_RELATIVE_SIZE` of the largest such face's height.
    """
    sized = [f for f in faces if f.height >= MIN_FACE_PX and f.det_score >= MIN_SUBJECT_DET_SCORE]
    if not sized:
        return []
    largest = max(f.height for f in sized)
    return [f for f in sized if f.height >= SUBJECT_RELATIVE_SIZE * largest]


#: Metric keys written per photo, in display order.
PHOTO_METRIC_KEYS: tuple[str, ...] = (
    "face_quality",
    "face_quality_worst",
    "eyes_open",
    "smile",
    "facing_camera",
    "face_sharpness",
    "face_visible",
    "face_count",
)


def aggregate_faces(faces: Sequence[FaceQuality]) -> dict[str, float]:
    """Summarise a photo's faces into per-photo metrics.

    Photos without subject faces get only ``face_count = 0`` so that they are
    not re-analysed on every run and drop out of face-metric filters.

    ``eyes_open`` and ``face_visible`` use the *worst* subject face (one blink
    or one cut-off head ruins a group shot); ``smile``, ``facing_camera`` and
    ``face_sharpness`` use the mean.  ``face_quality`` averages the mean and
    the worst per-face quality; ``face_quality_worst`` is the worst alone.
    """
    subjects = select_subject_faces(faces)
    if not subjects:
        return {"face_count": 0.0}
    qualities = [f.quality for f in subjects]
    worst = min(qualities)
    mean = float(np.mean(qualities))
    return {
        "face_quality": round(0.5 * mean + 0.5 * worst, 4),
        "face_quality_worst": round(worst, 4),
        "eyes_open": round(min(f.eyes_open for f in subjects), 4),
        "smile": round(float(np.mean([f.smile for f in subjects])), 4),
        "facing_camera": round(float(np.mean([f.facing_camera for f in subjects])), 4),
        "face_sharpness": round(float(np.mean([f.sharpness for f in subjects])), 4),
        "face_visible": round(min(f.visible for f in subjects), 4),
        "face_count": float(len(subjects)),
    }
