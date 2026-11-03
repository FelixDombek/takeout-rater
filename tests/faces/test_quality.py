"""Tests for the per-face quality metrics and per-photo aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from takeout_rater.faces import quality as fq


def _landmarks(eye_open_px: float, *, centre: tuple[float, float] = (100.0, 100.0)) -> np.ndarray:
    """106 landmarks around *centre* with both eyes 20 px wide and *eye_open_px* tall."""
    cx, cy = centre
    pts = np.tile(np.array([cx, cy], dtype=np.float64), (106, 1))
    for eye, ex in ((fq.EYE_106_IMAGE_LEFT, cx - 20.0), (fq.EYE_106_IMAGE_RIGHT, cx + 20.0)):
        for (upper, lower), dx in zip(eye[:3], (-5.0, 0.0, 5.0), strict=True):
            pts[upper] = (ex + dx, cy - 10 - eye_open_px / 2)
            pts[lower] = (ex + dx, cy - 10 + eye_open_px / 2)
        c1, c2 = eye[3]
        pts[c1] = (ex - 10.0, cy - 10)
        pts[c2] = (ex + 10.0, cy - 10)
    return pts


def _face(size: float = 100.0, **overrides: float) -> fq.FaceQuality:
    values = {
        "eyes_open": 1.0,
        "smile": 0.5,
        "facing_camera": 1.0,
        "sharpness": 1.0,
        "visible": 1.0,
    }
    values.update(overrides)
    return fq.FaceQuality(
        bbox=(0.0, 0.0, size, size),
        det_score=0.85,
        quality=fq.face_quality_score(**values),
        **values,
    )


def test_eye_aspect_ratio_open_vs_closed() -> None:
    assert fq.eye_aspect_ratio(_landmarks(6.0), fq.EYE_106_IMAGE_LEFT) == pytest.approx(0.3)
    assert fq.eye_aspect_ratio(_landmarks(1.0), fq.EYE_106_IMAGE_RIGHT) == pytest.approx(0.05)


def test_eyes_open_score_uses_worse_eye_when_frontal() -> None:
    lm = _landmarks(6.0)
    # Close the image-right eye only (a wink).
    closed = _landmarks(1.0)
    for idx in [i for pair in fq.EYE_106_IMAGE_RIGHT for i in pair]:
        lm[idx] = closed[idx]
    score, ear_l, ear_r = fq.eyes_open_score(lm, yaw_deg=0.0)
    assert ear_l > ear_r
    assert score == 0.0
    # Strongly turned head: the far eye is unreliable, the better eye counts.
    turned, _, _ = fq.eyes_open_score(lm, yaw_deg=60.0)
    assert turned == 1.0


def test_eye_open_score_ramp() -> None:
    assert fq.eye_open_score(0.05) == 0.0
    assert fq.eye_open_score(0.30) == 1.0
    mid = fq.eye_open_score((fq.EAR_CLOSED + fq.EAR_OPEN) / 2)
    assert mid == pytest.approx(0.5)


def test_facing_camera_score_falls_off_with_yaw() -> None:
    frontal = fq.facing_camera_score(fq.PITCH_OFFSET_DEG, 0.0)
    assert frontal == pytest.approx(1.0)
    s20 = fq.facing_camera_score(fq.PITCH_OFFSET_DEG, 20.0)
    s60 = fq.facing_camera_score(fq.PITCH_OFFSET_DEG, -60.0)
    assert 0.7 < s20 < 0.9
    assert s60 < 0.2


def test_smile_score() -> None:
    assert fq.smile_score(None) == 0.0
    assert fq.smile_score([0.1, 0.8, 0.1, 0, 0, 0, 0, 0]) == pytest.approx(0.8)


def test_sharpness_score_mapping() -> None:
    assert fq.sharpness_score(0.0) == 0.0
    assert fq.sharpness_score(float("nan")) == 0.0
    assert fq.sharpness_score(fq.SHARPNESS_RAW_HIGH * 2) == 1.0
    assert 0.0 < fq.sharpness_score(0.2) < 1.0


def test_in_frame_fraction_detects_cut_off_face() -> None:
    lm = _landmarks(6.0, centre=(100.0, 100.0))
    assert fq.in_frame_fraction((60, 60, 140, 140), lm, 400, 400) == pytest.approx(1.0)
    # Same face with the frame edge running through its middle.
    shifted = lm - np.array([100.0, 0.0])
    frac = fq.in_frame_fraction((-40, 60, 40, 140), shifted, 400, 400)
    assert 0.3 < frac < 0.7


def test_visibility_score() -> None:
    assert fq.visibility_score(1.0, 0.9) == pytest.approx(1.0)
    assert fq.visibility_score(0.5, 0.9) == 0.0
    assert fq.visibility_score(1.0, 0.5) == pytest.approx(0.6)


def test_closed_eyes_sink_face_quality() -> None:
    good = fq.face_quality_score(1.0, 0.0, 1.0, 1.0, 1.0)
    blink = fq.face_quality_score(0.0, 1.0, 1.0, 1.0, 1.0)
    assert blink < 0.5 * good


def test_build_face_quality_end_to_end() -> None:
    face = fq.build_face_quality(
        bbox=(60, 60, 140, 140),
        det_score=0.85,
        landmarks_106=_landmarks(6.0),
        pose=(fq.PITCH_OFFSET_DEG, 0.0, 0.0),
        expression_probs=[0.1, 0.9, 0, 0, 0, 0, 0, 0],
        sharpness_raw_value=1.0,
        width=400,
        height=400,
    )
    assert face.eyes_open == 1.0
    assert face.smile == pytest.approx(0.9)
    assert face.facing_camera == pytest.approx(1.0)
    assert face.visible == pytest.approx(1.0)
    assert face.quality > 0.9


def test_aggregate_without_faces() -> None:
    assert fq.aggregate_faces([]) == {"face_count": 0.0}


def test_aggregate_ignores_background_and_tiny_faces() -> None:
    faces = [_face(), _face(size=20.0, eyes_open=0.0), _face(size=10.0, eyes_open=0.0)]
    metrics = fq.aggregate_faces(faces)
    assert metrics["face_count"] == 1.0
    assert metrics["eyes_open"] == 1.0


def test_aggregate_worst_face_sinks_group_shot() -> None:
    faces = [_face(), _face(), _face(size=90.0, eyes_open=0.0)]
    metrics = fq.aggregate_faces(faces)
    assert set(metrics) == set(fq.PHOTO_METRIC_KEYS)
    assert metrics["face_count"] == 3.0
    assert metrics["eyes_open"] == 0.0
    assert metrics["face_quality_worst"] < metrics["face_quality"]
    all_good = fq.aggregate_faces([_face(), _face(), _face()])
    assert metrics["face_quality"] < all_good["face_quality"] - 0.2
    for key, value in metrics.items():
        if key != "face_count":
            assert 0.0 <= value <= 1.0
