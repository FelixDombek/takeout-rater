"""Tests for the face quality scorer (models are stubbed out)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from takeout_rater.faces.quality import PHOTO_METRIC_KEYS
from takeout_rater.scoring.scorers.face_quality import FaceQualityScorer
from takeout_rater.scoring.scorers.registry import list_scorers

cv2 = pytest.importorskip("cv2")
pytest.importorskip("insightface")


def test_spec() -> None:
    spec = FaceQualityScorer.spec()
    assert spec.scorer_id == "face_quality"
    assert spec.default_variant_id == "buffalo_l"
    variant = spec.variants[0]
    assert variant.primary_metric_key == "face_count"
    assert tuple(m.key for m in variant.metrics) == PHOTO_METRIC_KEYS
    assert all(m.higher_is_better for m in variant.metrics)


def test_registered() -> None:
    assert FaceQualityScorer in list_scorers()


def test_is_available_returns_bool() -> None:
    assert isinstance(FaceQualityScorer.is_available(), bool)


def test_empty_batch_does_not_load_models() -> None:
    scorer = FaceQualityScorer.create()
    assert scorer.score_batch([]) == []
    assert scorer._app is None  # noqa: SLF001


class _FakeApp:
    """Stands in for insightface FaceAnalysis: one frontal face with open eyes."""

    def __init__(self, faces_per_image: int) -> None:
        self.faces_per_image = faces_per_image

    def get(self, img: np.ndarray) -> list[SimpleNamespace]:
        from tests.faces.test_quality import _landmarks

        faces = []
        for i in range(self.faces_per_image):
            ox = 40.0 + 120.0 * i
            lm = _landmarks(6.0, centre=(ox + 40.0, 100.0))
            kps = np.array(
                [[ox + 25, 85], [ox + 55, 85], [ox + 40, 100], [ox + 28, 115], [ox + 52, 115]],
                dtype=np.float32,
            )
            faces.append(
                SimpleNamespace(
                    bbox=np.array([ox, 50.0, ox + 80.0, 150.0]),
                    det_score=0.88,
                    kps=kps,
                    landmark_2d_106=lm,
                    pose=np.array([-4.0, 0.0, 0.0]),
                )
            )
        return faces


class _FakeFer:
    def run(self, _outputs: object, _feeds: object) -> list[np.ndarray]:
        logits = np.array([[0.0, 3.0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
        return [logits]


def _stub_models(scorer: FaceQualityScorer, faces_per_image: int) -> None:
    scorer._app = _FakeApp(faces_per_image)  # noqa: SLF001
    scorer._fer = _FakeFer()  # noqa: SLF001
    scorer._fer_input = "x"  # noqa: SLF001


def test_score_batch_with_stubbed_models(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
    good = tmp_path / "ünïcode.jpg"
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    good.write_bytes(buf.tobytes())
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not a jpeg")
    missing = tmp_path / "missing.jpg"

    scorer = FaceQualityScorer.create()
    _stub_models(scorer, faces_per_image=2)
    results = scorer.score_batch([good, broken, missing])

    assert len(results) == 3
    assert results[1] == {}
    assert results[2] == {}
    metrics = results[0]
    assert set(metrics) == set(PHOTO_METRIC_KEYS)
    assert metrics["face_count"] == 2.0
    assert metrics["eyes_open"] == 1.0
    assert metrics["smile"] == pytest.approx(np.exp(3) / (np.exp(3) + 7), abs=1e-3)
    assert metrics["facing_camera"] == pytest.approx(1.0)


def test_score_batch_without_faces(tmp_path: Path) -> None:
    path = tmp_path / "plain.jpg"
    ok, buf = cv2.imencode(".jpg", np.zeros((64, 64, 3), dtype=np.uint8))
    assert ok
    path.write_bytes(buf.tobytes())

    scorer = FaceQualityScorer.create()
    _stub_models(scorer, faces_per_image=0)
    assert scorer.score_batch([path]) == [{"face_count": 0.0}]


def test_close_releases_models() -> None:
    scorer = FaceQualityScorer.create()
    _stub_models(scorer, faces_per_image=1)
    scorer.close()
    assert scorer._app is None  # noqa: SLF001
    assert scorer._fer is None  # noqa: SLF001
