"""Tests for the NoiseScorer heuristic."""

from __future__ import annotations

from pathlib import Path

import pytest

from takeout_rater.scorers.heuristics.noise import NoiseScorer


def test_spec_scorer_id() -> None:
    assert NoiseScorer.spec().scorer_id == "noise"


def test_spec_has_noise_metric() -> None:
    spec = NoiseScorer.spec()
    assert len(spec.metrics) == 1
    assert spec.metrics[0].key == "noise"


def test_spec_noise_range() -> None:
    m = NoiseScorer.spec().metrics[0]
    assert m.min_value == 0.0
    assert m.max_value == 100.0


def test_spec_noise_lower_is_better() -> None:
    m = NoiseScorer.spec().metrics[0]
    assert m.higher_is_better is False


def test_spec_has_default_variant() -> None:
    spec = NoiseScorer.spec()
    assert any(v.variant_id == "default" for v in spec.variants)
    assert spec.default_variant_id == "default"


def test_is_available_returns_bool() -> None:
    assert isinstance(NoiseScorer.is_available(), bool)


def test_score_batch_empty() -> None:
    if not NoiseScorer.is_available():
        pytest.skip("Pillow not available")
    scorer = NoiseScorer.create()
    assert scorer.score_batch([]) == []


def test_score_batch_missing_file_returns_zero(tmp_path: Path) -> None:
    if not NoiseScorer.is_available():
        pytest.skip("Pillow not available")
    scorer = NoiseScorer.create()
    result = scorer.score_batch([tmp_path / "does_not_exist.jpg"])
    assert len(result) == 1
    assert result[0]["noise"] == pytest.approx(0.0)


def test_score_batch_real_image_range(tmp_path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    p = tmp_path / "test.jpg"
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(p, "JPEG")

    scorer = NoiseScorer.create()
    result = scorer.score_batch([p])
    assert len(result) == 1
    assert 0.0 <= result[0]["noise"] <= 100.0


def test_uniform_image_low_noise(tmp_path: Path) -> None:
    """A uniform solid-colour image should have near-zero noise."""
    from PIL import Image  # noqa: PLC0415

    # Use PNG to avoid JPEG compression artefacts introducing false noise
    p = tmp_path / "uniform.png"
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(p, "PNG")

    scorer = NoiseScorer.create()
    result = scorer.score_batch([p])[0]
    # Uniform image should have very low noise (JPEG artefacts may add a tiny amount)
    assert result["noise"] < 20.0


def test_score_batch_length_matches_input(tmp_path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    paths = []
    for i in range(4):
        p = tmp_path / f"img{i}.jpg"
        Image.new("RGB", (32, 32), color=(i * 60, 100, 200)).save(p, "JPEG")
        paths.append(p)

    scorer = NoiseScorer.create()
    results = scorer.score_batch(paths)
    assert len(results) == len(paths)
    for r in results:
        assert "noise" in r
        assert 0.0 <= r["noise"] <= 100.0


def test_score_one(tmp_path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    p = tmp_path / "img.jpg"
    Image.new("RGB", (32, 32), color=(200, 150, 100)).save(p, "JPEG")
    scorer = NoiseScorer.create()
    result = scorer.score_one(p)
    assert "noise" in result
    assert 0.0 <= result["noise"] <= 100.0
