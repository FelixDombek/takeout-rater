"""Tests for the NoiseScorer backward-compat shim (now an alias for SimpleScorer)."""

from __future__ import annotations

from pathlib import Path

import pytest

from takeout_rater.scorers.heuristics.noise import NoiseScorer
from takeout_rater.scorers.heuristics.simple import SimpleScorer


def test_noise_scorer_alias_is_simple_scorer() -> None:
    assert NoiseScorer is SimpleScorer


def test_spec_scorer_id() -> None:
    assert NoiseScorer.spec().scorer_id == "simple"


def test_spec_has_noise_metric() -> None:
    spec = NoiseScorer.spec()
    assert any(m.key == "noise" for m in spec.metrics)


def test_spec_noise_range() -> None:
    m = next(m for m in NoiseScorer.spec().metrics if m.key == "noise")
    assert m.min_value == 0.0
    assert m.max_value == 100.0


def test_spec_noise_lower_is_better() -> None:
    m = next(m for m in NoiseScorer.spec().metrics if m.key == "noise")
    assert m.higher_is_better is False


def test_spec_has_noise_variant() -> None:
    spec = NoiseScorer.spec()
    assert any(v.variant_id == "noise" for v in spec.variants)


def test_is_available_returns_bool() -> None:
    assert isinstance(NoiseScorer.is_available(), bool)


def test_score_batch_empty() -> None:
    if not NoiseScorer.is_available():
        pytest.skip("Pillow not available")
    scorer = NoiseScorer.create(variant_id="noise")
    assert scorer.score_batch([]) == []


def test_score_batch_missing_file_returns_zero(tmp_path: Path) -> None:
    if not NoiseScorer.is_available():
        pytest.skip("Pillow not available")
    scorer = NoiseScorer.create(variant_id="noise")
    result = scorer.score_batch([tmp_path / "does_not_exist.jpg"])
    assert len(result) == 1
    assert result[0]["noise"] == pytest.approx(0.0)


def test_score_batch_real_image_range(tmp_path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    p = tmp_path / "test.jpg"
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(p, "JPEG")

    scorer = NoiseScorer.create(variant_id="noise")
    result = scorer.score_batch([p])
    assert len(result) == 1
    assert 0.0 <= result[0]["noise"] <= 100.0


def test_uniform_image_low_noise(tmp_path: Path) -> None:
    """A uniform solid-colour image should have near-zero noise."""
    from PIL import Image  # noqa: PLC0415

    p = tmp_path / "uniform.png"
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(p, "PNG")

    scorer = NoiseScorer.create(variant_id="noise")
    result = scorer.score_batch([p])[0]
    assert result["noise"] < 20.0


def test_score_batch_length_matches_input(tmp_path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    paths = []
    for i in range(4):
        p = tmp_path / f"img{i}.jpg"
        Image.new("RGB", (32, 32), color=(i * 60, 100, 200)).save(p, "JPEG")
        paths.append(p)

    scorer = NoiseScorer.create(variant_id="noise")
    results = scorer.score_batch(paths)
    assert len(results) == len(paths)
    for r in results:
        assert "noise" in r
        assert 0.0 <= r["noise"] <= 100.0


def test_score_one(tmp_path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    p = tmp_path / "img.jpg"
    Image.new("RGB", (32, 32), color=(200, 150, 100)).save(p, "JPEG")
    scorer = NoiseScorer.create(variant_id="noise")
    result = scorer.score_one(p)
    assert "noise" in result
    assert 0.0 <= result["noise"] <= 100.0
