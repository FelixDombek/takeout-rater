"""Tests for the BlurScorer backward-compat shim (now an alias for SimpleScorer)."""

from __future__ import annotations

from pathlib import Path

import pytest

from takeout_rater.scorers.heuristics.blur import BlurScorer
from takeout_rater.scorers.heuristics.simple import SimpleScorer


def test_blur_scorer_alias_is_simple_scorer() -> None:
    assert BlurScorer is SimpleScorer


def test_spec_scorer_id() -> None:
    assert BlurScorer.spec().scorer_id == "simple"


def test_spec_has_sharpness_metric() -> None:
    spec = BlurScorer.spec()
    assert any(m.key == "sharpness" for m in spec.metrics)


def test_spec_sharpness_range() -> None:
    m = next(m for m in BlurScorer.spec().metrics if m.key == "sharpness")
    assert m.min_value == 0.0
    assert m.max_value == 100.0
    assert m.higher_is_better is True


def test_spec_has_blur_variant() -> None:
    spec = BlurScorer.spec()
    assert any(v.variant_id == "blur" for v in spec.variants)


def test_is_available_returns_bool() -> None:
    result = BlurScorer.is_available()
    assert isinstance(result, bool)


def test_score_batch_empty() -> None:
    if not BlurScorer.is_available():
        pytest.skip("Pillow not available")
    scorer = BlurScorer.create(variant_id="blur")
    assert scorer.score_batch([]) == []


def test_score_batch_missing_file_returns_zero(tmp_path: Path) -> None:
    """A missing image file should yield sharpness = 0.0, not raise."""
    if not BlurScorer.is_available():
        pytest.skip("Pillow not available")
    scorer = BlurScorer.create(variant_id="blur")
    result = scorer.score_batch([tmp_path / "does_not_exist.jpg"])
    assert len(result) == 1
    assert result[0]["sharpness"] == pytest.approx(0.0)


def test_score_batch_real_image(tmp_path: Path) -> None:
    """A valid image should produce a sharpness in [0, 100]."""
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (64, 64))
    for x in range(64):
        for y in range(64):
            img.putpixel((x, y), (x * 4, y * 4, 128))
    img.save(img_path, "JPEG")

    scorer = BlurScorer.create(variant_id="blur")
    result = scorer.score_batch([img_path])
    assert len(result) == 1
    sharpness = result[0]["sharpness"]
    assert 0.0 <= sharpness <= 100.0


def test_score_batch_uniform_image_lower_than_gradient(tmp_path: Path) -> None:
    """A uniform (blurry) image should score lower than a high-contrast image."""
    from PIL import Image  # noqa: PLC0415

    uniform = tmp_path / "uniform.jpg"
    gradient = tmp_path / "gradient.jpg"

    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(uniform, "JPEG")

    img = Image.new("RGB", (64, 64))
    for x in range(64):
        for y in range(64):
            color = (0, 0, 0) if (x + y) % 2 == 0 else (255, 255, 255)
            img.putpixel((x, y), color)
    img.save(gradient, "JPEG")

    scorer = BlurScorer.create(variant_id="blur")
    uniform_score = scorer.score_batch([uniform])[0]["sharpness"]
    gradient_score = scorer.score_batch([gradient])[0]["sharpness"]
    assert gradient_score > uniform_score


def test_score_batch_length_matches_input(tmp_path: Path) -> None:
    """score_batch must return one result per input path."""
    from PIL import Image  # noqa: PLC0415

    paths = []
    for i in range(4):
        p = tmp_path / f"img{i}.jpg"
        Image.new("RGB", (32, 32), color=(i * 60, 100, 200)).save(p, "JPEG")
        paths.append(p)

    scorer = BlurScorer.create(variant_id="blur")
    results = scorer.score_batch(paths)
    assert len(results) == len(paths)
    for r in results:
        assert "sharpness" in r


def test_score_one(tmp_path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    p = tmp_path / "img.jpg"
    Image.new("RGB", (32, 32), color=(200, 150, 100)).save(p, "JPEG")
    scorer = BlurScorer.create(variant_id="blur")
    result = scorer.score_one(p)
    assert "sharpness" in result
    assert 0.0 <= result["sharpness"] <= 100.0
