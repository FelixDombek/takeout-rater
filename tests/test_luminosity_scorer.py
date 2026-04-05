"""Tests for the LuminosityScorer heuristic."""

from __future__ import annotations

from pathlib import Path

import pytest

from takeout_rater.scorers.heuristics.luminosity import LuminosityScorer


def test_spec_scorer_id() -> None:
    assert LuminosityScorer.spec().scorer_id == "luminosity"


def test_spec_has_brightness_and_contrast_metrics() -> None:
    spec = LuminosityScorer.spec()
    keys = {m.key for m in spec.metrics}
    assert "brightness" in keys
    assert "contrast" in keys


def test_spec_metrics_range() -> None:
    for m in LuminosityScorer.spec().metrics:
        assert m.min_value == 0.0
        assert m.max_value == 100.0


def test_spec_brightness_higher_is_better() -> None:
    spec = LuminosityScorer.spec()
    brightness_metric = next(m for m in spec.metrics if m.key == "brightness")
    assert brightness_metric.higher_is_better is True


def test_spec_contrast_higher_is_better() -> None:
    spec = LuminosityScorer.spec()
    contrast_metric = next(m for m in spec.metrics if m.key == "contrast")
    assert contrast_metric.higher_is_better is True


def test_spec_has_default_variant() -> None:
    spec = LuminosityScorer.spec()
    assert any(v.variant_id == "default" for v in spec.variants)
    assert spec.default_variant_id == "default"


def test_is_available_returns_bool() -> None:
    assert isinstance(LuminosityScorer.is_available(), bool)


def test_score_batch_empty() -> None:
    if not LuminosityScorer.is_available():
        pytest.skip("Pillow not available")
    scorer = LuminosityScorer.create()
    assert scorer.score_batch([]) == []


def test_score_batch_missing_file_returns_zeros(tmp_path: Path) -> None:
    if not LuminosityScorer.is_available():
        pytest.skip("Pillow not available")
    scorer = LuminosityScorer.create()
    result = scorer.score_batch([tmp_path / "does_not_exist.jpg"])
    assert len(result) == 1
    assert result[0]["brightness"] == pytest.approx(0.0)
    assert result[0]["contrast"] == pytest.approx(0.0)


def test_score_batch_real_image_range(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from PIL import Image  # noqa: PLC0415

    p = tmp_path / "test.jpg"
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(p, "JPEG")

    scorer = LuminosityScorer.create()
    result = scorer.score_batch([p])
    assert len(result) == 1
    assert 0.0 <= result[0]["brightness"] <= 100.0
    assert 0.0 <= result[0]["contrast"] <= 100.0


def test_dark_image_lower_brightness_than_bright(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from PIL import Image  # noqa: PLC0415

    dark = tmp_path / "dark.jpg"
    bright = tmp_path / "bright.jpg"
    Image.new("RGB", (64, 64), color=(20, 20, 20)).save(dark, "JPEG")
    Image.new("RGB", (64, 64), color=(235, 235, 235)).save(bright, "JPEG")

    scorer = LuminosityScorer.create()
    dark_score = scorer.score_batch([dark])[0]["brightness"]
    bright_score = scorer.score_batch([bright])[0]["brightness"]
    assert bright_score > dark_score


def test_uniform_image_lower_contrast_than_gradient(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
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

    scorer = LuminosityScorer.create()
    uniform_contrast = scorer.score_batch([uniform])[0]["contrast"]
    gradient_contrast = scorer.score_batch([gradient])[0]["contrast"]
    assert gradient_contrast > uniform_contrast


def test_score_batch_length_matches_input(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from PIL import Image  # noqa: PLC0415

    paths = []
    for i in range(4):
        p = tmp_path / f"img{i}.jpg"
        Image.new("RGB", (32, 32), color=(i * 60, 100, 200)).save(p, "JPEG")
        paths.append(p)

    scorer = LuminosityScorer.create()
    results = scorer.score_batch(paths)
    assert len(results) == len(paths)
    for r in results:
        assert "brightness" in r
        assert "contrast" in r


def test_score_one(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from PIL import Image  # noqa: PLC0415

    p = tmp_path / "img.jpg"
    Image.new("RGB", (32, 32), color=(200, 150, 100)).save(p, "JPEG")
    scorer = LuminosityScorer.create()
    result = scorer.score_one(p)
    assert "brightness" in result
    assert "contrast" in result
    assert 0.0 <= result["brightness"] <= 100.0
    assert 0.0 <= result["contrast"] <= 100.0
