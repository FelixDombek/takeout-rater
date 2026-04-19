"""Tests for the SimpleScorer — the merged Pillow-only heuristic scorer."""

from __future__ import annotations

from pathlib import Path

import pytest

from takeout_rater.scoring.scorers.simple import SimpleScorer

# ---------------------------------------------------------------------------
# Spec / metadata
# ---------------------------------------------------------------------------


def test_spec_scorer_id() -> None:
    assert SimpleScorer.spec().scorer_id == "simple"


def test_spec_has_three_variants() -> None:
    spec = SimpleScorer.spec()
    ids = {v.variant_id for v in spec.variants}
    assert ids == {"blur", "luminosity", "noise"}


def test_spec_default_variant_is_blur() -> None:
    assert SimpleScorer.spec().default_variant_id == "blur"


def test_spec_has_all_metrics() -> None:
    keys = {m.key for m in SimpleScorer.spec().all_metrics()}
    assert keys == {"sharpness", "brightness", "contrast", "noise"}


def test_spec_sharpness_range() -> None:
    m = next(m for m in SimpleScorer.spec().all_metrics() if m.key == "sharpness")
    assert m.min_value == 0.0
    assert m.max_value == 100.0
    assert m.higher_is_better is True


def test_spec_brightness_range() -> None:
    m = next(m for m in SimpleScorer.spec().all_metrics() if m.key == "brightness")
    assert m.min_value == 0.0
    assert m.max_value == 100.0
    assert m.higher_is_better is True


def test_spec_contrast_range() -> None:
    m = next(m for m in SimpleScorer.spec().all_metrics() if m.key == "contrast")
    assert m.min_value == 0.0
    assert m.max_value == 100.0
    assert m.higher_is_better is True


def test_spec_noise_range_lower_is_better() -> None:
    m = next(m for m in SimpleScorer.spec().all_metrics() if m.key == "noise")
    assert m.min_value == 0.0
    assert m.max_value == 100.0
    assert m.higher_is_better is False


def test_is_available_returns_bool() -> None:
    assert isinstance(SimpleScorer.is_available(), bool)


# ---------------------------------------------------------------------------
# blur variant
# ---------------------------------------------------------------------------


def test_blur_variant_score_batch_empty() -> None:
    if not SimpleScorer.is_available():
        pytest.skip("Pillow not available")
    scorer = SimpleScorer.create(variant_id="blur")
    assert scorer.score_batch([]) == []


def test_blur_variant_missing_file_returns_zero(tmp_path: Path) -> None:
    if not SimpleScorer.is_available():
        pytest.skip("Pillow not available")
    scorer = SimpleScorer.create(variant_id="blur")
    result = scorer.score_batch([tmp_path / "does_not_exist.jpg"])
    assert len(result) == 1
    assert result[0]["sharpness"] == pytest.approx(0.0)


def test_blur_variant_real_image(tmp_path: Path) -> None:
    from PIL import Image

    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (64, 64))
    for x in range(64):
        for y in range(64):
            img.putpixel((x, y), (x * 4, y * 4, 128))
    img.save(img_path, "JPEG")

    scorer = SimpleScorer.create(variant_id="blur")
    result = scorer.score_batch([img_path])
    assert len(result) == 1
    sharpness = result[0]["sharpness"]
    assert 0.0 <= sharpness <= 100.0


def test_blur_variant_uniform_vs_gradient(tmp_path: Path) -> None:
    from PIL import Image

    uniform = tmp_path / "uniform.jpg"
    gradient = tmp_path / "gradient.jpg"
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(uniform, "JPEG")
    img = Image.new("RGB", (64, 64))
    for x in range(64):
        for y in range(64):
            color = (0, 0, 0) if (x + y) % 2 == 0 else (255, 255, 255)
            img.putpixel((x, y), color)
    img.save(gradient, "JPEG")

    scorer = SimpleScorer.create(variant_id="blur")
    uniform_score = scorer.score_batch([uniform])[0]["sharpness"]
    gradient_score = scorer.score_batch([gradient])[0]["sharpness"]
    assert gradient_score > uniform_score


# ---------------------------------------------------------------------------
# luminosity variant
# ---------------------------------------------------------------------------


def test_luminosity_variant_score_batch_empty() -> None:
    if not SimpleScorer.is_available():
        pytest.skip("Pillow not available")
    scorer = SimpleScorer.create(variant_id="luminosity")
    assert scorer.score_batch([]) == []


def test_luminosity_variant_missing_file_returns_zeros(tmp_path: Path) -> None:
    if not SimpleScorer.is_available():
        pytest.skip("Pillow not available")
    scorer = SimpleScorer.create(variant_id="luminosity")
    result = scorer.score_batch([tmp_path / "does_not_exist.jpg"])
    assert len(result) == 1
    assert result[0]["brightness"] == pytest.approx(0.0)
    assert result[0]["contrast"] == pytest.approx(0.0)


def test_luminosity_variant_real_image_range(tmp_path: Path) -> None:
    from PIL import Image

    p = tmp_path / "test.jpg"
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(p, "JPEG")

    scorer = SimpleScorer.create(variant_id="luminosity")
    result = scorer.score_batch([p])
    assert len(result) == 1
    assert 0.0 <= result[0]["brightness"] <= 100.0
    assert 0.0 <= result[0]["contrast"] <= 100.0


def test_luminosity_variant_dark_vs_bright(tmp_path: Path) -> None:
    from PIL import Image

    dark = tmp_path / "dark.jpg"
    bright = tmp_path / "bright.jpg"
    Image.new("RGB", (64, 64), color=(20, 20, 20)).save(dark, "JPEG")
    Image.new("RGB", (64, 64), color=(235, 235, 235)).save(bright, "JPEG")

    scorer = SimpleScorer.create(variant_id="luminosity")
    dark_score = scorer.score_batch([dark])[0]["brightness"]
    bright_score = scorer.score_batch([bright])[0]["brightness"]
    assert bright_score > dark_score


def test_luminosity_variant_uniform_vs_gradient_contrast(tmp_path: Path) -> None:
    from PIL import Image

    uniform = tmp_path / "uniform.jpg"
    gradient = tmp_path / "gradient.jpg"
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(uniform, "JPEG")
    img = Image.new("RGB", (64, 64))
    for x in range(64):
        for y in range(64):
            color = (0, 0, 0) if (x + y) % 2 == 0 else (255, 255, 255)
            img.putpixel((x, y), color)
    img.save(gradient, "JPEG")

    scorer = SimpleScorer.create(variant_id="luminosity")
    uniform_contrast = scorer.score_batch([uniform])[0]["contrast"]
    gradient_contrast = scorer.score_batch([gradient])[0]["contrast"]
    assert gradient_contrast > uniform_contrast


# ---------------------------------------------------------------------------
# noise variant
# ---------------------------------------------------------------------------


def test_noise_variant_score_batch_empty() -> None:
    if not SimpleScorer.is_available():
        pytest.skip("Pillow not available")
    scorer = SimpleScorer.create(variant_id="noise")
    assert scorer.score_batch([]) == []


def test_noise_variant_missing_file_returns_zero(tmp_path: Path) -> None:
    if not SimpleScorer.is_available():
        pytest.skip("Pillow not available")
    scorer = SimpleScorer.create(variant_id="noise")
    result = scorer.score_batch([tmp_path / "does_not_exist.jpg"])
    assert len(result) == 1
    assert result[0]["noise"] == pytest.approx(0.0)


def test_noise_variant_real_image_range(tmp_path: Path) -> None:
    from PIL import Image

    p = tmp_path / "test.jpg"
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(p, "JPEG")

    scorer = SimpleScorer.create(variant_id="noise")
    result = scorer.score_batch([p])
    assert len(result) == 1
    assert 0.0 <= result[0]["noise"] <= 100.0


def test_noise_variant_uniform_image_low_noise(tmp_path: Path) -> None:
    from PIL import Image

    p = tmp_path / "uniform.png"
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(p, "PNG")

    scorer = SimpleScorer.create(variant_id="noise")
    result = scorer.score_batch([p])[0]
    assert result["noise"] < 20.0


# ---------------------------------------------------------------------------
# General: invalid variant, batch length, score_one
# ---------------------------------------------------------------------------


def test_invalid_variant_raises() -> None:
    if not SimpleScorer.is_available():
        pytest.skip("Pillow not available")
    scorer = SimpleScorer.create(variant_id="invalid")
    with pytest.raises(ValueError, match="Unknown SimpleScorer variant"):
        scorer.score_batch([])


def test_score_batch_length_matches_input(tmp_path: Path) -> None:
    from PIL import Image

    paths = []
    for i in range(4):
        p = tmp_path / f"img{i}.jpg"
        Image.new("RGB", (32, 32), color=(i * 60, 100, 200)).save(p, "JPEG")
        paths.append(p)

    for variant in ("blur", "luminosity", "noise"):
        scorer = SimpleScorer.create(variant_id=variant)
        results = scorer.score_batch(paths)
        assert len(results) == len(paths)


def test_score_one_blur(tmp_path: Path) -> None:
    from PIL import Image

    p = tmp_path / "img.jpg"
    Image.new("RGB", (32, 32), color=(200, 150, 100)).save(p, "JPEG")
    scorer = SimpleScorer.create(variant_id="blur")
    result = scorer.score_one(p)
    assert "sharpness" in result
    assert 0.0 <= result["sharpness"] <= 100.0
