"""Tests for the BRISQUEScorer adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from takeout_rater.scorers.adapters.brisque import BRISQUEScorer


def test_spec_scorer_id() -> None:
    assert BRISQUEScorer.spec().scorer_id == "brisque"


def test_spec_has_brisque_metric() -> None:
    spec = BRISQUEScorer.spec()
    assert len(spec.metrics) == 1
    assert spec.metrics[0].key == "brisque"


def test_spec_brisque_range() -> None:
    m = BRISQUEScorer.spec().metrics[0]
    assert m.min_value == 0.0
    assert m.max_value == 100.0


def test_spec_brisque_lower_is_better() -> None:
    m = BRISQUEScorer.spec().metrics[0]
    assert m.higher_is_better is False


def test_spec_has_default_variant() -> None:
    spec = BRISQUEScorer.spec()
    assert any(v.variant_id == "default" for v in spec.variants)
    assert spec.default_variant_id == "default"


def test_spec_requires_brisque_extra() -> None:
    assert "brisque" in BRISQUEScorer.spec().requires_extras


def test_is_available_returns_bool() -> None:
    assert isinstance(BRISQUEScorer.is_available(), bool)


@pytest.mark.skipif(
    not BRISQUEScorer.is_available(),
    reason="scikit-image not installed",
)
def test_score_batch_empty() -> None:
    scorer = BRISQUEScorer.create()
    assert scorer.score_batch([]) == []


@pytest.mark.skipif(
    not BRISQUEScorer.is_available(),
    reason="scikit-image not installed",
)
def test_score_batch_missing_file_returns_max(tmp_path: Path) -> None:
    scorer = BRISQUEScorer.create()
    result = scorer.score_batch([tmp_path / "does_not_exist.jpg"])
    assert len(result) == 1
    assert result[0]["brisque"] == pytest.approx(100.0)


@pytest.mark.skipif(
    not BRISQUEScorer.is_available(),
    reason="scikit-image not installed",
)
def test_score_batch_real_image_range(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from PIL import Image  # noqa: PLC0415

    p = tmp_path / "test.png"
    img = Image.new("RGB", (128, 128))
    for x in range(128):
        for y in range(128):
            img.putpixel((x, y), (x * 2, y * 2, 128))
    img.save(p, "PNG")

    scorer = BRISQUEScorer.create()
    result = scorer.score_batch([p])
    assert len(result) == 1
    assert 0.0 <= result[0]["brisque"] <= 100.0


@pytest.mark.skipif(
    not BRISQUEScorer.is_available(),
    reason="scikit-image not installed",
)
def test_score_batch_length_matches_input(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from PIL import Image  # noqa: PLC0415

    paths = []
    for i in range(3):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (64, 64), color=(i * 80, 100, 150)).save(p, "PNG")
        paths.append(p)

    scorer = BRISQUEScorer.create()
    results = scorer.score_batch(paths)
    assert len(results) == len(paths)
    for r in results:
        assert "brisque" in r
        assert 0.0 <= r["brisque"] <= 100.0
