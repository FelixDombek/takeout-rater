"""Tests for the PyIQAScorer adapter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from takeout_rater.scorers.adapters.pyiqa_adapter import PyIQAScorer, _to_higher_is_better

# ---------------------------------------------------------------------------
# Spec tests — no dependencies needed
# ---------------------------------------------------------------------------


def test_spec_scorer_id() -> None:
    assert PyIQAScorer.spec().scorer_id == "pyiqa"


def test_spec_has_iqa_quality_metric() -> None:
    spec = PyIQAScorer.spec()
    assert len(spec.metrics) == 1
    assert spec.metrics[0].key == "iqa_quality"


def test_spec_range() -> None:
    m = PyIQAScorer.spec().metrics[0]
    assert m.min_value == 0.0
    assert m.max_value == 1.0
    assert m.higher_is_better is True


def test_spec_has_musiq_topiq_niqe_variants() -> None:
    spec = PyIQAScorer.spec()
    variant_ids = {v.variant_id for v in spec.variants}
    assert "musiq" in variant_ids
    assert "topiq_nr" in variant_ids
    assert "niqe" in variant_ids


def test_spec_default_variant_is_musiq() -> None:
    assert PyIQAScorer.spec().default_variant_id == "musiq"


def test_spec_requires_no_extras() -> None:
    assert PyIQAScorer.spec().requires_extras == ()


def test_spec_display_name_not_empty() -> None:
    spec = PyIQAScorer.spec()
    assert spec.display_name
    assert spec.description


# ---------------------------------------------------------------------------
# _to_higher_is_better normalisation
# ---------------------------------------------------------------------------


def test_normalise_musiq_midrange() -> None:
    """MUSIQ raw=50 → quality=0.5."""
    assert _to_higher_is_better("musiq", 50.0) == pytest.approx(0.5)


def test_normalise_musiq_max() -> None:
    assert _to_higher_is_better("musiq", 100.0) == pytest.approx(1.0)


def test_normalise_musiq_above_max_clamped() -> None:
    assert _to_higher_is_better("musiq", 150.0) == pytest.approx(1.0)


def test_normalise_topiq_passthrough() -> None:
    """TOPIQ raw is already in [0, 1]."""
    assert _to_higher_is_better("topiq_nr", 0.6) == pytest.approx(0.6)


def test_normalise_niqe_zero_maps_to_one() -> None:
    """NIQE raw=0 (perfect) → quality=1.0."""
    assert _to_higher_is_better("niqe", 0.0) == pytest.approx(1.0)


def test_normalise_niqe_large_maps_near_zero() -> None:
    """NIQE raw=999 (terrible) → quality ≈ 0."""
    quality = _to_higher_is_better("niqe", 999.0)
    assert quality < 0.01


def test_normalise_niqe_monotone_decreasing() -> None:
    """Higher raw NIQE must yield lower quality."""
    q1 = _to_higher_is_better("niqe", 5.0)
    q2 = _to_higher_is_better("niqe", 10.0)
    assert q1 > q2


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------


def test_is_available_returns_bool() -> None:
    result = PyIQAScorer.is_available()
    assert isinstance(result, bool)


def test_is_available_false_when_pyiqa_missing() -> None:
    import builtins  # noqa: PLC0415

    real_import = builtins.__import__

    def _mock_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "pyiqa":
            raise ImportError("mocked missing pyiqa")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_mock_import):
        assert PyIQAScorer.is_available() is False


# ---------------------------------------------------------------------------
# score_batch edge cases (no model required)
# ---------------------------------------------------------------------------


def test_score_batch_empty_returns_empty() -> None:
    scorer = PyIQAScorer.create()
    assert scorer.score_batch([]) == []


# ---------------------------------------------------------------------------
# score_batch with mocked pyiqa metric
# ---------------------------------------------------------------------------


def _make_mock_scorer(raw_score: float = 70.0, variant_id: str = "musiq") -> PyIQAScorer:
    """Return a PyIQAScorer with a mocked pyiqa metric that returns *raw_score*."""
    import torch  # noqa: PLC0415

    scorer = PyIQAScorer.create(variant_id=variant_id)

    # Return per-image scores of shape (N, 1) to match the batched score_batch.
    fake_metric = MagicMock(side_effect=lambda t: torch.full((t.shape[0], 1), raw_score))
    scorer._metric = fake_metric
    return scorer


def test_score_batch_returns_iqa_quality_key(tmp_path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64)).save(img_path, "JPEG")

    scorer = _make_mock_scorer(raw_score=70.0, variant_id="musiq")
    results = scorer.score_batch([img_path])
    assert len(results) == 1
    assert "iqa_quality" in results[0]


def test_score_batch_musiq_normalisation(tmp_path: Path) -> None:
    """MUSIQ raw=70 should yield quality≈0.7."""
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64)).save(img_path, "JPEG")

    scorer = _make_mock_scorer(raw_score=70.0, variant_id="musiq")
    results = scorer.score_batch([img_path])
    assert results[0]["iqa_quality"] == pytest.approx(0.7, abs=1e-4)


def test_score_batch_niqe_inversion(tmp_path: Path) -> None:
    """NIQE raw=0 should map to quality=1.0."""
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64)).save(img_path, "JPEG")

    scorer = _make_mock_scorer(raw_score=0.0, variant_id="niqe")
    results = scorer.score_batch([img_path])
    assert results[0]["iqa_quality"] == pytest.approx(1.0)


def test_score_batch_missing_file_returns_zero(tmp_path: Path) -> None:
    scorer = _make_mock_scorer()
    result = scorer.score_batch([tmp_path / "does_not_exist.jpg"])
    assert len(result) == 1
    assert result[0]["iqa_quality"] == pytest.approx(0.0)


def test_score_batch_length_matches_input(tmp_path: Path) -> None:
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    paths = []
    for i in range(3):
        p = tmp_path / f"img{i}.jpg"
        Image.new("RGB", (32, 32), color=(i * 80, 100, 200)).save(p, "JPEG")
        paths.append(p)

    scorer = PyIQAScorer.create(variant_id="musiq")
    scorer._metric = MagicMock(side_effect=lambda t: torch.full((t.shape[0], 1), 50.0))
    results = scorer.score_batch(paths)
    assert len(results) == len(paths)


def test_score_one(tmp_path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    p = tmp_path / "img.jpg"
    Image.new("RGB", (32, 32), color=(200, 150, 100)).save(p, "JPEG")
    scorer = _make_mock_scorer(raw_score=80.0, variant_id="musiq")
    result = scorer.score_one(p)
    assert "iqa_quality" in result
    assert 0.0 <= result["iqa_quality"] <= 1.0


def test_topiq_variant_passthrough(tmp_path: Path) -> None:
    """TOPIQ raw=0.85 should yield quality=0.85."""
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64)).save(img_path, "JPEG")

    scorer = _make_mock_scorer(raw_score=0.85, variant_id="topiq_nr")
    results = scorer.score_batch([img_path])
    assert results[0]["iqa_quality"] == pytest.approx(0.85, abs=1e-4)
