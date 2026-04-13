"""Tests for the NIMAScorer adapter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from takeout_rater.scorers.adapters.nima import _VARIANT_PYIQA_METRIC, NIMAScorer

# ---------------------------------------------------------------------------
# Spec tests — no dependencies needed
# ---------------------------------------------------------------------------


def test_spec_scorer_id() -> None:
    assert NIMAScorer.spec().scorer_id == "nima"


def test_spec_has_nima_score_metric() -> None:
    spec = NIMAScorer.spec()
    assert len(spec.metrics) == 1
    assert spec.metrics[0].key == "nima_score"


def test_spec_range() -> None:
    m = NIMAScorer.spec().metrics[0]
    assert m.min_value == 1.0
    assert m.max_value == 10.0
    assert m.higher_is_better is True


def test_spec_has_aesthetic_and_technical_variants() -> None:
    spec = NIMAScorer.spec()
    variant_ids = {v.variant_id for v in spec.variants}
    assert "aesthetic" in variant_ids
    assert "technical" in variant_ids


def test_spec_default_variant_is_aesthetic() -> None:
    assert NIMAScorer.spec().default_variant_id == "aesthetic"


def test_spec_requires_no_extras() -> None:
    assert NIMAScorer.spec().requires_extras == ()


def test_spec_display_name_not_empty() -> None:
    spec = NIMAScorer.spec()
    assert spec.display_name
    assert spec.description


# ---------------------------------------------------------------------------
# Variant → pyiqa metric mapping
# ---------------------------------------------------------------------------


def test_variant_pyiqa_metric_covers_both_variants() -> None:
    assert "aesthetic" in _VARIANT_PYIQA_METRIC
    assert "technical" in _VARIANT_PYIQA_METRIC


def test_variant_pyiqa_metric_aesthetic_uses_ava() -> None:
    assert "ava" in _VARIANT_PYIQA_METRIC["aesthetic"]


def test_variant_pyiqa_metric_technical_uses_koniq() -> None:
    assert "koniq" in _VARIANT_PYIQA_METRIC["technical"]


# ---------------------------------------------------------------------------
# _ensure_loaded validates variant
# ---------------------------------------------------------------------------


def test_ensure_loaded_unknown_variant_raises() -> None:
    scorer = NIMAScorer.create(variant_id="aesthetic")
    scorer.variant_id = "unknown_variant"  # type: ignore[misc]
    with pytest.raises(ValueError, match="Unknown NIMA variant"):
        scorer._ensure_loaded()


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------


def test_is_available_returns_bool() -> None:
    result = NIMAScorer.is_available()
    assert isinstance(result, bool)


def test_is_available_false_when_pyiqa_missing() -> None:
    import builtins  # noqa: PLC0415

    real_import = builtins.__import__

    def _mock_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "pyiqa":
            raise ImportError("mocked missing pyiqa")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_mock_import):
        assert NIMAScorer.is_available() is False


# ---------------------------------------------------------------------------
# score_batch edge cases (no model required)
# ---------------------------------------------------------------------------


def test_score_batch_empty_returns_empty() -> None:
    scorer = NIMAScorer.create()
    assert scorer.score_batch([]) == []


def test_score_batch_missing_file_returns_one(tmp_path: Path) -> None:
    """A missing file should yield nima_score=1.0 (minimum), not raise."""
    scorer = NIMAScorer.create()
    # Inject a no-op pyiqa-style metric: returns score tensor of shape (N,)
    import torch  # noqa: PLC0415

    fake_metric = MagicMock()
    fake_metric.return_value = torch.tensor([5.5])
    scorer._model = fake_metric

    result = scorer.score_batch([tmp_path / "does_not_exist.jpg"])
    assert len(result) == 1
    assert result[0]["nima_score"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# score_batch with mocked model
# ---------------------------------------------------------------------------


def _make_mock_scorer(fixed_score: float = 7.0, variant_id: str = "aesthetic") -> NIMAScorer:
    """Return a NIMAScorer whose pyiqa metric returns a fixed score."""
    import torch  # noqa: PLC0415

    scorer = NIMAScorer.create(variant_id=variant_id)

    # Pyiqa-style metric: returns a (N,) tensor with each value = fixed_score.
    fake_metric = MagicMock()
    fake_metric.side_effect = lambda batch_tensor: torch.full((batch_tensor.shape[0],), fixed_score)

    scorer._model = fake_metric
    return scorer


def test_score_batch_returns_nima_score_key(tmp_path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64)).save(img_path, "JPEG")

    scorer = _make_mock_scorer()
    results = scorer.score_batch([img_path])
    assert len(results) == 1
    assert "nima_score" in results[0]


def test_score_batch_value_in_range(tmp_path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64)).save(img_path, "JPEG")

    scorer = _make_mock_scorer(fixed_score=7.0)
    results = scorer.score_batch([img_path])
    assert 1.0 <= results[0]["nima_score"] <= 10.0


def test_score_batch_expected_score(tmp_path: Path) -> None:
    """One-hot distribution on rating 7 should yield nima_score ≈ 7.0."""
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64)).save(img_path, "JPEG")

    scorer = _make_mock_scorer(fixed_score=7.0)
    results = scorer.score_batch([img_path])
    assert results[0]["nima_score"] == pytest.approx(7.0)


def test_score_batch_length_matches_input(tmp_path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    paths = []
    for i in range(4):
        p = tmp_path / f"img{i}.jpg"
        Image.new("RGB", (32, 32), color=(i * 60, 100, 200)).save(p, "JPEG")
        paths.append(p)

    scorer = _make_mock_scorer()
    results = scorer.score_batch(paths)
    assert len(results) == len(paths)


def test_score_batch_technical_variant(tmp_path: Path) -> None:
    """Technical variant uses a separate model load; spec is the same."""
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (64, 64)).save(img_path, "JPEG")

    scorer = _make_mock_scorer(fixed_score=5.0, variant_id="technical")
    assert scorer.variant_id == "technical"
    results = scorer.score_batch([img_path])
    assert "nima_score" in results[0]


def test_score_one(tmp_path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    p = tmp_path / "img.jpg"
    Image.new("RGB", (32, 32), color=(200, 150, 100)).save(p, "JPEG")
    scorer = _make_mock_scorer(fixed_score=6.0)
    result = scorer.score_one(p)
    assert "nima_score" in result
    assert 1.0 <= result["nima_score"] <= 10.0
