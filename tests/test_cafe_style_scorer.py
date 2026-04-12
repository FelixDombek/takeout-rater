"""Tests for the CafeStyleScorer adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from takeout_rater.scorers.adapters.cafe_style import (
    _LABEL_TO_METRIC,
    CafeStyleScorer,
    _preds_to_scores,
)
from takeout_rater.scorers.base import ScorerSpec

# ---------------------------------------------------------------------------
# Expected metric keys
# ---------------------------------------------------------------------------

_EXPECTED_METRIC_KEYS = {
    "style_photo",
    "style_anime",
    "style_illustration",
    "style_3d",
    "style_cgi",
}

# ---------------------------------------------------------------------------
# Spec tests — no dependencies needed
# ---------------------------------------------------------------------------


def test_spec_returns_scorer_spec() -> None:
    spec = CafeStyleScorer.spec()
    assert isinstance(spec, ScorerSpec)


def test_spec_scorer_id() -> None:
    assert CafeStyleScorer.spec().scorer_id == "cafe_style"


def test_spec_has_all_metrics() -> None:
    keys = {m.key for m in CafeStyleScorer.spec().metrics}
    assert keys == _EXPECTED_METRIC_KEYS


def test_spec_all_metrics_are_probabilities() -> None:
    for m in CafeStyleScorer.spec().metrics:
        assert m.min_value == 0.0
        assert m.max_value == 1.0


def test_spec_all_metrics_higher_is_better() -> None:
    for m in CafeStyleScorer.spec().metrics:
        assert m.higher_is_better is True


def test_spec_has_variant() -> None:
    spec = CafeStyleScorer.spec()
    assert any(v.variant_id == "cafeai_v1" for v in spec.variants)
    assert spec.default_variant_id == "cafeai_v1"


def test_spec_requires_no_extras() -> None:
    assert CafeStyleScorer.spec().requires_extras == ()


def test_spec_display_name_and_descriptions_not_empty() -> None:
    spec = CafeStyleScorer.spec()
    assert spec.display_name
    assert spec.description
    assert spec.technical_description


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------


def test_is_available_returns_bool() -> None:
    result = CafeStyleScorer.is_available()
    assert isinstance(result, bool)


def test_is_available_false_when_transformers_missing() -> None:
    import builtins  # noqa: PLC0415

    real_import = builtins.__import__

    def _mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "transformers":
            raise ImportError("mocked missing transformers")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_mock_import):
        assert CafeStyleScorer.is_available() is False


# ---------------------------------------------------------------------------
# _preds_to_scores unit tests (pure function, no model needed)
# ---------------------------------------------------------------------------


def test_preds_to_scores_all_labels() -> None:
    preds = [
        {"label": "photo", "score": 0.7},
        {"label": "anime", "score": 0.1},
        {"label": "illustration", "score": 0.1},
        {"label": "3d", "score": 0.05},
        {"label": "CGI", "score": 0.05},
    ]
    scores = _preds_to_scores(preds)
    assert scores["style_photo"] == pytest.approx(0.7)
    assert scores["style_anime"] == pytest.approx(0.1)
    assert scores["style_illustration"] == pytest.approx(0.1)
    assert scores["style_3d"] == pytest.approx(0.05)
    assert scores["style_cgi"] == pytest.approx(0.05)


def test_preds_to_scores_label_case_insensitive() -> None:
    """Labels from the pipeline may be mixed case; they should be lowercased."""
    preds = [{"label": "CGI", "score": 0.9}]
    scores = _preds_to_scores(preds)
    assert scores["style_cgi"] == pytest.approx(0.9)


def test_preds_to_scores_unknown_label_ignored() -> None:
    preds = [{"label": "unknown_category", "score": 0.9}]
    scores = _preds_to_scores(preds)
    # All known keys should default to 0.0
    assert all(v == 0.0 for v in scores.values())


def test_preds_to_scores_missing_labels_default_to_zero() -> None:
    """If a label is absent from the pipeline output it defaults to 0.0."""
    preds = [{"label": "photo", "score": 0.95}]
    scores = _preds_to_scores(preds)
    assert scores["style_photo"] == pytest.approx(0.95)
    assert scores["style_anime"] == 0.0
    assert scores["style_illustration"] == 0.0
    assert scores["style_3d"] == 0.0
    assert scores["style_cgi"] == 0.0


def test_preds_to_scores_returns_all_expected_keys() -> None:
    scores = _preds_to_scores([])
    assert set(scores.keys()) == _EXPECTED_METRIC_KEYS


# ---------------------------------------------------------------------------
# Label-to-metric mapping coverage
# ---------------------------------------------------------------------------


def test_label_to_metric_covers_all_expected_metrics() -> None:
    """Every expected metric key must be reachable via _LABEL_TO_METRIC."""
    assert set(_LABEL_TO_METRIC.values()) == _EXPECTED_METRIC_KEYS


# ---------------------------------------------------------------------------
# score_batch edge cases (no model required)
# ---------------------------------------------------------------------------


def test_score_batch_empty_returns_empty() -> None:
    scorer = CafeStyleScorer()
    assert scorer.score_batch([]) == []


# ---------------------------------------------------------------------------
# score_batch with mocked pipeline
# ---------------------------------------------------------------------------


def _make_mock_scorer(photo_prob: float = 0.8) -> CafeStyleScorer:
    """Return a CafeStyleScorer with a lightweight fake pipeline injected."""
    scorer = CafeStyleScorer()

    remaining = (1.0 - photo_prob) / 4.0

    def fake_pipeline(imgs: Any, batch_size: int | None = None) -> list[list[dict[str, Any]]]:
        single_result = [
            {"label": "photo", "score": photo_prob},
            {"label": "anime", "score": remaining},
            {"label": "illustration", "score": remaining},
            {"label": "3d", "score": remaining},
            {"label": "CGI", "score": remaining},
        ]
        if isinstance(imgs, list):
            return [single_result for _ in imgs]
        return single_result

    scorer._pipeline = fake_pipeline
    return scorer


def test_score_batch_returns_all_metric_keys(tmp_path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (64, 64), color=(200, 150, 100)).save(img_path, "JPEG")

    scorer = _make_mock_scorer()
    results = scorer.score_batch([img_path])
    assert len(results) == 1
    assert set(results[0].keys()) == _EXPECTED_METRIC_KEYS


def test_score_batch_photo_probability_correct(tmp_path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (64, 64)).save(img_path, "JPEG")

    scorer = _make_mock_scorer(photo_prob=0.8)
    results = scorer.score_batch([img_path])
    assert results[0]["style_photo"] == pytest.approx(0.8)


def test_score_batch_all_values_in_range(tmp_path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (64, 64)).save(img_path, "JPEG")

    scorer = _make_mock_scorer()
    results = scorer.score_batch([img_path])
    for v in results[0].values():
        assert 0.0 <= v <= 1.0


def test_score_batch_missing_file_returns_zeros(tmp_path: Path) -> None:
    scorer = _make_mock_scorer()
    results = scorer.score_batch([tmp_path / "does_not_exist.jpg"])
    assert len(results) == 1
    assert all(v == 0.0 for v in results[0].values())


def test_score_batch_length_matches_input(tmp_path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    paths = []
    for i in range(3):
        p = tmp_path / f"img{i}.jpg"
        Image.new("RGB", (32, 32), color=(i * 80, 100, 200)).save(p, "JPEG")
        paths.append(p)

    scorer = _make_mock_scorer()
    results = scorer.score_batch(paths)
    assert len(results) == len(paths)


def test_score_one(tmp_path: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    p = tmp_path / "img.jpg"
    Image.new("RGB", (32, 32), color=(200, 150, 100)).save(p, "JPEG")
    scorer = _make_mock_scorer(photo_prob=0.9)
    result = scorer.score_one(p)
    assert set(result.keys()) == _EXPECTED_METRIC_KEYS
    assert result["style_photo"] == pytest.approx(0.9)
