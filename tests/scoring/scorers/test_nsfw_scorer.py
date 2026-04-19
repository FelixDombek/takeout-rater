"""Tests for the NSFW scorer."""

from __future__ import annotations

import pytest

from takeout_rater.scoring.scorers.base import ScorerSpec
from takeout_rater.scoring.scorers.nsfw import NSFWScorer

# ── spec ─────────────────────────────────────────────────────────────────────


def test_nsfw_scorer_spec_returns_scorer_spec() -> None:
    spec = NSFWScorer.spec()
    assert isinstance(spec, ScorerSpec)


def test_nsfw_scorer_id() -> None:
    assert NSFWScorer.spec().scorer_id == "nsfw"


def test_nsfw_scorer_has_nsfw_metric() -> None:
    spec = NSFWScorer.spec()
    keys = [m.key for m in spec.all_metrics()]
    assert "nsfw" in keys


def test_nsfw_metric_range() -> None:
    spec = NSFWScorer.spec()
    metric = next(m for m in spec.all_metrics() if m.key == "nsfw")
    assert metric.min_value == 0.0
    assert metric.max_value == 1.0


def test_nsfw_metric_lower_is_better() -> None:
    spec = NSFWScorer.spec()
    metric = next(m for m in spec.all_metrics() if m.key == "nsfw")
    assert metric.higher_is_better is False


def test_nsfw_scorer_has_variant() -> None:
    spec = NSFWScorer.spec()
    assert len(spec.variants) >= 1
    assert spec.default_variant_id == "falconsai_vit"


def test_nsfw_scorer_requires_no_extras() -> None:
    spec = NSFWScorer.spec()
    assert spec.requires_extras == ()


# ── is_available ──────────────────────────────────────────────────────────────


def test_nsfw_scorer_is_available_returns_bool() -> None:
    result = NSFWScorer.is_available()
    assert isinstance(result, bool)


# ── score_batch edge cases (no model required) ────────────────────────────────


def test_nsfw_scorer_score_batch_empty_returns_empty() -> None:
    scorer = NSFWScorer()
    # Does not need to load the model for an empty list
    result = scorer.score_batch([])
    assert result == []


# ── integration: score_batch with mocked pipeline ────────────────────────────


def test_nsfw_scorer_score_batch_with_mock_pipeline(tmp_path) -> None:
    """score_batch extracts the 'nsfw' label score from the pipeline output.

    The transformers pipeline is replaced with a lightweight fake so this test
    runs without downloading any model weights.
    """
    from PIL import Image

    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (64, 64), color=(200, 150, 100)).save(img_path, "JPEG")

    scorer = NSFWScorer()

    # Inject a fake pipeline that returns a fixed prediction (no network call).
    # The batched score_batch passes a list of images; return a list-of-lists.
    def fake_pipeline(imgs, batch_size=None):  # type: ignore[no-untyped-def]
        single_result = [{"label": "nsfw", "score": 0.1}, {"label": "normal", "score": 0.9}]
        if isinstance(imgs, list):
            return [single_result for _ in imgs]
        return single_result

    scorer._pipeline = fake_pipeline

    results = scorer.score_batch([img_path])
    assert len(results) == 1
    score = results[0]["nsfw"]
    assert score == pytest.approx(0.1)
