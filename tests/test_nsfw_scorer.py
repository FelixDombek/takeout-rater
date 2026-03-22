"""Tests for the NSFW scorer."""

from __future__ import annotations

import pytest

from takeout_rater.scorers.adapters.nsfw import NSFWScorer
from takeout_rater.scorers.base import ScorerSpec

# ── spec ─────────────────────────────────────────────────────────────────────


def test_nsfw_scorer_spec_returns_scorer_spec() -> None:
    spec = NSFWScorer.spec()
    assert isinstance(spec, ScorerSpec)


def test_nsfw_scorer_id() -> None:
    assert NSFWScorer.spec().scorer_id == "nsfw"


def test_nsfw_scorer_has_nsfw_metric() -> None:
    spec = NSFWScorer.spec()
    keys = [m.key for m in spec.metrics]
    assert "nsfw" in keys


def test_nsfw_metric_range() -> None:
    spec = NSFWScorer.spec()
    metric = next(m for m in spec.metrics if m.key == "nsfw")
    assert metric.min_value == 0.0
    assert metric.max_value == 1.0


def test_nsfw_metric_lower_is_better() -> None:
    spec = NSFWScorer.spec()
    metric = next(m for m in spec.metrics if m.key == "nsfw")
    assert metric.higher_is_better is False


def test_nsfw_scorer_has_variant() -> None:
    spec = NSFWScorer.spec()
    assert len(spec.variants) >= 1
    assert spec.default_variant_id == "falconsai_vit"


def test_nsfw_scorer_requires_nsfw_extra() -> None:
    spec = NSFWScorer.spec()
    assert "nsfw" in spec.requires_extras


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


# ── integration: skip if deps not available ───────────────────────────────────


@pytest.mark.skipif(not NSFWScorer.is_available(), reason="nsfw deps not installed")
def test_nsfw_scorer_scores_real_image(tmp_path) -> None:
    pytest.importorskip("PIL")
    from PIL import Image  # noqa: PLC0415

    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (64, 64), color=(200, 150, 100)).save(img_path, "JPEG")

    scorer = NSFWScorer()
    results = scorer.score_batch([img_path])
    assert len(results) == 1
    score = results[0]["nsfw"]
    assert 0.0 <= score <= 1.0
