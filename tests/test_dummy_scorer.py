"""Tests for the DummyScorer."""

from __future__ import annotations

from pathlib import Path

import pytest

from takeout_rater.scorers.heuristics.dummy import DummyScorer


def test_is_available() -> None:
    assert DummyScorer.is_available() is True


def test_spec_scorer_id() -> None:
    assert DummyScorer.spec().scorer_id == "dummy"


def test_spec_has_metric() -> None:
    spec = DummyScorer.spec()
    assert len(spec.metrics) == 1
    assert spec.metrics[0].key == "dummy_score"


def test_spec_has_variant() -> None:
    spec = DummyScorer.spec()
    assert len(spec.variants) == 1
    assert spec.variants[0].variant_id == "default"


def test_score_batch_empty() -> None:
    scorer = DummyScorer.create()
    results = scorer.score_batch([])
    assert results == []


def test_score_batch_returns_correct_keys() -> None:
    scorer = DummyScorer.create()
    results = scorer.score_batch([Path("fake/image.jpg"), Path("fake/image2.jpg")])
    assert len(results) == 2
    for result in results:
        assert "dummy_score" in result
        assert result["dummy_score"] == pytest.approx(0.5)


def test_score_one() -> None:
    scorer = DummyScorer.create()
    result = scorer.score_one(Path("fake/image.jpg"))
    assert result["dummy_score"] == pytest.approx(0.5)


def test_score_batch_length_matches_input() -> None:
    scorer = DummyScorer.create()
    paths = [Path(f"image_{i}.jpg") for i in range(10)]
    results = scorer.score_batch(paths)
    assert len(results) == len(paths)
