"""Tests for the scorer registry."""

from __future__ import annotations

from takeout_rater.scorers.registry import list_scorers, list_specs


def test_list_scorers_returns_list() -> None:
    scorers = list_scorers()
    assert isinstance(scorers, list)


def test_list_scorers_non_empty() -> None:
    """At least one scorer must always be registered."""
    scorers = list_scorers()
    assert len(scorers) >= 1


def test_available_only_subset() -> None:
    all_scorers = list_scorers(available_only=False)
    available = list_scorers(available_only=True)
    # Available must be a subset of all
    assert set(available).issubset(set(all_scorers))


def test_list_specs_matches_list_scorers() -> None:
    specs = list_specs()
    scorers = list_scorers()
    assert len(specs) == len(scorers)


def test_all_specs_have_scorer_id() -> None:
    for spec in list_specs():
        assert spec.scorer_id, "Every ScorerSpec must have a non-empty scorer_id"


def test_all_specs_have_at_least_one_metric() -> None:
    for spec in list_specs():
        assert len(spec.metrics) >= 1, (
            f"Scorer '{spec.scorer_id}' must declare at least one MetricSpec"
        )
