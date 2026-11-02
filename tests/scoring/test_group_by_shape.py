"""Tests for grouping mixed-size thumbnail tensors into stackable batches."""

from __future__ import annotations

from types import SimpleNamespace

from takeout_rater.scoring.scorers.base import group_indices_by_shape


def _t(*shape: int) -> SimpleNamespace:
    return SimpleNamespace(shape=shape)


def test_groups_by_shape_preserving_order() -> None:
    tensors = [_t(3, 384, 512), _t(3, 512, 384), None, _t(3, 384, 512), _t(3, 512, 384)]
    assert group_indices_by_shape(tensors) == [[0, 3], [1, 4]]


def test_all_same_shape_is_one_group() -> None:
    tensors = [_t(3, 384, 512) for _ in range(4)]
    assert group_indices_by_shape(tensors) == [[0, 1, 2, 3]]


def test_empty_and_all_failed() -> None:
    assert group_indices_by_shape([]) == []
    assert group_indices_by_shape([None, None]) == []
