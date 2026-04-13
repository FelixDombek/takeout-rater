"""Backward-compatible shim: NoiseScorer has been merged into SimpleScorer.

Import :class:`~takeout_rater.scorers.heuristics.simple.SimpleScorer` directly
and use ``variant_id="noise"`` instead.
"""

from __future__ import annotations

from takeout_rater.scorers.heuristics.simple import SimpleScorer as NoiseScorer

__all__ = ["NoiseScorer"]
