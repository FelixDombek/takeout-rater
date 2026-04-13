"""Backward-compatible shim: LuminosityScorer has been merged into SimpleScorer.

Import :class:`~takeout_rater.scorers.heuristics.simple.SimpleScorer` directly
and use ``variant_id="luminosity"`` instead.
"""

from __future__ import annotations

from takeout_rater.scorers.heuristics.simple import SimpleScorer as LuminosityScorer

__all__ = ["LuminosityScorer"]
