"""Backward-compatible shim: BlurScorer has been merged into SimpleScorer.

Import :class:`~takeout_rater.scorers.heuristics.simple.SimpleScorer` directly
and use ``variant_id="blur"`` instead.
"""

from __future__ import annotations

from takeout_rater.scorers.heuristics.simple import SimpleScorer as BlurScorer

__all__ = ["BlurScorer"]
