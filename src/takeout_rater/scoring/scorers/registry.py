"""Scorer registry: explicit import list and discovery helpers.

Add new scorers to ``_SCORER_CLASSES`` below.  The registry is intentionally
*explicit* (no dynamic plugin discovery) so that agents and humans can always
see at a glance which scorers are active.

See ``docs/agents/how-to-add-a-scorer.md`` for the step-by-step workflow.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Explicit scorer class list
# Add your scorer class here after creating it.
# ---------------------------------------------------------------------------
from takeout_rater.scoring.scorers.base import BaseScorer, ScorerSpec
from takeout_rater.scoring.scorers.brisque import BRISQUEScorer
from takeout_rater.scoring.scorers.cafe_style import CafeStyleScorer
from takeout_rater.scoring.scorers.clip_iqa import CLIPIQAScorer
from takeout_rater.scoring.scorers.laion import AestheticScorer
from takeout_rater.scoring.scorers.nima import NIMAScorer
from takeout_rater.scoring.scorers.nsfw import NSFWScorer
from takeout_rater.scoring.scorers.pyiqa_adapter import PyIQAScorer
from takeout_rater.scoring.scorers.simple import SimpleScorer

_SCORER_CLASSES: list[type[BaseScorer]] = [
    SimpleScorer,
    BRISQUEScorer,
    AestheticScorer,
    NSFWScorer,
    CLIPIQAScorer,
    NIMAScorer,
    PyIQAScorer,
    CafeStyleScorer,
]


def list_scorers(*, available_only: bool = False) -> list[type[BaseScorer]]:
    """Return scorer classes registered in this module.

    Args:
        available_only: If ``True``, filter to scorers whose
            :meth:`~BaseScorer.is_available` returns ``True``.

    Returns:
        List of scorer classes in registration order.
    """
    if available_only:
        return [cls for cls in _SCORER_CLASSES if cls.is_available()]
    return list(_SCORER_CLASSES)


def list_specs(*, available_only: bool = False) -> list[ScorerSpec]:
    """Return :class:`~takeout_rater.scoring.scorers.base.ScorerSpec` for each scorer."""
    return [cls.spec() for cls in list_scorers(available_only=available_only)]
