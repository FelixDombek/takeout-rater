"""Scorer base types: MetricSpec, VariantSpec, ScorerSpec, and BaseScorer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricSpec:
    """Describes one metric output produced by a scorer.

    Attributes:
        key: Machine-readable identifier stored in the DB (e.g. ``"aesthetic"``).
        display_name: Human-readable label shown in the UI.
        description: Short explanation of what the metric measures.
        min_value: Theoretical minimum (used for UI range sliders).
        max_value: Theoretical maximum (used for UI range sliders).
        higher_is_better: Whether larger values are considered better.
    """

    key: str
    display_name: str
    description: str = ""
    min_value: float = 0.0
    max_value: float = 1.0
    higher_is_better: bool = True


@dataclass(frozen=True)
class VariantSpec:
    """Describes one model/algorithm variant of a scorer.

    Different variants produce scores that are *not* directly comparable to
    each other, so each is stored under its own ``variant_id``.

    Attributes:
        variant_id: Stable identifier stored in ``scorer_runs`` (e.g. ``"laion_v2"``).
        display_name: Human-readable label for the UI.
        description: Short explanation of what distinguishes this variant.
    """

    variant_id: str
    display_name: str
    description: str = ""


@dataclass(frozen=True)
class ScorerSpec:
    """Full static description of a scorer: its identity, metrics, and variants.

    Attributes:
        scorer_id: Stable machine-readable identifier (e.g. ``"aesthetic"``).
        display_name: Human-readable name shown in the UI.
        description: Simplified, layman-friendly explanation shown by default in the UI.
        technical_description: Concise technical description for readers familiar with
            image-quality algorithms (e.g. cites the paper, algorithm, or metric formula).
            Shown when the user toggles to "Technical" mode on the Scoring page.
        version: Implementation version string (e.g. ``"1"``).  Increment this
            whenever the scoring algorithm changes in a way that would produce
            different scores for the same image, so that previously scored
            images can be identified and re-scored.
        metrics: Ordered list of metrics the scorer outputs.
        variants: Available model/algorithm variants (may be empty if only one).
        default_variant_id: The variant used when none is specified.
        requires_extras: Optional-dependency extras needed (e.g. ``["aesthetic"]``).
    """

    scorer_id: str
    display_name: str
    description: str = ""
    technical_description: str = ""
    version: str = "1"
    metrics: tuple[MetricSpec, ...] = field(default_factory=tuple)
    variants: tuple[VariantSpec, ...] = field(default_factory=tuple)
    default_variant_id: str = "default"
    requires_extras: tuple[str, ...] = field(default_factory=tuple)


class BaseScorer(ABC):
    """Abstract base class for all scorers.

    Subclasses must implement :meth:`spec`, :meth:`is_available`, and
    :meth:`score_batch`.  The class method :meth:`create` is provided as a
    convenience factory; override it if the scorer needs non-trivial
    initialisation (e.g. loading a model).

    Scorer instances are *stateful* (they may hold a loaded model), but they
    must be safe to use from a single thread at a time.

    Attributes:
        variant_id: The active variant for this instance.  Defaults to the
            spec's ``default_variant_id`` when ``None`` is passed.
    """

    def __init__(self, variant_id: str | None = None, **kwargs: Any) -> None:
        self.variant_id = variant_id or self.spec().default_variant_id
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Class-level API (introspection without instantiation)
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def spec(cls) -> ScorerSpec:
        """Return the static description of this scorer."""

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return ``True`` if all required dependencies are installed.

        This check must be *fast* (no model loading).  It is used at startup
        to determine which scorers can be offered to the user.
        """

    @classmethod
    def create(cls, variant_id: str | None = None, **kwargs: Any) -> BaseScorer:
        """Instantiate the scorer, optionally selecting a variant.

        The default implementation calls ``cls(variant_id=variant_id, **kwargs)``.
        Override when construction is more involved.
        """
        return cls(variant_id=variant_id, **kwargs)  # type: ignore[call-arg]

    # ------------------------------------------------------------------
    # Instance API
    # ------------------------------------------------------------------

    @abstractmethod
    def score_batch(
        self,
        image_paths: list[Path],
        *,
        variant_id: str | None = None,
    ) -> list[dict[str, float]]:
        """Score a batch of images.

        Args:
            image_paths: Absolute paths to image files.
            variant_id: Which variant to use; falls back to the spec default.

        Returns:
            A list (same length as ``image_paths``) of dicts mapping
            ``metric_key`` → ``float`` value.  Keys must be a subset of
            those declared in :attr:`ScorerSpec.metrics`.
        """

    def score_one(
        self,
        image_path: Path,
        *,
        variant_id: str | None = None,
    ) -> dict[str, float]:
        """Convenience wrapper: score a single image."""
        results = self.score_batch([image_path], variant_id=variant_id)
        return results[0]
