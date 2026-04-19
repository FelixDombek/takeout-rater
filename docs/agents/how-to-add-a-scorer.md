# How to add a scorer

Follow these steps to add a new scorer to `takeout-rater`.

---

## 1. Choose the module path

Place new scorers directly in `src/takeout_rater/scorers/<name>.py`.  Use a
single `.py` file unless the scorer spans multiple source files.

---

## 2. Implement the scorer

Use this template for scorers that derive metrics from image data, whether the
implementation uses Pillow statistics, a pretrained model, or an external
library. See `simple.py`, `nsfw.py`, and `laion.py` for working examples.

```python
"""<Short description of what this scorer measures>."""

from __future__ import annotations

from pathlib import Path

from takeout_rater.scoring.scorers.base import BaseScorer, MetricSpec, ScorerSpec, VariantSpec


class MyScorer(BaseScorer):
    """One-line description."""

    @classmethod
    def spec(cls) -> ScorerSpec:
        return ScorerSpec(
            scorer_id="my_scorer",          # stable, lowercase, underscore-separated
            display_name="My Scorer",
            description="What this scorer measures and how.",
            variants=(
                VariantSpec(
                    variant_id="default",
                    display_name="Default",
                    description="Algorithm description.",
                    metrics=(
                        MetricSpec(
                            key="my_metric",
                            display_name="My metric",
                            description="What this number means.",
                            min_value=0.0,
                            max_value=100.0,
                            higher_is_better=True,  # set False when lower is better
                        ),
                    ),
                ),
            ),
            default_variant_id="default",
            requires_extras=(),
        )

    @classmethod
    def is_available(cls) -> bool:
        try:
            from PIL import Image  # noqa: F401
            return True
        except ImportError:
            return False

    def score_batch(
        self,
        image_paths: list[Path],
        *,
        variant_id: str | None = None,
    ) -> list[dict[str, float]]:
        """Score a batch of images.

        Returns:
            List (same length as *image_paths*) of dicts mapping metric key → float.
            On per-image errors, return a safe default (e.g. ``0.0``) rather than raising.
        """
        from PIL import Image  # noqa: PLC0415

        results: list[dict[str, float]] = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                score = 0.0  # TODO: compute actual score
            except (OSError, ValueError):
                score = 0.0
            results.append({"my_metric": score})
        return results
```

For model-backed scorers:
- `__init__` stores lazy-load placeholders so the model is only loaded on
  first use.
- `_ensure_loaded()` does the actual loading (downloads weights on first run).
- `score_batch` guards against an empty list **before** triggering a load.

```python
"""<Short description of what this scorer measures>."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from takeout_rater.scoring.scorers.base import BaseScorer, MetricSpec, ScorerSpec, VariantSpec

#: HuggingFace / other model identifier
_MODEL_ID = "org/model-name"


class MyMLScorer(BaseScorer):
    """One-line description."""

    def __init__(self, variant_id: str | None = None, **kwargs: Any) -> None:
        super().__init__(variant_id=variant_id, **kwargs)
        # Lazy-loaded — populated by _ensure_loaded()
        self._model: Any = None

    @classmethod
    def spec(cls) -> ScorerSpec:
        return ScorerSpec(
            scorer_id="my_ml_scorer",
            display_name="My ML Scorer",
            description="What this scorer measures and how.",
            variants=(
                VariantSpec(
                    variant_id="v1",
                    display_name="Model v1",
                    description="Which checkpoint / version this is.",
                    metrics=(
                        MetricSpec(
                            key="my_metric",
                            display_name="My metric",
                            description="What this number means.",
                            min_value=0.0,
                            max_value=1.0,
                            higher_is_better=True,  # set False when lower is better
                        ),
                    ),
                ),
            ),
            default_variant_id="v1",
            requires_extras=(),  # base install; use extras only in exceptional cases
        )

    @classmethod
    def is_available(cls) -> bool:
        """Return True when all optional runtime deps are importable (no model load)."""
        try:
            import some_optional_dep  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_loaded(self) -> None:
        """Load the model on first call (lazy init).

        Downloads weights to the standard cache if not already present.
        """
        if self._model is not None:
            return
        import some_optional_dep  # noqa: PLC0415
        self._model = some_optional_dep.load(_MODEL_ID)

    def score_batch(
        self,
        image_paths: list[Path],
        *,
        variant_id: str | None = None,
    ) -> list[dict[str, float]]:
        """Score a batch of images.

        Returns:
            List (same length as *image_paths*) of dicts mapping metric key → float.
            On per-image errors, return a safe default (e.g. ``0.0``) rather than raising.
        """
        if not image_paths:
            return []

        from PIL import Image  # noqa: PLC0415

        self._ensure_loaded()

        results: list[dict[str, float]] = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                score = float(self._model(img))  # TODO: adapt to your model's API
            except (OSError, ValueError, RuntimeError):
                score = 0.0
            results.append({"my_metric": score})
        return results
```

---

## 3. Add dependencies to the base install

New scorers should have their dependencies added to the `[project] dependencies`
list in `pyproject.toml`.  Every feature is part of the base install — optional
extras are reserved for exceptional cases only.

```toml
[project]
dependencies = [
    ...
    "some-new-dep>=1.0",
]
```

---

## 4. Register the scorer

Open `src/takeout_rater/scorers/registry.py` and add:

```python
from takeout_rater.scoring.scorers.my_ml_scorer import MyMLScorer

_SCORER_CLASSES: list[type[BaseScorer]] = [
    # ...existing scorers...
    MyMLScorer,   # ← add here
]
```

---

## 5. Write tests

Create `tests/test_my_scorer.py`.  At minimum test:

- `MyScorer.spec().scorer_id == "my_scorer"`
- `MyScorer.spec().metrics_for_variant("default")[0].higher_is_better` has the correct value
- `MyScorer.spec()` has at least one variant and a `default_variant_id`
- `is_available()` returns a `bool` (don't assert `True`/`False` — depends on the environment)
- `score_batch([])` returns `[]`
- A missing/unreadable file yields the metric key with a safe default value (e.g. `0.0`), not an exception

For model-backed scorers, gate integration tests with:

```python
@pytest.mark.skipif(not MyMLScorer.is_available(), reason="deps not installed")
def test_scores_real_image(tmp_path): ...
```

See `tests/test_simple_scorer.py` and `tests/test_nsfw_scorer.py` for examples.

---

## 6. Check the Definition of Done

See `docs/agents/definition-of-done.md` before opening a PR.

---

## Notes

- `scorer_id` must be **stable** — it is stored in the DB.  Changing it will orphan existing scores.
- `variant_id` is also stored per `asset_scores` row.  Add a new variant rather than renaming an existing one when upgrading a model.
- Metrics belong on the `VariantSpec` that produces them.  The persisted hierarchy is scorer → variant → metric.
- `higher_is_better=False` for metrics where a *lower* value is better (e.g. NSFW probability, noise level).
- Multi-metric scorers should document what each metric measures in the `MetricSpec.description`.
- `score_batch` must **always** return a list of the same length as `image_paths`, even when individual images fail.  Use a safe default (typically `0.0`) and catch `(OSError, ValueError, RuntimeError)`.
- Scorers are run via the scoring pipeline (`takeout_rater.scoring.pipeline.run_scorer`).  They do not need to know about DB internals.
