# How to add a scorer

Follow these steps to add a new scorer to `takeout-rater`.

---

## 1. Decide: heuristic or adapter?

| Type | Use when | Location |
|---|---|---|
| **Heuristic** | Score derived from image properties only (no ML model) | `src/takeout_rater/scorers/heuristics/` |
| **Adapter** | Wraps an ML model or external tool | `src/takeout_rater/scorers/adapters/<name>/` |

---

## 2. Create the scorer module

### File template

```python
"""<Short description of what this scorer measures>."""

from __future__ import annotations

from pathlib import Path

from takeout_rater.scorers.base import BaseScorer, MetricSpec, ScorerSpec, VariantSpec


class MyScorer(BaseScorer):
    """One-line description."""

    @classmethod
    def spec(cls) -> ScorerSpec:
        return ScorerSpec(
            scorer_id="my_scorer",          # stable, lowercase, underscore-separated
            display_name="My Scorer",
            description="What this scorer measures and how.",
            metrics=(
                MetricSpec(
                    key="my_metric",
                    display_name="My metric",
                    description="What this number means.",
                    min_value=0.0,
                    max_value=10.0,
                    higher_is_better=True,
                ),
                # Add more MetricSpec entries for multi-metric scorers
            ),
            variants=(
                VariantSpec(
                    variant_id="v1",
                    display_name="Version 1",
                    description="Initial model / algorithm.",
                ),
            ),
            default_variant_id="v1",
            requires_extras=("my_extra",),  # matches pyproject.toml optional extra name
        )

    @classmethod
    def is_available(cls) -> bool:
        try:
            import some_optional_dep  # noqa: F401
            return True
        except ImportError:
            return False

    def score_batch(
        self,
        image_paths: list[Path],
        *,
        variant_id: str | None = None,
    ) -> list[dict[str, float]]:
        results = []
        for path in image_paths:
            # TODO: compute actual score
            results.append({"my_metric": 0.0})
        return results
```

---

## 3. Declare optional dependencies (if any)

In `pyproject.toml`, add an entry under `[project.optional-dependencies]`:

```toml
my_extra = [
    "some-optional-dep>=1.0",
]
```

---

## 4. Register the scorer

Open `src/takeout_rater/scorers/registry.py` and add:

```python
from takeout_rater.scorers.heuristics.my_scorer import MyScorer  # or adapters/…

_SCORER_CLASSES: list[type[BaseScorer]] = [
    DummyScorer,
    MyScorer,   # ← add here
]
```

---

## 5. Write tests

Create `tests/test_my_scorer.py`.  At minimum test:
- `MyScorer.spec().scorer_id == "my_scorer"`
- `score_batch([])` returns `[]`
- `score_batch([some_path])` returns a list of length 1 with the expected key(s)
- `is_available()` returns a bool (don't assert True/False — it may depend on the environment)

---

## 6. Check the Definition of Done

See `docs/agents/definition-of-done.md` before opening a PR.

---

## Notes

- `scorer_id` must be **stable** — it is stored in the DB.  Changing it will orphan existing scores.
- `variant_id` is also stored per `scorer_run`.  Add a new variant rather than renaming an existing one when upgrading a model.
- Multi-metric scorers should document what each metric measures in the `MetricSpec.description`.
