# ADR-0003: Scorer plugin architecture

**Status:** Accepted  
**Date:** 2026-03

---

## Context

The tool needs to support multiple scoring algorithms:
- Aesthetic quality (ML, heavyweight)
- Perceptual hash (heuristic, lightweight)
- Blur / exposure (heuristic)
- Future: NSFW, face detection, composition, …

Each scorer may:
- Have optional heavy dependencies (PyTorch, ONNX Runtime, …)
- Produce multiple metric outputs per image
- Have multiple algorithm variants (different models, thresholds)
- Be unavailable on some installations

---

## Decision

Use an **adapter / spec pattern** with an **explicit import registry**:

### Core types (`takeout_rater/scorers/base.py`)
- `MetricSpec` — describes one output dimension (key, range, display name)
- `VariantSpec` — describes one model/algorithm variant
- `ScorerSpec` — full static description of a scorer
- `BaseScorer` — abstract class with `spec()`, `is_available()`, `create()`, `score_batch()`

### Registry (`takeout_rater/scorers/registry.py`)
- Scorers are listed **explicitly** in a `_SCORER_CLASSES` list.
- No dynamic discovery (no entry-points, no directory scanning).
- `list_scorers(available_only=False)` returns the registered classes.

### Variants
- Each `ScorerSpec` declares available `VariantSpec` objects.
- `variant_id` is stored per `scorer_run` row in the DB.
- Scores from different variants are **not** directly comparable.

### Multi-metric output
- `score_batch()` returns `list[dict[str, float]]` — multiple keys per image.
- Stored as separate `asset_scores` rows: one per `(asset, scorer_run, metric_key)`.

### Optional dependencies
- ~~Heavyweight deps were originally declared as optional extras in `pyproject.toml`.~~
- **Update (Iteration 10):** All scorer dependencies are now part of the base `[project] dependencies`.  Optional extras are reserved for exceptional cases only.
- `is_available()` checks imports without loading models; it continues to serve as a graceful-degradation mechanism should an import fail at runtime.

---

## Consequences

**Positive:**
- Explicit registry is always visible in code review; no magic activation.
- `is_available()` allows graceful degradation on minimal installs.
- Multi-metric design supports composite scorers without schema changes.
- Variants make model swaps auditable (old scores preserved under old variant_id).

**Negative:**
- Adding a scorer requires editing `registry.py` (intentional friction).
- No third-party scorer packages can self-register (acceptable for now).

---

## Alternatives considered

| Option | Rejected because |
|---|---|
| `importlib.metadata` entry-points | Third-party packages can inject scorers silently; harder to audit |
| Directory scanning | Order-dependent, fragile, hard for agents to reason about |
| Config-file scorer list | Extra indirection with no benefit over code |
