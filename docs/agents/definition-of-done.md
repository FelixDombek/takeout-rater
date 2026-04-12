# Definition of Done

A contribution is complete when **all** of the following are true.

---

## Code quality

- [ ] `ruff format --check` passes (no formatting changes needed)
- [ ] `ruff check` passes (no lint errors)
- [ ] All new code has type annotations (PEP 526 / PEP 604 style — use `str | None`, not `Optional[str]`)

## Tests

- [ ] `pytest` passes with no failures
- [ ] New behaviour has at least one test covering the happy path
- [ ] Edge cases (empty input, missing optional dep, …) have tests where non-obvious

## Documentation

- [ ] Public functions and classes have docstrings
- [ ] If a significant architectural decision was made, an ADR exists in `docs/decisions/`
- [ ] `docs/agents/repo-map.md` is updated if a new module or directory was added

## Scorer-specific additions

- [ ] New scorer is listed in `takeout_rater/scorers/registry.py`
- [ ] New scorer follows the workflow in `docs/agents/how-to-add-a-scorer.md`
- [ ] Dependencies are added to `pyproject.toml` base `[project] dependencies`

## PR hygiene

- [ ] PR title is descriptive (e.g. `feat: add LaionAestheticScorer`)
- [ ] PR description references the relevant issue or iteration
- [ ] No debug print statements, commented-out code, or TODO without an issue reference
- [ ] **UI changes** include at least one screenshot in the PR description
