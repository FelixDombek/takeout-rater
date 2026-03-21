# Contributing to takeout-rater

Thank you for contributing!  This project is designed to be worked on largely
by GitHub Copilot Agents, so keeping decisions explicit and discoverable is
especially important.

---

## Quick start

```bash
git clone https://github.com/FelixDombek/takeout-rater.git
cd takeout-rater
pip install -e ".[dev]"
```

---

## Key conventions

### Style

- Python 3.12+, PEP 604 union types (`str | None`, not `Optional[str]`)
- Formatter: `ruff format` (line length 100)
- Linter: `ruff check`
- All public functions/classes must have docstrings

### Tests

- Test files: `tests/test_<module>.py`
- Run: `pytest`
- New behaviour must have tests; see `docs/agents/definition-of-done.md`

### Architectural decisions

Before making a significant design choice, check `docs/decisions/` for an
existing ADR.  If your change overrides or extends a decision, either update
the ADR or create a new one.

ADR template:
```
# ADR-NNNN: <short title>

**Status:** Proposed | Accepted | Superseded by ADR-XXXX
**Date:** YYYY-MM

## Context
## Decision
## Consequences
## Alternatives considered
```

### Adding a scorer

Follow `docs/agents/how-to-add-a-scorer.md` step-by-step.

### Database changes

Follow `docs/agents/db-guidelines.md`.  Every schema change needs a numbered
migration script.

---

## Branch and PR conventions

| Convention | Example |
|---|---|
| Branch name | `feat/laion-aesthetic-scorer`, `fix/thumbnail-cache-path` |
| PR title | `feat: add LaionAestheticScorer`, `fix: handle missing sidecar` |
| PR description | Reference the issue or iteration; tick the Definition of Done checklist |

---

## Definition of Done

See [`docs/agents/definition-of-done.md`](docs/agents/definition-of-done.md).

---

## License

By contributing, you agree that your contributions will be licensed under
GPL-3.0-only, the same license as this project.
