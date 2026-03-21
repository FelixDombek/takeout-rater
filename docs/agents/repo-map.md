# Repository map

Quick reference for where things live and what they do.

---

## Top-level layout

```
takeout-rater/
├── src/takeout_rater/   ← Python package (src layout)
├── tests/               ← pytest tests
├── docs/
│   ├── design.md        ← architecture overview
│   ├── decisions/       ← ADRs (numbered markdown files)
│   └── agents/          ← agent enablement docs (this directory)
├── .github/workflows/   ← CI (ruff + pytest)
├── pyproject.toml       ← PEP 621 project config, deps, tool config
├── README.md
├── CONTRIBUTING.md
└── LICENSE              ← GPL-3.0-only
```

---

## Python package: `src/takeout_rater/`

| Module / Package | Purpose |
|---|---|
| `__init__.py` | Package version |
| `cli.py` | `takeout-rater` CLI entry-point (`python -m takeout_rater`) |
| `scorers/` | Scorer interface, registry, and implementations |
| `scorers/base.py` | `MetricSpec`, `VariantSpec`, `ScorerSpec`, `BaseScorer` |
| `scorers/registry.py` | Explicit scorer class list + `list_scorers()` |
| `scorers/heuristics/` | Lightweight heuristic scorers (no ML model needed) |
| `scorers/heuristics/dummy.py` | Trivial constant scorer — used in tests |
| `scorers/adapters/` | ML/external-tool scorer wrappers (optional deps) |
| `indexing/` | *(Iteration 1)* Takeout scanner, sidecar parser, thumbnail generator |
| `db/` | *(Iteration 1)* SQLite schema, migrations, query helpers |
| `api/` | *(Iteration 1)* FastAPI routes |
| `ui/` | *(Iteration 1)* Jinja2 templates + static assets |

---

## Where to add things

| Task | Location |
|---|---|
| New heuristic scorer | `src/takeout_rater/scorers/heuristics/<name>.py` + entry in `registry.py` |
| New ML/adapter scorer | `src/takeout_rater/scorers/adapters/<name>/` + entry in `registry.py` |
| New CLI sub-command | `src/takeout_rater/cli.py` |
| New API route | `src/takeout_rater/api/<router>.py` (Iteration 1+) |
| New DB migration | `src/takeout_rater/db/migrations/` (Iteration 1+) |
| New ADR | `docs/decisions/ADR-NNNN-<slug>.md` |
| New agent doc | `docs/agents/<topic>.md` |
| New test | `tests/test_<module>.py` |

---

## Key design documents

| Document | What it covers |
|---|---|
| `docs/design.md` | Full architecture, data model, iteration plan |
| `docs/decisions/ADR-0001-local-web-ui-fastapi.md` | Why FastAPI + HTMX |
| `docs/decisions/ADR-0002-sqlite-metadata-store.md` | DB schema philosophy |
| `docs/decisions/ADR-0003-scorer-plugin-architecture.md` | Scorer interface + explicit registry |
| `docs/decisions/ADR-0004-library-state-location.md` | Where `takeout-rater/` lives |
| `docs/decisions/ADR-0005-v1-filtering-sorting-no-dsl.md` | Structured filters in v1 |
