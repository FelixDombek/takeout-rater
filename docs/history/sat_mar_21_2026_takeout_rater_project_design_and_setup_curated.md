# takeout-rater — Project Design & Setup (2026-03-21, curated)

> Curated transcript. See the verbatim version for the full exchange.

---

## Decisions taken

| # | Topic | Decision |
|---|-------|----------|
| 1 | Metric storage key | Use **two-part key `(scorer_id, metric_key)`** in DB. In API/UI represent as `"scorer_id/metric_key"` for convenience. *(Single-string option discarded: relies on string-splitting conventions and is harder to index.)* |
| 2 | Registry discovery | **Explicit import list only** — adding a scorer requires a code change in `registry.py`. *(Dev-only folder scan discarded: less deterministic, problematic for packaging.)* |
| 3 | Build approach | **Iterative** — agent builds in milestones; Iteration 2 explicitly proves the "add a scorer" workflow. |
| 4 | Python version | **3.12** |
| 5 | License | **GPL** |
| 6 | Target repo | **FelixDombek/takeout-rater**, default branch `main` |

---

## Iteration plan

### Iteration 0 — Repo + agent-operable foundation
- `pyproject.toml` with ruff + pytest
- docs skeleton: `docs/architecture.md`, `docs/agents/how-to-add-a-scorer.md`, `docs/decisions/ADR-0001…`
- minimal CLI `takeout-rater --help`

*Goal: agents can run tests/lint and follow conventions.*

### Iteration 1 — Library indexing + DB + thumbnails (no ML)
- Select root dir containing `Takeout/`; create sibling `takeout-rater/` with `library.sqlite` + `thumbs/`
- Index images, parse `*.supplemental-metadata.json`, store `taken_at` / `google_photos_url` / relpath
- Thumbnail cache; minimal UI: list years/albums, grid of thumbs

*Goal: stable asset IDs and a browsable library.*

### Iteration 2 — Scorer framework + 1–2 scorers end-to-end *(key iteration)*
- Scorer interfaces + registry; DB tables for scorer runs + metric scores; basic background job runner
- Two initial scorers:
  1. **pHash scorer** — fast, no heavy deps
  2. **Aesthetic scorer** — first "real" ML adapter, multi-metric output

*Goal: prove "add scorer → registry → run → results stored → sort/filter".*

### Iteration 3 — Clustering + representative view + export
- pHash-based cluster builder
- UI: cluster view (rep + members)
- Export "top 1 per cluster by selected metric" to folder

*Goal: uniqueness + export workflow.*

---

## Scorer architecture

### File layout
```
takeout_rater/scorers/
  base.py                          # ScorerSpec, VariantSpec, MetricSpec, BaseScorer, ScoreResult
  registry.py                      # explicit import list; list_scorers(), get_spec(id)
  adapters/<name>/
    spec.py                        # exports SCORER_SPEC
    adapter.py                     # scoring logic
```

### "Add a scorer" workflow (3 steps)
1. Create `adapters/<name>/spec.py` + `adapter.py`
2. Add import to `registry.py`
3. Add tests: registry includes scorer id; scorer declares metrics + variants; running on one sample image yields metrics

---

## Repo bootstrap
User confirmed Python 3.12 + GPL. Bootstrap PR to add design doc, ADRs, agent docs, CI, license, and Python scaffolding was queued against `main`.
