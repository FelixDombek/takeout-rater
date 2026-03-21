User: 1. The initial version will be built in iterations. The workflow to add at least 1-2 new scorers should be part of one of these iterations. The agent won't be able to build the whole app in one go.

2. I don't understand the difference between "xxx scoped by scorer" and "scorer.xxx"

A: ### 2) "scoped by scorer" vs "scorer.xxx" (what it means in practice)

You're trying to uniquely identify a metric like "overall aesthetic score", but multiple scorers can each have an "overall" metric. There are two equivalent ways to name/store that.

#### Option A — **Scoped by scorer (recommended)**: `(scorer_id, metric_key)`
You store *two separate fields*:
- `scorer_id = "rsinema_aesthetic"`
- `metric_key = "overall"`
- `value = 7.3`

This is like a two-part key. In the UI you show it as:  
- **rsinema_aesthetic / overall**

**Pros**
- Avoids naming collisions cleanly.
- Easy to query and index in SQL (`WHERE scorer_id=? AND metric_key=?`).
- Doesn't require inventing string conventions.

**Cons**
- When you serialize to JSON you carry two fields instead of one.

#### Option B — **Fully qualified metric key**: `"scorer_id.metric_key"`
You store a *single string*:
- `metric = "rsinema_aesthetic.overall"`
- `value = 7.3`

**Pros**
- Simple to pass around as one string (handy for API/UI).

**Cons**
- You now rely on string parsing/splitting rules.
- Harder to enforce consistency (typos become "new metrics").
- Slightly less clean DB indexing (still doable, just not as nice).

**Bottom line:** Use **Option A in the DB** (scorer_id + metric_key), and in API/UI you can still represent a "metric selector" as a single string like `"rsinema_aesthetic/overall"` or `"rsinema_aesthetic.overall"` for convenience.

---

## 1) Iterative plan: include "add a scorer workflow" early

Agreed—the agent shouldn't attempt the whole app at once. You want early iterations that establish the *interfaces and scaffolding* so adding scorers is routine.

Here's a practical milestone breakdown where **Iteration 2** explicitly proves "add 1–2 scorers" as a repeatable workflow.

### Iteration 0 — Repo + agent-operable foundation (no real features yet)
Deliverables:
- `pyproject.toml` with ruff + pytest (and maybe mypy later)
- docs skeleton:
  - `docs/architecture.md`
  - `docs/agents/how-to-add-a-scorer.md`
  - `docs/decisions/ADR-0001...`
- minimal CLI `takeout-rater --help`

Goal: agents can run tests/lint and follow conventions.

### Iteration 1 — Library indexing + DB + thumbnails (no ML)
Deliverables:
- Select a root dir containing `Takeout/`
- Create sibling `takeout-rater/` directory with:
  - `library.sqlite`
  - `thumbs/`
- Index:
  - find images + parse `*.supplemental-metadata.json`
  - store `taken_at`, `google_photos_url`, relpath
- Thumbnail generator (cache)
- Minimal UI/API: list years/albums, show a grid of thumbs

Goal: you can browse the library and have stable asset IDs.

### Iteration 2 — Scorer framework + add 1–2 scorers end-to-end (this is the key iteration you asked for)
Deliverables:
- Scorer interfaces + registry
- DB tables for scorer runs + metric scores
- Background job runner (even if basic)
- Implement + integrate two scorers:
  1) **pHash scorer** (fast, no heavy deps) → metric like `phash` (or store in a dedicated table)
  2) **Aesthetic scorer (first "real" ML adapter)** → multi-metric output supported

Goal: prove the workflow "add scorer → appears in registry → can run → results stored → sort/filter works".

### Iteration 3 — Clustering + "representative-only" view + export
Deliverables:
- cluster builder using pHash
- UI: cluster view, show rep + members
- export "top 1 per cluster by selected metric" to folder

Goal: your uniqueness + export workflow works.

---

## Concrete v1 scorer discovery/layout (what the agent will implement)

### Files
- `takeout_rater/scorers/base.py`
  - `ScorerSpec`, `VariantSpec`, `MetricSpec`, `BaseScorer`, `ScoreResult`
- `takeout_rater/scorers/registry.py`
  - explicit list of scorer specs
  - `list_scorers()`, `get_spec(id)`
- `takeout_rater/scorers/adapters/<name>/spec.py`
  - exports `SCORER_SPEC`
- `takeout_rater/scorers/adapters/<name>/adapter.py`
  - the actual scoring logic

### What "add a scorer" means (repeatable steps)
1. Create `takeout_rater/scorers/adapters/newthing/spec.py` + `adapter.py`
2. Add import to `registry.py`
3. Add tests:
   - registry includes scorer id
   - scorer declares metrics + variants
   - running on one sample image yields metrics

That's it. No magic scanning, no packaging surprises.

---

## One design decision to make now (small but important)

For the registry, do you want:
- **"explicit import list" only** (most deterministic), or
- explicit list + optional "dev-only scan of adapters folder"?

Given you want agents + eventual packaging: I recommend **explicit import list only** for v1.

If you confirm that, the next step is to write the ADR for scorer architecture and a template for `spec.py`/`adapter.py` so adding scorers becomes mechanical.

User: I agree with the recommendation (scorer, metric, value).

Explicit import list (code change to add scorer) is totally fine.

Great. At this point, I think we should follow your implementation plan and put the whole solution design into a design document, and let an agent create the initial state of the repo.

A: To proceed I need one missing piece: **which GitHub repository (owner/name) should I create the initial repo state in?** You haven't provided a repo yet, and I'm not allowed to infer it.

Reply with one of these:

1) **Create in an existing repo**: give `owner/repo` (and default branch name if not `main`)  
2) **You will create the empty repo first** and tell me `owner/repo`  
3) If you don't want GitHub changes yet, I can still draft the design doc + repo skeleton files here for you to paste.

Once you provide `owner/repo`, I will:
- write `docs/design.md` capturing the full solution design + iteration plan
- add initial ADRs in `docs/decisions/`
- add agent-enablement docs in `docs/agents/`
- add a Python project skeleton (`pyproject.toml`, `src/takeout_rater/...`)
- add CI (ruff + pytest)
- include templates for adding scorers (spec/adapter + tests)

Confirm also:
- Python version target (recommend **3.11** or **3.12**)  
- License (MIT/Apache-2.0/other) or "leave unlicensed for now"

User: 3.12. GPL. 

A: We're waiting for your confirmation in the GitHub dialog (approve/deny) before I can proceed to create the PR that adds the design document, ADRs, agent docs, CI, GPL license, and the initial Python 3.12 repo scaffolding.
