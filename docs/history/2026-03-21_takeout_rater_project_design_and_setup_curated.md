# takeout-rater — Project Design & Setup (2026-03-21, curated)

> Curated transcript. See the verbatim version for the full exchange.

---

## Decisions taken

| # | Topic | Decision |
|---|-------|----------|
| 1 | Architecture | **Python backend + localhost web UI (FastAPI + browser)**. *(Native GUI via Qt/PySide6 and early desktop wrapper via Tauri/Electron both discarded: slower UI iteration, heavier Windows packaging.)* |
| 2 | Metadata store | **SQLite**. *(JSON-per-folder discarded: painful cross-folder queries and concurrent writes.)* |
| 3 | Library location | Sibling directory `<takeout-root>/takeout-rater/` containing `library.sqlite`, `thumbs/`, `cache/`, `logs/`, `exports/`. Takeout folder is never modified. |
| 4 | Asset identity | `relpath` as stable locator; `sha256` computed lazily for exact dedupliction; `pHash` for near-duplicate grouping. |
| 5 | Scope | **Photos only** (no videos). |
| 6 | HEIC support | Opportunistic via `ImageLoader` abstraction (pillow-heif). Missing decoder → asset flagged, not blocked. No auto-conversion on export. |
| 7 | Scorer quality | **Best available Torch-based models** wrapped as adapters. Install size is not a constraint. Model download on first run is acceptable. |
| 8 | Scorer output | **Multi-metric**. Each scorer run produces a dict of named metrics (e.g. overall, composition, lighting…). Stored as one DB row per metric. *(Single-float-per-scorer and JSON-blob options discarded.)* |
| 9 | Metric key | **Two-part key `(scorer_id, metric_key)`** in DB. API/UI may represent as `scorer_id/metric_key`. *(Fully-qualified single string `scorer_id.metric_key` discarded: requires string-splitting conventions, harder to index.)* |
| 10 | Registry discovery | **Explicit import list** in `registry.py` — adding a scorer requires a code change. *(Dev-only folder scan and entry-point plugin discovery deferred to v2.)* |
| 11 | Filtering v1 | **Structured AND-combined filters + single sort** — no free-form query language. Upgradeable to DSL in v2. |
| 12 | Score scale | **Native model scale** (not normalized to 0–10). |
| 13 | Filter scope | **Only scored assets** are shown. Missing score = excluded from results. |
| 14 | Multiple libraries | **One library at a time**. Switching = re-opening another folder. |
| 15 | Python version | **3.12** |
| 16 | License | **GPL** |
| 17 | Target repo | **FelixDombek/takeout-rater**, default branch `main` |

---

## Context & goals

The user wants a local tool to rate 236k Google Photos (715 GB takeout) to find images worthy of posting online. Key priorities in order:
1. **Aesthetic scoring** — primary ranking signal.
2. **Near-duplicate grouping** — pick the best image from a burst/similar set.
3. **Export** — copy selected images to a folder.

Agents should be able to implement and extend the tool with minimal context beyond the docs in the repo.

---

## Data model (SQLite)

```
assets              id, relpath, filename, ext, size_bytes, sha256 (nullable),
                    taken_at, width, height, sidecar_json_relpath,
                    google_photos_url, geo_lat, geo_lon, geo_alt,
                    origin_device_type, mime

albums              id, name, relpath
album_assets        album_id, asset_id

scorer_runs         id, scorer_id, scorer_version, variant_id,
                    params_json, params_hash, started_at, finished_at

asset_metric_scores asset_id, run_id, metric_key, metric_value (float),
                    metric_unit (optional)
                    UNIQUE(asset_id, run_id, metric_key)

phash               asset_id, phash_64, algo, computed_at

clusters            id, method, params_json, created_at
cluster_members     cluster_id, asset_id, distance, is_representative
```

**Sidecar fields indexed as first-class columns:** `taken_at` (from `photoTakenTime.timestamp`), `google_photos_url`, `geo_lat/lon/alt`.

---

## Takeout parsing

- Primary timestamp: `photoTakenTime.timestamp` from `*.supplemental-metadata.json` sidecars; fall back to `creationTime.timestamp`. If both are absent, `taken_at` is stored as NULL and the asset is placed in an "Unknown date" bucket.
- Each physical file is one `asset` row (relpath is unique).
- "Open in Google Photos" action uses stored `google_photos_url`.
- Cross-folder duplicates resolved later via sha256; near-duplicates via pHash cluster view (representative per cluster).

---

## Filtering & sorting (v1)

Filter model stored as a JSON preset:
```json
{
  "sort": {"field": "aesthetic.overall", "direction": "desc"},
  "filters": [
    {"field": "aesthetic.overall", "op": ">=", "value": 7.0},
    {"field": "nsfw.score",        "op": "<",  "value": 5.0}
  ],
  "toggles": { "only_cluster_representatives": true }
}
```

Supported filter types: **numeric range**, **boolean/enum**, **set membership** (album ∈ {…}, year ∈ {…}).

Limitation: no OR / parentheses. Multi-select album filter mitigates the most common OR case. DSL deferred to v2.

---

## Scorer architecture

### File layout
```
takeout_rater/scorers/
  base.py           # ScorerSpec, VariantSpec, MetricSpec, BaseScorer, ScoreResult
  registry.py       # explicit import list; list_scorers(), get_spec(id)
  adapters/
    <name>/
      spec.py       # exports SCORER_SPEC
      adapter.py    # calls upstream library; implements score_batch()
      models.py     # supported variant/model IDs
  heuristics/
    phash/
    blur/
    exposure/
```

### ScorerSpec fields
- `id`, `version`, `name`, `description`
- `metrics: list[MetricSpec]` — names + descriptions of all output metrics
- `variants: list[VariantSpec]` — selectable model backends (e.g. OpenCLIP, ConvNeXt)
- `default_variant`
- `is_available() -> bool` + `get_install_hint() -> str`
- `create(variant, params) -> BaseScorer`

### BaseScorer interface
- `prepare(library_ctx)` — load model once
- `score_batch(assets, images) -> list[ScoreResult]` — vectorized for 236k scale
- ScoreResult: `dict[metric_key, float]` + optional `extra_json`

### Scorer dependency model
- Third-party scorer projects are **wrapped as adapters**, not reimplemented. Call their Python API directly (not subprocess). *(Subprocess wrapping discarded: poor performance and error handling at 236k scale.)* Planned adapter targets include `rsinema/aesthetic-scorer`, `LAION-AI/aesthetic-predictor`, and `kenjiqq/aesthetics-scorer`; specific versions and compatibility will be pinned per adapter in its `models.py`.
- Optional extras in `pyproject.toml`: `takeout-rater[aesthetic_rsinema]`, `takeout-rater[laion_predictor]`, etc.

### "Add a scorer" workflow (4 steps)
1. Create `takeout_rater/scorers/adapters/<name>/spec.py` + `adapter.py`
2. Add import to `registry.py`
3. Add tests: registry lists scorer id; metrics declared; `score_batch` returns metrics on one sample image
4. Done — scorer appears in API, UI run-list, and filter builder automatically.

---

## Iteration plan

### Iteration 0 — Repo + agent-operable foundation
- `pyproject.toml` with ruff + pytest
- docs: `docs/architecture.md`, `docs/agents/how-to-add-a-scorer.md`, `docs/decisions/ADR-0001…`
- minimal CLI `takeout-rater --help`

*Goal: agents can run tests/lint and follow conventions.*

### Iteration 1 — Library indexing + DB + thumbnails (no ML)
- Select root containing `Takeout/`; create sibling `takeout-rater/` dir
- Index images, parse sidecar JSON, store `taken_at` / `google_photos_url` / relpath
- Thumbnail cache; minimal UI: list years/albums, grid of thumbs

*Goal: stable asset IDs and a browsable library.*

### Iteration 2 — Scorer framework + 1–2 scorers end-to-end *(key iteration)*
- Scorer interfaces + registry; DB tables for scorer runs + metric scores; basic background job runner
- Two initial scorers:
  1. **pHash scorer** — fast, no heavy deps
  2. **Aesthetic scorer** — first ML adapter, multi-metric output

*Goal: prove "add scorer → registry → run → results stored → sort/filter".*

### Iteration 3 — Clustering + representative view + export
- pHash-based cluster builder (prefix-bucket strategy for 236k scale)
- UI: cluster view (representative + members)
- Export: copy "top 1 per cluster by selected metric" to `exports/`

*Goal: uniqueness + export workflow.*

---

## Agent-enabling repo structure

```
docs/
  architecture.md
  design.md                   # full solution design
  agents/
    repo-map.md
    how-to-add-a-scorer.md    # step-by-step + template
    db-guidelines.md
    performance-budgets.md
    testing-strategy.md
  decisions/
    ADR-0001-local-web-ui.md
    ADR-0002-sqlite-metadata.md
    ADR-0003-scorer-plugin-arch.md
    ADR-0004-thumbnail-caching.md
    ADR-0005-asset-identity.md
    ADR-0006-phash-clustering.md
    ADR-0007-heic-support.md
.github/
  workflows/ci.yml            # ruff + pytest
  ISSUE_TEMPLATE/
  PULL_REQUEST_TEMPLATE.md
```

---

## Repo bootstrap
User confirmed Python 3.12 + GPL-3.0. Bootstrap PR to create the above scaffolding was queued against `main` after the user initialized the empty repository.
