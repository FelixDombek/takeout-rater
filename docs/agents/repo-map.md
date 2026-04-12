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
| `scorers/heuristics/blur.py` | Laplacian variance sharpness scorer (Pillow-based) |
| `scorers/adapters/` | ML/external-tool scorer wrappers (optional deps) |
| `scorers/adapters/laion.py` | LAION Aesthetic Predictor v2 (CLIP ViT-L/14 + MLP, 0–10 scale) |
| `scorers/adapters/nsfw.py` | NSFW detector (Falconsai ViT classifier, 0–1 probability) |
| `scorers/adapters/clip_iqa.py` | CLIP-IQA zero-shot quality scorer (CLIP ViT-L/14, 0–1) |
| `scorers/adapters/nima.py` | NIMA aesthetic/technical scorer (MobileNet-V2, 1–10) |
| `scorers/adapters/pyiqa_adapter.py` | PyIQA adapter: MUSIQ, TOPIQ, NIQE (0–1 normalised) |
| `scorers/adapters/cafe_style.py` | CafeAI style classifier: photo/anime/illustration/3D/CGI probabilities (0–1 each) |
| `scoring/` | Scoring pipeline and pHash computation |
| `scoring/pipeline.py` | `run_scorer()` — runs a scorer, writes to `scorer_runs` + `asset_scores` |
| `scoring/phash.py` | `compute_dhash()`, `compute_phash_all()` — pHash via dhash algorithm |
| `indexing/` | Takeout scanner, sidecar parser, and thumbnail generator |
| `indexing/scanner.py` | `scan_takeout()` — walk Takeout tree, enumerate `AssetFile` objects |
| `indexing/sidecar.py` | `parse_sidecar()` — parse `*.supplemental-metadata.json` → `SidecarData` |
| `indexing/thumbnailer.py` | `generate_thumbnail()` — 512 px JPEG thumbnail cache |
| `db/` | SQLite schema, migrations, and query helpers |
| `db/schema.py` | Migration runner (`migrate()`) |
| `db/connection.py` | `open_library_db()` — open / create the library database |
| `db/queries.py` | `upsert_asset()`, `list_assets()`, `count_assets()`, scoring helpers, pHash helpers, preset helpers |
| `db/migrations/` | Numbered SQL migration files |
| `api/` | FastAPI routers |
| `api/assets.py` | Routes: `GET /assets`, `GET /assets/{id}`, `GET /thumbs/{id}` |
| `api/clusters.py` | Routes: `GET /clusters`, `GET /clusters/{id}` |
| `api/presets.py` | Routes: `GET/POST /api/presets`, `DELETE /api/presets/{id}` |
| `ui/` | Jinja2 application factory + HTML templates |
| `ui/app.py` | `create_app()` — FastAPI + Jinja2 setup |
| `ui/templates/` | `base.html`, `browse.html`, `detail.html`, `clusters.html`, `cluster_detail.html` |

---

## Where to add things

| Task | Location |
|---|---|
| New heuristic scorer | `src/takeout_rater/scorers/heuristics/<name>.py` + entry in `registry.py` |
| New ML/adapter scorer | `src/takeout_rater/scorers/adapters/<name>.py` + entry in `registry.py` |
| New CLI sub-command | `src/takeout_rater/cli.py` |
| New API route | `src/takeout_rater/api/<router>.py` |
| New Jinja2 template | `src/takeout_rater/ui/templates/<name>.html` |
| New DB migration | `src/takeout_rater/db/migrations/<NNNN>_<slug>.sql` + entry in `schema.py` |
| New pipeline step | `src/takeout_rater/scoring/<step>.py` |
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
