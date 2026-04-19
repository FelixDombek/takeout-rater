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
│   ├── agents/          ← agent enablement docs (this directory)
│   ├── history/         ← original chat transcripts
│   └── tools/           ← helper scripts documentation
├── scripts/             ← launcher scripts
├── .github/workflows/   ← CI: lint (ruff format + ruff check), test (pytest), slow-tests
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
| `__main__.py` | Entry point for `python -m takeout_rater` |
| `cli.py` | `takeout-rater` CLI entry-point: `index`, `score`, `browse`, `cluster`, `export`, `serve`, `rehash` |
| `config.py` | Library path configuration (read/write Takeout root path) |
| **Scoring pipeline** | |
| `scoring/pipeline.py` | `run_scorer()` — runs a scorer, writes to `asset_scores` |
| **Scorers** | |
| `scoring/scorers/base.py` | `MetricSpec`, `VariantSpec`, `ScorerSpec`, `BaseScorer` |
| `scoring/scorers/registry.py` | Explicit scorer class list + `list_scorers()` |
| `scoring/scorers/simple.py` | Simple Pillow scorer: sharpness, brightness/contrast, noise variants |
| `scoring/scorers/brisque.py` | BRISQUE no-reference IQA scorer (piq wrapper) |
| `scoring/scorers/laion.py` | LAION Aesthetic Predictor v2 (CLIP ViT-L/14 + MLP, 0–10 scale) |
| `scoring/scorers/nsfw.py` | NSFW detector (Falconsai ViT classifier, 0–1 probability) |
| `scoring/scorers/clip_iqa.py` | CLIP-IQA zero-shot quality scorer (CLIP ViT-L/14, 0–1) |
| `scoring/scorers/nima.py` | NIMA aesthetic/technical scorer (MobileNet-V2, 1–10) |
| `scoring/scorers/pyiqa_adapter.py` | PyIQA scorer: MUSIQ, TOPIQ, NIQE (0–1 normalised) |
| `scoring/scorers/cafe_style.py` | CafeAI style classifier: photo/anime/illustration/3D/CGI probabilities (0–1 each) |
| **Indexing** | |
| `indexing/scanner.py` | `scan_takeout()` — walk Takeout tree, enumerate `AssetFile` objects |
| `indexing/sidecar.py` | `parse_sidecar()` — parse `*.supplemental-metadata.json` → `SidecarData` |
| `indexing/thumbnailer.py` | `generate_thumbnail()` — 512 px JPEG thumbnail cache |
| `indexing/run.py` | `run_index()` — main indexing pipeline with progress tracking |
| **Database** | |
| `db/schema.py` | Migration runner (`migrate()`), `CURRENT_SCHEMA_VERSION = 6` |
| `db/connection.py` | `open_library_db()` — open / create the library database |
| `db/queries.py` | Asset CRUD, scoring queries, clustering helpers, preset helpers, `CURRENT_INDEXER_VERSION` |
| `db/migrations/` | Single consolidated schema: `0001_initial_schema.sql` (version 6) |
| **Clustering** | |
| `clustering/builder.py` | `build_clusters()` — pHash-based near-duplicate grouping |
| `clustering/phash.py` | `compute_dhash()`, `compute_phash_all()` — pHash via dhash algorithm |
| **API routers** | |
| `api/assets.py` | Routes: `GET /assets`, `GET /assets/{id}`, `GET /thumbs/{id}`, `GET /api/timeline`, `GET /api/timeline/seek` |
| `api/clusters.py` | Routes: `GET /clusters`, `GET /clusters/{id}` |
| `api/presets.py` | Routes: `GET/POST /api/presets`, `DELETE /api/presets/{id}` |
| `api/config_routes.py` | Routes: `GET /health`, `GET /api/config`, `GET /api/library/status`, `POST /api/config/takeout-path`, `POST /api/config/open-picker` |
| `api/jobs.py` | Routes: `GET /api/jobs/status`, `GET /api/jobs/scorers`, `POST /api/jobs/{type}/start` (index, score, cluster, export, rehash, rescan) |
| **UI** | |
| `ui/app.py` | `create_app()` — FastAPI + Jinja2 setup, router registration |
| `ui/templates/base.html` | Layout wrapper, navigation header (Browse, Clusters, Scoring, Jobs, Setup) |
| `ui/templates/browse.html` | Full browse page (pagination, filters, sort, favorites, timeline) |
| `ui/templates/browse_partial.html` | Card grid fragment (HTMX infinite scroll) |
| `ui/templates/detail.html` | Asset detail page (scores, EXIF, sidecar, aliases) |
| `ui/templates/detail_partial.html` | Detail panel fragment (lightbox overlay) |
| `ui/templates/clusters.html` | Cluster browse page (paginated grid of representatives) |
| `ui/templates/cluster_detail.html` | Cluster detail page (list members with distances) |
| `ui/templates/setup.html` | Configuration page (path picker, directory browser) |
| `ui/templates/jobs.html` | Background jobs page (progress bars, start/cancel buttons) |
| `ui/templates/scoring.html` | Scorer selection page (checkboxes, variants, Simple/Technical toggle) |

---

## Where to add things

| Task | Location |
|---|---|
| New scorer | `src/takeout_rater/scorers/<name>.py` + entry in `registry.py` |
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
