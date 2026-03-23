# Design: takeout-rater

*Last updated: 2026-03 (Iteration 7 complete: all core workflows available from the UI)*

---

## Problem statement

Google Photos Takeout produces a large, immutable archive of photos (potentially 200k+ files, hundreds of GB).  The archive contains one or more `Takeout/` directories with album sub-folders, per-year folders, and a `*.supplemental-metadata.json` sidecar for almost every image.

There is no good local tool to:
- **rank** photos by visual quality / aesthetics,
- **group** near-duplicate shots, and
- **export** the best-of set for sharing.

`takeout-rater` fills this gap: it reads the takeout archive in place (never modifies it), builds an external SQLite library, scores photos using pluggable scorers, and lets the user browse, filter, and export through a local web UI.

---

## Non-goals (v1)

| Out-of-scope | Rationale |
|---|---|
| Video files | Adds complexity (thumbnails, playback, different scorers). Add later. |
| EXIF modification | Takeout is immutable; all metadata lives in the DB. |
| Cloud sync / remote UI | Local-only is simpler and privacy-preserving. |
| Full query language / DSL | Structured filters with AND semantics are sufficient for v1. |
| Multi-library simultaneous open | One library at a time; switch by relaunching. |

---

## Architecture overview

```
Takeout/                          ← immutable Google Photos archive
  Google Photos/                  ← localized app name (new format; may also be
    Albums/                          "Google Fotos", "Google Foto", etc.)
    Photos from YYYY/
  Photos from YYYY/               ← old format: album dirs directly in Takeout/
  *.supplemental-metadata.json

takeout-rater/                    ← sibling directory, all mutable state
  library.sqlite                  ← single-file metadata + score store
  thumbs/                         ← 512 px JPEG thumbnail cache (indexed by asset id)
  cache/                          ← scorer intermediates (embeddings, etc.)
  logs/                           ← structured application logs
  exports/                        ← copies of selected assets

src/takeout_rater/
  indexing/     ← takeout scanner, sidecar parser, thumbnail generator
  db/           ← SQLite schema, migrations, query helpers
  scorers/      ← BaseScorer interface, explicit registry, heuristics/, adapters/
  api/          ← FastAPI routes (Iteration 1+)
  ui/           ← Jinja2 templates / static assets (Iteration 1+)
  cli.py        ← `takeout-rater` entry-point
```

The **indexer** walks the Takeout tree in two passes:

1. **Directory enumeration** (`os.walk` – fast, no per-file `stat`): all directory names are collected and image filenames are identified by extension.  Progress is reported as *dirs_scanned / total_dirs*.
2. **Metadata collection** (one `stat` + sidecar probe per file): file sizes are read and sidecar JSON files are located.  Progress is reported as *files_indexed / total_files* plus the name of the directory currently being processed.

**Scorers** are run as background jobs; they read thumbnails and write results to `scorer_runs` + `asset_scores`.
The **API** layer exposes filtered/sorted asset lists and serves thumbnails.
The **UI** renders pages with HTMX-driven interactions (no full SPA).

---

## Library layout

```
<library-root>/               ← directory the user points the app at
  Takeout/                    ← Google Photos export (untouched)
  takeout-rater/              ← created by the app on first run
    library.sqlite
    thumbs/
    cache/
    logs/
    exports/
```

The library root is the directory that *contains* `Takeout/`.  All app state lives in the sibling `takeout-rater/` directory so the takeout itself is never modified.

See [ADR-0004](decisions/ADR-0004-library-state-location.md).

---

## Data model overview

### Core tables (conceptual)

**assets** — one row per image file in the takeout  
`id, relpath (unique), filename, ext, size_bytes, sha256 (nullable), taken_at, created_at_sidecar, width, height, title, description, image_views, google_photos_url, favorited, archived, trashed, geo_lat, geo_lon, geo_alt, geo_exif_lat, geo_exif_lon, geo_exif_alt, origin_type, origin_device_type, origin_device_folder, app_source_package, sidecar_relpath, mime, indexed_at`

**albums** + **album_assets** — many-to-many album membership

**scorer_runs** — one row per scorer execution  
`id, scorer_id, variant_id, scorer_version, params_json, params_hash, started_at, finished_at`

**asset_scores** — one row per (asset, scorer_run, metric)  
`asset_id, scorer_run_id, metric_key, value (float)`  
Unique on `(asset_id, scorer_run_id, metric_key)`.

**phash** — perceptual hash per asset  
`asset_id, phash_hex, algo ("phash"), computed_at`

**clusters** + **cluster_members** — near-duplicate groups  
`cluster_id, method, params_json; cluster_id, asset_id, distance, is_representative`

### Sidecar fields stored

`*.supplemental-metadata.json` is the **primary metadata source**.
Fields are mapped to `assets` columns as follows:

| Sidecar path | Column | Notes |
|---|---|---|
| `photoTakenTime.timestamp` | `taken_at` | Preferred timestamp; stored as Unix int |
| `creationTime.timestamp` | `created_at_sidecar` | Upload/creation time; fallback if `taken_at` missing |
| `title` | `title` | Usually equals the filename |
| `description` | `description` | Often empty string |
| `imageViews` | `image_views` | Stored as int (sidecar encodes as string) |
| `url` | `google_photos_url` | Deep link back to Google Photos |
| `geoData.latitude` | `geo_lat` | Always present; 0.0 when no location |
| `geoData.longitude` | `geo_lon` | Always present; 0.0 when no location |
| `geoData.altitude` | `geo_alt` | Always present; 0.0 when unknown |
| `geoDataExif.latitude` | `geo_exif_lat` | Present in ~51 % of sidecars |
| `geoDataExif.longitude` | `geo_exif_lon` | Present in ~51 % of sidecars |
| `geoDataExif.altitude` | `geo_exif_alt` | Present in ~51 % of sidecars |
| `favorited` | `favorited` | Optional bool; stored as 0/1 INTEGER |
| `archived` | `archived` | Optional bool; stored as 0/1 INTEGER |
| `trashed` | `trashed` | Optional bool; stored as 0/1 INTEGER |
| `googlePhotosOrigin` (key) | `origin_type` | First key present: `mobileUpload`, `driveSync`, `fromPartnerSharing`, `fromSharedAlbum`, `webUpload`, `composition` |
| `googlePhotosOrigin.mobileUpload.deviceType` | `origin_device_type` | e.g. `ANDROID_PHONE` |
| `googlePhotosOrigin.mobileUpload.deviceFolder.localFolderName` | `origin_device_folder` | Often empty |
| `appSource.androidPackageName` | `app_source_package` | Present in ~13 % of sidecars |

Fields not mapped to columns (`removeResultReason`, `sharedAlbumComments`) are not
currently stored; they can be added in a later migration if needed.

---

## Scorer system

### Concepts

- **ScorerSpec** — static description: `scorer_id`, display name, list of `MetricSpec`, list of `VariantSpec`.
- **MetricSpec** — one output dimension: `key`, `min_value`, `max_value`, `higher_is_better`.
- **VariantSpec** — one model/algorithm variant: `variant_id`.  Variants within a scorer produce **non-comparable** scores and are stored separately.
- **BaseScorer** — abstract class with `spec()`, `is_available()`, `create()`, and `score_batch()`.

Scorers are **multi-metric**: a single scorer run may produce several metric keys (e.g. `aesthetic`, `technical_quality`, `composition`).

### Scorer discovery (v1)

Discovery is **explicit**: scorers are imported and listed in `takeout_rater/scorers/registry.py`.  No dynamic plugin scanning.  This is intentional — it keeps the active scorer set visible in code review and prevents accidental activation of unfinished scorers.

See [ADR-0003](decisions/ADR-0003-scorer-plugin-architecture.md) and `docs/agents/how-to-add-a-scorer.md`.

---

## Filtering and sorting (v1)

No query language in v1.  Filters are structured objects combined with **AND** semantics.

Supported filter types:
1. **Numeric range** (`min`/`max`) — for scorer metrics (aesthetic ≥ 8, nsfw ≤ 2, …)
2. **Boolean toggle** — `is_cluster_representative`, `has_google_photos_url`, `is_heic`
3. **Set membership** — `album in {…}`, `year in {…}`

A **single active sort** is applied (e.g. `aesthetic DESC`), with an optional secondary sort for tie-breaking.

Assets missing the primary sort metric are excluded by default ("only scored" view).

View presets (filter + sort combinations) are stored as JSON in the DB for re-use.

See [ADR-0005](decisions/ADR-0005-v1-filtering-sorting-no-dsl.md).

---

## Clustering (near-duplicate grouping)

1. Compute pHash for each image (on thumbnail, fast).
2. Bucket by prefix bits to reduce all-pairs comparisons.
3. Within each bucket, compare by Hamming distance; group into clusters.
4. Persist `clusters` + `cluster_members` including `is_representative`.
5. UI provides a "cluster view": best-scored representative per cluster.

Threshold and algorithm are stored as `params_json` on `clusters` rows so results are reproducible and auditable.

---

## HEIC support

HEIC is handled via a pluggable `ImageLoader` abstraction:
- Default: Pillow (JPEG, PNG, WebP)
- HEIC: `pillow-heif` (included in the base install)

If HEIC decoding fails for a specific file, the asset is indexed but marked `unsupported/skipped` and excluded from scoring.

---

## Iteration plan

| Iteration | Scope |
|---|---|
| **0** ✅ | Repo foundation: design docs, ADRs, agent docs, pyproject, scorer interface + registry, CI, skeleton modules |
| **1** ✅ | Indexing: takeout scanner, sidecar parser, asset DB, thumbnail cache, minimal browse UI (FastAPI + HTMX) |
| **2** ✅ | Scorer framework end-to-end: scoring pipeline, BlurScorer + DummyScorer wired to DB, pHash computation, `score` CLI command, score display in UI |
| **3** ✅ | Clustering: pHash-based cluster builder, cluster persistence, cluster view in UI, best-of-cluster `export` CLI command |
| **4** ✅ | Aesthetic scorer: LAION Aesthetic Predictor v2 (CLIP ViT-L/14 + MLP) as scorer, `aesthetic` metric (0–10), sort-by-aesthetic in UI |
| **5** ✅ | NSFW / quality filter: NSFW detector wired as a scorer, filter-by-score range in UI, view presets saved to DB |
| **6** ✅ | SHA-256 deduplication: `rehash` CLI command, deduplicate browse UI, duplicate paths in detail view; DB at schema version 3 |
| **7** ✅ | All core workflows available from the UI: background jobs API (`/api/jobs/*`) for score, cluster, export, and rehash; `/jobs` page with progress bars and last-run status; Jobs nav link added to every page |

---

## UI-first development guideline

> **All future features must be accessible from the web UI from day one.**
>
> Starting with Iteration 7, every new workflow (scorer, export mode, admin tool, etc.) **must** include:
> 1. A UI trigger (button, form, or page) on the `/jobs` page or a dedicated feature page.
> 2. A status / progress mechanism so users can see the job running and its result.
> 3. API endpoint(s) that the UI calls (under `/api/…`).
>
> CLI commands may still be added as a convenience for power users and scripting, but they are
> **never** the primary or only way to access a feature.  New features must not require
> CLI usage to be functional.  This rule applies to all scorers, data-management tools,
> export modes, and any other capability added in future iterations.
