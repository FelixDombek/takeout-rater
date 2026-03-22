# Performance budgets

Constraints that all code must respect to keep the app responsive with 200k+ photos.

---

## Thumbnails

| Constraint | Value | Rationale |
|---|---|---|
| Max thumbnail dimension | 512 px (longest edge) | Sufficient for grid browsing; fast to serve |
| Thumbnail format | JPEG, quality 85 | Good compression/quality ratio |
| Max thumbnail file size | ~50 KB target | 200k thumbnails ≈ 10 GB at this budget |
| Thumbnail generation | Streaming, one at a time | Avoid holding multiple full-res images in memory |

**Never load a full-resolution image just to serve a thumbnail.**  Thumbnails must be pre-generated and cached on disk.

---

## API / UI response times

| Operation | Budget |
|---|---|
| Page load (100 thumbnails) | < 500 ms |
| Single thumbnail serve | < 20 ms (from disk cache) |
| Filter/sort query (indexed DB) | < 100 ms |
| Scorer status poll | < 50 ms |

---

## Scoring pipeline

| Constraint | Value | Rationale |
|---|---|---|
| Scorer batch size | 16–64 images | Balance GPU utilisation vs. memory |
| Input to scorer | Thumbnail (512 px), not full-res | Speed; most aesthetic models trained on 224–512 px |
| Model load | Once per scorer process | Avoid repeated cold starts |
| Embedding cache | Store in `takeout-rater/cache/` | Avoid recomputing on re-runs |

**Never pass full-resolution images to a scorer unless specifically required.**

---

## Memory

| Constraint | Value |
|---|---|
| Peak RAM for indexing | < 500 MB |
| Peak RAM for scoring (no GPU) | < 2 GB |
| SQLite page cache | Default (2 MB); increase only if profiling shows benefit |

---

## Indexing

| Constraint | Value |
|---|---|
| Sidecar parse | Streaming JSON; do not buffer entire file in memory |
| Directory walk | `os.scandir()` / `pathlib` — avoid `glob('**/*')` on 200k file trees |
| Incremental indexing | Only process new/changed files (compare `mtime` or `size_bytes`) |

---

## Thumbnail dimensions reference

| Use case | Recommended size |
|---|---|
| Grid view | 256 px |
| Detail/lightbox | 1024 px (generate separately if needed) |
| Scorer input | 512 px (re-use grid thumbnail) |
| pHash computation | 128 px or smaller |
