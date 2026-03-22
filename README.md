# takeout-rater

Aesthetics scoring orchestrator for Google Photos Takeout folders.

Point it at the directory containing your `Takeout/` export and it will build
a sibling `takeout-rater/` directory with a SQLite library, thumbnail cache,
and exports — without ever modifying the original archive.

---

## What it does

- **Indexes** your Google Photos Takeout in place (read-only)
- **Scores** photos using pluggable scorers (aesthetic quality, perceptual hash, …)
- **Browses** your library via a local web UI with filters and sorting
- **Exports** your best photos (top-N, or best-of-cluster) to a folder

See [`docs/design.md`](docs/design.md) for the full architecture overview.

---

## Quick start

```bash
# Install
pip install -e ".[index,web]"

# Index your Takeout folder (generates DB + thumbnail cache)
takeout-rater index /path/to/folder-containing-Takeout

# Launch the local web UI
takeout-rater browse /path/to/folder-containing-Takeout
# → open http://127.0.0.1:8765/assets in your browser
```

---

## Roadmap

| Iteration | Scope | Status |
|---|---|---|
| **0** | Repo foundation: design docs, ADRs, agent docs, scorer interface, CI | ✅ Done |
| **1** | Indexing, DB, thumbnail cache, minimal browse UI | ✅ Done |
| **2** | Scorer pipeline end-to-end + BlurScorer + pHash | ✅ Done |
| **3** | Clustering, cluster view, best-of-cluster export | ✅ Done |
| **4** | Aesthetic scorer (ONNX/Torch), sort by aesthetic in UI | Planned |

---

## Development setup

**Requirements:** Python 3.12+, pip

```bash
# Clone
git clone https://github.com/FelixDombek/takeout-rater.git
cd takeout-rater

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# (Optional) install optional scorer dependencies
pip install -e ".[aesthetic]"   # PyTorch-based aesthetic scorer
pip install -e ".[heic]"        # HEIC image support via pillow-heif
```

---

## Running lint and tests

```bash
# Format check
ruff format --check src/ tests/

# Lint
ruff check src/ tests/

# Tests
pytest
```

---

## CLI

```bash
# Show help
python -m takeout_rater --help
# or, after installing:
takeout-rater --help

# Index a Takeout folder
takeout-rater index /path/to/library-root

# Browse the indexed library in a local web UI
takeout-rater browse /path/to/library-root [--port 8765]
```

---

## Tools

See [`docs/tools/README.md`](docs/tools/README.md).

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and the agent enablement docs in
[`docs/agents/`](docs/agents/).

---

## License

GPL-3.0-only — see [`LICENSE`](LICENSE).

