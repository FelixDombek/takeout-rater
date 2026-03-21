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

## Roadmap

| Iteration | Scope |
|---|---|
| **0** *(this branch)* | Repo foundation: design docs, ADRs, agent docs, scorer interface, CI |
| **1** | Indexing, DB, thumbnail cache, minimal browse UI |
| **2** | Scorer pipeline end-to-end + 1–2 real scorers (aesthetic) |
| **3** | Clustering, cluster view, best-of-cluster export |

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
```

Sub-commands (`index`, `score`, `browse`, `export`) are implemented in later iterations.

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and the agent enablement docs in
[`docs/agents/`](docs/agents/).

---

## License

GPL-3.0-only — see [`LICENSE`](LICENSE).

