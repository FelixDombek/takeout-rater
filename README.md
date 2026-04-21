# takeout-rater

Aesthetics scoring orchestrator for image libraries, e.g. Google Photos Takeout folders.

Point it at the directory containing your images or albums of images, e.g. `Takeout/Google Fotos`,
and give it a location to build its database, a directory called `takeout-rater/` with a SQLite library, thumbnail cache,
and exports, and it will give you access to tools such as:

 - Library and album browsing
 - Several aesthetics scorers
 - CLIP-based semantic search
 - Face detection and grouping

and much more.

---

## Quickstart (one command)

```bash
git clone https://github.com/FelixDombek/takeout-rater.git
cd takeout-rater

./run
```

The launcher will:
1. Create a local Python virtual environment (`.venv/`) and install dependencies on first run.
2. Start the local web server.
3. Open your browser automatically.

**First time:** the browser will show a setup page.
Click **Browse…** (or type the path) to select the directory that contains your photos or album folders, then click **Save & continue**.

After indexing (see below), the browser shows your full photo library.

### Changing the Takeout folder path

Open the browser and navigate to **http://127.0.0.1:8765/setup**,
or click the ⚙ setup link in the header when present.
Enter the new path and click **Save & continue**.

---

## What it does

- **Indexes** your photo library in place (read-only)
- **Scores** photos using pluggable scorers (aesthetic quality, media type, …)
- **Browses** your library via a local web UI with filters and sorting
- **Exports** your best photos (top-N, or best-of-cluster) to a folder

See [`docs/design.md`](docs/design.md) for the full architecture overview.

---

## Screenshots

| 📷 Browse | 🔗 Clusters |
|:---------:|:-----------:|
| [![Browse](docs/screenshots/browse.png)](docs/screenshots/browse.png) | [![Clusters](docs/screenshots/clusters.png)](docs/screenshots/clusters.png) |
| **🎯 Scoring** | **⚙ Jobs** |
| [![Scoring](docs/screenshots/scoring.png)](docs/screenshots/scoring.png) | [![Jobs](docs/screenshots/jobs.png)](docs/screenshots/jobs.png) |

---

## Developer quick start (manual install)

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Launch the web UI (first-run shows setup page to select Takeout path)
takeout-rater serve
# → open http://127.0.0.1:8765 in your browser

# Or, point directly at a library root and browse
takeout-rater index /path/to/folder-containing-Takeout
takeout-rater browse /path/to/folder-containing-Takeout
# → open http://127.0.0.1:8765/assets in your browser
```

---

## Development setup

**Requirements:** Python 3.12+, pip

```bash
# Clone
git clone https://github.com/FelixDombek/takeout-rater.git
cd takeout-rater

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
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
pytest tests/slow
```

---

## CLI (section outdated)

```bash
# Show help
python -m takeout_rater --help
# or, after installing:
takeout-rater --help

# Launch the web UI (shows setup page on first run)
takeout-rater serve [--port 8765]

# Index a Takeout folder (generates DB + thumbnail cache)
takeout-rater index /path/to/library-root

# Browse the indexed library in a local web UI
takeout-rater browse /path/to/library-root [--port 8765]

# Run scorers over indexed assets
takeout-rater score /path/to/library-root [--scorer blur] [--phash]

# Group near-duplicates by perceptual hash
takeout-rater cluster /path/to/library-root [--threshold 10]

# Export best-of-cluster photos to a folder
takeout-rater export /path/to/library-root [--scorer aesthetic] [--out ./export]

# Compute SHA-256 hashes for deduplication
takeout-rater rehash /path/to/library-root
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
