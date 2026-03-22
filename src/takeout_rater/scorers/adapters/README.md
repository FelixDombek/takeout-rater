# Adapter scorers

Adapters wrap external tools or heavyweight ML pipelines (e.g. LAION
aesthetic predictor, CLIP-based regressors, ViT classifiers) and expose them
through the `BaseScorer` interface.

Each adapter is a single `.py` module inside this directory (e.g.
`takeout_rater/scorers/adapters/laion.py`).  Use a sub-package (directory with
`__init__.py`) only if the adapter needs to split across multiple source files.
Heavy dependencies are declared as optional extras in `pyproject.toml`.

## Adding an adapter

See `docs/agents/how-to-add-a-scorer.md` for the full workflow.
