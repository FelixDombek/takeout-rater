# Adapter scorers

Adapters wrap external tools or heavyweight ML pipelines (e.g. LAION
aesthetic predictor, CLIP-based regressors) and expose them through the
`BaseScorer` interface.

Each adapter lives in its own sub-package (e.g. `takeout_rater/scorers/adapters/laion/`).
Heavy dependencies are declared as optional extras in `pyproject.toml`.

## Adding an adapter

See `docs/agents/how-to-add-a-scorer.md` for the full workflow.
