# Agent Instructions

## Testing

- Run pytest with repository-local temp directories only.
- For concurrent pytest runs, use unique basetemp paths under `.pytest-tmp/`, for example:
  `--basetemp .pytest-tmp/jobs` or `--basetemp .pytest-tmp/indexer`.
- Do not delete `.pytest-tmp/` or `.pytest-tmp*` directories after test runs.
- Do not issue recursive cleanup commands such as `Remove-Item -Recurse`, `rm -rf`, or equivalent
  for pytest temp directories unless the user explicitly asks for cleanup.
