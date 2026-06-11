"""common.logging — structured logging & observability setup (§4.1).

Configures structured, per-stage logging so run progress is visible and failures
are diagnosable. Backs the `run_summary.json` / `prepare_summary.json` style
observability artifacts each stage emits (counts, gender balance, elapsed,
bytes/row-groups touched).
"""
