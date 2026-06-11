"""common — §4.1 cross-cutting packaging, reproducibility & observability.

Shared infrastructure used by every stage of the pipeline:
    manifest    — the per-utterance manifest schema (trainer-compatible)
    config      — config loading, seed fixing, determinism
    checkpoint  — idempotent resume and partial-failure handling
    logging     — structured logging / observability setup
"""
