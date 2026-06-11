"""common.logging — structured logging & retry helpers (§4.1).

The single canonical logger factory + exponential-backoff retry wrapper shared by
every stage (ported from kathbath-extraction/logging_utils.py, which the audio and
QC modules had near-duplicates of). Logging to stdout makes progress visible in a
Colab cell; `with_retries` backs the fault-tolerant HF/IO reads.
"""
from __future__ import annotations

import logging
import sys
import time


def get_logger(name: str = "indic_synth", level: int = logging.INFO) -> logging.Logger:
    """Return a stdout logger; idempotent (won't double-add handlers)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def with_retries(fn, *args, max_retries: int = 4, backoff: float = 2.0,
                 logger: logging.Logger | None = None, what: str = "op", **kwargs):
    """Call fn with exponential-backoff retries. Re-raises the last error."""
    last = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # network/IO/parse errors are all transient-ish here
            last = exc
            wait = backoff ** (attempt - 1)
            if logger:
                logger.warning(
                    "%s failed (attempt %d/%d): %s%s",
                    what, attempt, max_retries, exc,
                    f"; retrying in {wait:.1f}s" if attempt < max_retries else "",
                )
            if attempt < max_retries:
                time.sleep(wait)
    raise last
