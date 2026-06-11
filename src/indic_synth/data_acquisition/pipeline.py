"""data_acquisition.pipeline — Stage 4.2 entrypoint.

`run(cfg, logger)` is the stage's uniform entry called by scripts/run.py. It
builds the stage config from the unified config subsection, then runs the two
stages across the cost boundary:

    Stage A : build_catalog  -> catalog.parquet
              build_selection -> selection_manifest.jsonl   (seeded, deterministic)
    --- cost boundary ---
    Stage B : pull_audio     -> ref_audio/*.<ext> + reference_manifest.jsonl

reference_manifest.jsonl is the input to Stage 4.3 (audio_engineering). Returns a
run_summary dict (selection counts, gender balance, row groups / bytes touched).
"""
from __future__ import annotations

import json
import os
import time

from indic_synth.common.logging import get_logger

from .catalog import build_catalog
from .config import AcquireConfig
from .puller import pull_audio
from .sampler import build_selection


def run(cfg, logger=None) -> dict:
    """Run §4.2 end-to-end. `cfg` is the unified Config (has `.stage(...)`)."""
    logger = logger or get_logger("acquire")
    acfg = AcquireConfig.from_dict(cfg.stage("data_acquisition"))
    logger.info("Acquisition config: %s", acfg.public_dict())

    t0 = time.time()
    os.makedirs(acfg.out_dir, exist_ok=True)

    catalog = build_catalog(acfg, logger)
    records, sel_summary = build_selection(catalog, acfg, logger)
    pull_stats = pull_audio(records, acfg, logger)

    summary = {
        "stage": "data_acquisition",
        "elapsed_sec": round(time.time() - t0, 2),
        "selection": sel_summary,
        "pull": pull_stats,
        "reference_manifest": os.path.join(acfg.out_dir, "reference_manifest.jsonl"),
    }
    with open(os.path.join(acfg.out_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Acquisition done in %.1fs: %d clips, %d speakers",
                summary["elapsed_sec"], sel_summary["total_clips"], sel_summary["total_speakers"])
    return summary
