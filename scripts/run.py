"""run — CLI entrypoint for the synthetic speech data pipeline.

Usage (the Colab Cell 3):
    python scripts/run.py --config config.yaml
    python scripts/run.py --config config.yaml --stages tts_generation quality_control

Loads one config.yaml, pins determinism, then runs each stage's run(cfg, logger) in
order, threading manifests between them via the shared out_dir:

    data_acquisition    (§4.2) -> reference_manifest.jsonl
    audio_engineering   (§4.3) -> prepared_manifest.jsonl
    sentence_generation (§4.4) -> sentences.jsonl
    tts_generation      (§4.5) -> tts_manifest.jsonl
    quality_control     (§4.6) -> dataset_manifest.jsonl   (final, packaged + QC'd)

Each stage is independently resumable, so a dropped session is restarted by simply
re-running this command. Writes a top-level pipeline_summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Make `import indic_synth` work before `pip install -e .`
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from indic_synth.audio_engineering import pipeline as audio_engineering  # noqa: E402
from indic_synth.common.config import load_config, set_determinism  # noqa: E402
from indic_synth.common.logging import get_logger  # noqa: E402
from indic_synth.data_acquisition import pipeline as data_acquisition  # noqa: E402
from indic_synth.quality_control import pipeline as quality_control  # noqa: E402
from indic_synth.sentence_generation import pipeline as sentence_generation  # noqa: E402
from indic_synth.tts_generation import pipeline as tts_generation  # noqa: E402

STAGES = [
    ("data_acquisition", data_acquisition.run),
    ("audio_engineering", audio_engineering.run),
    ("sentence_generation", sentence_generation.run),
    ("tts_generation", tts_generation.run),
    ("quality_control", quality_control.run),
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Indic synthetic-speech pipeline orchestrator")
    p.add_argument("--config", default="config.yaml", help="path to config.yaml")
    p.add_argument("--stages", nargs="+", default=None,
                   help="subset of stage names to run (default: all, in order)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    set_determinism(cfg.seed)
    logger = get_logger("run")
    os.makedirs(cfg.out_dir, exist_ok=True)
    logger.info("Pipeline start | out_dir=%s seed=%d", cfg.out_dir, cfg.seed)

    t0 = time.time()
    summaries = {}
    for name, fn in STAGES:
        if args.stages and name not in args.stages:
            continue
        logger.info("=========== stage: %s ===========", name)
        summaries[name] = fn(cfg, logger)

    pipeline_summary = {
        "out_dir": cfg.out_dir,
        "seed": cfg.seed,
        "elapsed_sec": round(time.time() - t0, 2),
        "stages": summaries,
    }
    with open(os.path.join(cfg.out_dir, "pipeline_summary.json"), "w", encoding="utf-8") as f:
        json.dump(pipeline_summary, f, indent=2, ensure_ascii=False)
    logger.info("Pipeline done in %.1fs. Final dataset: %s",
                pipeline_summary["elapsed_sec"],
                os.path.join(cfg.out_dir, "dataset_manifest.jsonl"))


if __name__ == "__main__":
    main()
