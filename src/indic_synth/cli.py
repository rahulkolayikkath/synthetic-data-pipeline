"""indic_synth.cli — installable command-line entry point (`indic-synth`).

After `pip install -e .` the pipeline is runnable as a real CLI command:

    indic-synth --config config.yaml                       # all stages
    indic-synth --config config.yaml --stages tts_generation quality_control
    indic-synth --config config.yaml --validate-only       # check config, run nothing

This is the same orchestration as `scripts/run.py` (which now delegates here), so the
documented `python scripts/run.py ...` flow keeps working unchanged. The config is
validated against the schema in `indic_synth.common.config` before any stage runs;
invalid configs fail fast with a clear, itemized message (exit code 2).
"""
from __future__ import annotations

import argparse
import json
import os
import time

from indic_synth.common.config import (ConfigError, load_config,
                                        set_determinism, validate_config)
from indic_synth.common.logging import get_logger

STAGE_NAMES = ["data_acquisition", "audio_engineering", "sentence_generation",
               "tts_generation", "quality_control"]


def _build_stages():
    """Lazily import stage pipelines so importing the CLI (and --validate-only) stays
    cheap and doesn't pull torch / soundfile until a stage actually runs."""
    from indic_synth.audio_engineering import pipeline as audio_engineering
    from indic_synth.data_acquisition import pipeline as data_acquisition
    from indic_synth.quality_control import pipeline as quality_control
    from indic_synth.sentence_generation import pipeline as sentence_generation
    from indic_synth.tts_generation import pipeline as tts_generation
    runners = {
        "data_acquisition": data_acquisition.run,
        "audio_engineering": audio_engineering.run,
        "sentence_generation": sentence_generation.run,
        "tts_generation": tts_generation.run,
        "quality_control": quality_control.run,
    }
    return [(name, runners[name]) for name in STAGE_NAMES]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="indic-synth",
        description="Indic synthetic-speech pipeline orchestrator")
    p.add_argument("--config", default="config.yaml", help="path to config.yaml")
    p.add_argument("--stages", nargs="+", default=None, choices=STAGE_NAMES,
                   metavar="STAGE",
                   help="subset of stages to run (default: all, in order). "
                        f"choices: {', '.join(STAGE_NAMES)}")
    p.add_argument("--validate-only", action="store_true",
                   help="validate the config against the schema and exit")
    p.add_argument("--no-validate", action="store_true",
                   help="skip config schema validation (not recommended)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logger = get_logger("run")
    cfg = load_config(args.config)

    if not args.no_validate:
        try:
            validate_config(cfg, logger)
        except ConfigError as e:
            logger.error("%s", e)
            raise SystemExit(2)
    if args.validate_only:
        logger.info("Config valid: %s", args.config)
        return

    set_determinism(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    logger.info("Pipeline start | out_dir=%s seed=%d", cfg.out_dir, cfg.seed)

    t0 = time.time()
    summaries = {}
    for name, fn in _build_stages():
        if args.stages and name not in args.stages:
            continue  # skip stages not requested
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
    logger.info("Requested stages in pipeline done in %.1fs.",
                pipeline_summary["elapsed_sec"])


if __name__ == "__main__":
    main()
