"""audio_engineering.pipeline — Stage 4.3 entrypoint.

`run(cfg, logger)` reads reference_manifest.jsonl (the downloaded clips from §4.2),
runs each through prepare.process_clip, and writes prepared_audio/<ref_id>.wav +
prepared_manifest.jsonl (input to §4.5) + prepare_summary.json. Resumable per
ref_id via common.checkpoint; partial failures are isolated and logged.
"""
from __future__ import annotations

import json
import os
import time

from indic_synth.common.checkpoint import load_done
from indic_synth.common.logging import get_logger
from indic_synth.common.manifest import append_jsonl, read_jsonl

from .config import AudioConfig
from .prepare import process_clip


def run(cfg, logger=None) -> dict:
    """Run §4.3 end-to-end. `cfg` is the unified Config (has `.stage(...)`)."""
    logger = logger or get_logger("prepare")
    acfg = AudioConfig.from_dict(cfg.stage("audio_engineering"))

    out_dir = os.path.abspath(acfg.out_dir)
    base = os.path.dirname(os.path.abspath(acfg.in_manifest))
    prepared_dir = os.path.join(out_dir, "prepared_audio")
    out_manifest = os.path.join(out_dir, "prepared_manifest.jsonl")
    os.makedirs(prepared_dir, exist_ok=True)

    # Only clips the puller actually downloaded.
    records = [r for r in read_jsonl(acfg.in_manifest)
               if r.get("status") == "downloaded" and r.get("local_audio_path")]

    done = load_done(out_manifest, key="ref_id", status="prepared",
                     audio_field="prepared_audio_path", out_dir=out_dir)
    pending = [r for r in records if r["ref_id"] not in done]
    logger.info("%d downloaded clips, %d already prepared, %d to process",
                len(records), len(done), len(pending))

    t0 = time.time()
    stats = {"prepared": 0, "failed": 0, "sr_in": {}, "flags": {}, "backends": {}}
    for r in pending:
        src = os.path.join(base, r["local_audio_path"])
        dst = os.path.join(prepared_dir, r["ref_id"] + ".wav")
        if not os.path.exists(src):
            logger.warning("%s: source missing (%s)", r["ref_id"], src)
            append_jsonl(out_manifest, {**r, "status": "failed", "error": "source audio missing"})
            stats["failed"] += 1
            continue
        try:
            metrics = process_clip(src, dst, acfg, logger)
        except Exception as exc:
            logger.warning("%s failed: %s", r["ref_id"], exc)
            append_jsonl(out_manifest, {**r, "status": "failed", "error": str(exc)})
            stats["failed"] += 1
            continue

        append_jsonl(out_manifest, {**r, **metrics,
                                    "prepared_audio_path": os.path.relpath(dst, out_dir),
                                    "status": "prepared"})
        stats["prepared"] += 1
        sr = str(metrics["source_sr"])
        stats["sr_in"][sr] = stats["sr_in"].get(sr, 0) + 1
        stats["backends"][metrics["decode_backend"]] = stats["backends"].get(metrics["decode_backend"], 0) + 1
        for fl in metrics["qc_flags"]:
            stats["flags"][fl] = stats["flags"].get(fl, 0) + 1

    summary = {"stage": "audio_engineering", "elapsed_sec": round(time.time() - t0, 2),
               "target_sr": acfg.target_sr, "norm": acfg.norm, "trim": acfg.trim, **stats,
               "prepared_manifest": out_manifest}
    with open(os.path.join(out_dir, "prepare_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Prepare complete: %s", json.dumps(summary, ensure_ascii=False))
    return summary
