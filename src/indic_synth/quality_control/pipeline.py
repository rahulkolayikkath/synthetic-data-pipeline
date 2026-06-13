"""quality_control.pipeline — Stage 4.6 entrypoint (final packaging).

`validate_utterance` runs all three §4.6 gates on one clip (cheap DSP first, then
ASR + speaker models), never raising on a bad clip. `run(cfg, logger)` applies it
across tts_manifest.jsonl (§4.5) and writes the final dataset_manifest.jsonl —
the packaged, validated mini-dataset, each row carrying the §4.1 schema + QC
metrics (qc_passed, cer, speaker_sim, asr_transcript). Resumable per utt_id.

run_asr / run_speaker can be turned off to exercise the orchestration + DSP gates
without the GPU models (used by scripts/smoke_qc.py).
"""
from __future__ import annotations

import json
import os
import time

from indic_synth.common.checkpoint import load_done
from indic_synth.common.logging import get_logger
from indic_synth.common.manifest import append_jsonl, read_jsonl

from . import checks as C
from .config import QCConfig


def validate_utterance(utt_id, synth_path, intended_text, lang_code, ref_path=None,
                       cfg=None, run_asr=True, run_speaker=True) -> C.UtteranceQC:
    """Run all §4.6 gates on one utterance. Order: cheap DSP first, then models."""
    cfg = cfg or QCConfig()
    checks = []
    transcript = None

    try:
        wav, sr = C.load_audio(synth_path, cfg.asr_sr)
    except Exception as e:
        return C.UtteranceQC(utt_id, False, [f"load_error:{e}"])

    duration_s = len(wav) / sr

    # --- 3. hard failures (DSP) ---
    checks.append(C.check_silence(wav, sr, cfg))
    checks.append(C.check_clipping(wav, sr, cfg))
    checks.append(C.check_truncation(wav, sr, cfg))
    checks.append(C.check_duration_per_char(wav, sr, intended_text, cfg))
    try:
        checks.append(C.check_looping_audio(wav, sr, cfg))
    except Exception as e:
        checks.append(C.CheckResult("looping_audio", True, detail=f"skipped:{e}"))

    # --- 1. content fidelity (ASR + CER) ---
    if run_asr:
        try:
            from .models import transcribe
            transcript = transcribe(wav, sr, lang_code, cfg)
            cer = C.compute_cer(intended_text, transcript, lang_code)
            checks.append(C.CheckResult("cer", cer <= cfg.cer_max, metric=cer,
                                        detail=f"CER={cer:.3f} (max {cfg.cer_max})"))
            checks.append(C.detect_text_repetition(transcript, cfg))
        except Exception as e:
            checks.append(C.CheckResult("cer", False, detail=f"asr_error:{e}"))

    # --- 2. speaker fidelity (ECAPA cosine) ---
    if run_speaker and ref_path:
        try:
            from .models import speaker_cosine
            ref_wav, _ = C.load_audio(ref_path, cfg.spk_sr)
            cos = speaker_cosine(wav, ref_wav, cfg.spk_sr, cfg)
            checks.append(C.CheckResult("speaker_cos", cos >= cfg.spk_cos_min, metric=cos,
                                        detail=f"cos={cos:.3f} (min {cfg.spk_cos_min})"))
        except Exception as e:
            checks.append(C.CheckResult("speaker_cos", False, detail=f"spk_error:{e}"))

    reasons = [f"{c.name}({c.detail})" for c in checks if not c.passed]
    return C.UtteranceQC(utt_id, len(reasons) == 0, reasons, checks,
                         round(duration_s, 3), transcript)


def _metric(result, name):
    for c in result.checks:
        if c.name == name:
            return c.metric
    return None


def run(cfg, logger=None, run_asr=True, run_speaker=True) -> dict:
    """Run §4.6 across the TTS manifest -> dataset_manifest.jsonl (final dataset)."""
    logger = logger or get_logger("qc")
    qcfg = QCConfig.from_dict(cfg.stage("quality_control"))
    out_dir = os.path.abspath(qcfg.out_dir)
    tts_manifest = os.path.join(out_dir, "tts_manifest.jsonl")
    out_manifest = os.path.join(out_dir, "dataset_manifest.jsonl")

    utts = [r for r in read_jsonl(tts_manifest) if r.get("status") == "synthesized"]
    done = load_done(out_manifest, key="utt_id")
    pending = [r for r in utts if r["utt_id"] not in done]
    logger.info("%d utterances, %d already QC'd, %d to check", len(utts), len(done), len(pending))

    t0 = time.time()
    stats = {"checked": 0, "passed": 0, "failed": 0}
    n_pending = len(pending)
    for i, r in enumerate(pending, 1):
        synth_path = os.path.join(out_dir, r["audio_filepath"])
        ref_path = os.path.join(out_dir, r["ref_audio_path"]) if r.get("ref_audio_path") else None
        result = validate_utterance(
            r["utt_id"], synth_path, r.get("text", ""), r.get("language", ""),
            ref_path=ref_path, cfg=qcfg, run_asr=run_asr, run_speaker=run_speaker)

        row = {**r,
               "qc_passed": result.passed,
               "qc_reasons": result.reasons,
               "asr_transcript": result.transcript,
               "cer": _metric(result, "cer"),
               "speaker_sim": _metric(result, "speaker_cos"),
               "qc_checks": [c.__dict__ for c in result.checks],
               "status": "validated"}
        append_jsonl(out_manifest, row)
        stats["checked"] += 1
        stats["passed" if result.passed else "failed"] += 1

        if i % qcfg.log_every == 0 or i == n_pending:
            rate = i / max(time.time() - t0, 1e-9)             # utts/sec
            eta_min = (n_pending - i) / max(rate, 1e-9) / 60
            logger.info("[%d/%d] %.1f utt/min, %d pass / %d fail (%.0f%% pass), ETA ~%.0f min",
                        i, n_pending, rate * 60, stats["passed"], stats["failed"],
                        100 * stats["passed"] / max(stats["checked"], 1), eta_min)

    summary = {
        "stage": "quality_control",
        "elapsed_sec": round(time.time() - t0, 2),
        "thresholds": {"cer_max": qcfg.cer_max, "spk_cos_min": qcfg.spk_cos_min,
                       "dur_per_char": [qcfg.dur_per_char_min, qcfg.dur_per_char_max]},
        **stats,
        "pass_rate": round(stats["passed"] / max(stats["checked"], 1), 4),
        "dataset_manifest": out_manifest,
    }
    with open(os.path.join(out_dir, "qc_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("QC done: %s", json.dumps(summary, ensure_ascii=False))
    return summary
