"""smoke_qc.py — Step 5 smoke test for quality_control (§4.6).

CPU-only (ASR + speaker models off): tests the model-free gates directly
(silence, clipping, dur-per-char, CER, text-repetition) and drives the full
orchestration over a tts_manifest.jsonl, asserting the final dataset_manifest.jsonl
carries qc_passed + per-gate results, and that re-running resumes.

The content/speaker gates (run_asr/run_speaker) need the real models and are
exercised in Colab.

Run:  python scripts/smoke_qc.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from indic_synth.common.config import Config  # noqa: E402
from indic_synth.common.logging import get_logger  # noqa: E402
from indic_synth.common.manifest import append_jsonl, read_jsonl  # noqa: E402
from indic_synth.quality_control import checks as C  # noqa: E402
from indic_synth.quality_control import pipeline  # noqa: E402
from indic_synth.quality_control.config import QCConfig  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def speech_like(seconds=1.2, sr=16000):
    """Envelope-modulated broadband noise with a decay tail — non-repeating, so it
    passes silence/clipping/truncation AND the acoustic-loop gate (a pure tone would
    be perfectly self-similar and correctly flagged as looping)."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)           # syllable-rate amplitude wobble
    y = 0.3 * env * rng.standard_normal(t.shape)
    y[-int(0.2 * sr):] *= np.linspace(1, 0, int(0.2 * sr))  # natural decay tail
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.6).astype(np.float32)


def test_gates():
    print("[checks] model-free gates")
    cfg = QCConfig()
    sr = 16000
    speech = speech_like()
    check(C.check_silence(speech, sr, cfg).passed, "speech-like clip passes silence")
    check(not C.check_silence(np.zeros(sr, np.float32), sr, cfg).passed, "zeros -> silence reject")

    clipped = np.ones(sr, np.float32)  # full-scale, long clipped run
    check(not C.check_clipping(clipped, sr, cfg).passed, "full-scale -> clipping reject")
    check(C.check_clipping(speech, sr, cfg).passed, "speech-like passes clipping")

    # dur-per-char: 1.2 s over a short text -> within [0.04, 0.30]? pick text length ~10 chars
    r_ok = C.check_duration_per_char(speech, sr, "दस अक्षर शब्द", cfg)
    check(r_ok.name == "dur_per_char", "dur_per_char returns a result")
    r_bad = C.check_duration_per_char(speech, sr, "क", cfg)   # 1.2s/1char = 1.2 -> out of range
    check(not r_bad.passed, "1.2s for 1 char -> dur_per_char reject")

    check(C.compute_cer("नमस्ते दुनिया", "नमस्ते दुनिया") == 0.0, "identical text -> CER 0")
    check(C.compute_cer("नमस्ते", "अलविदा") > 0.0, "different text -> CER > 0")
    check(not C.detect_text_repetition("राम राम राम राम राम", cfg).passed, "repeated tokens -> looping_text")
    check(C.detect_text_repetition("आज मौसम बहुत अच्छा है", cfg).passed, "normal text -> no loop")


def test_pipeline(tmp):
    print("[pipeline] orchestration over tts_manifest (models off)")
    sr = 16000
    audio_dir = os.path.join(tmp, "tts_audio")
    os.makedirs(audio_dir, exist_ok=True)
    tts_manifest = os.path.join(tmp, "tts_manifest.jsonl")

    # one good clip, one silent (should fail hard-gate)
    for utt, wav, text in [("good", speech_like(), "दस अक्षर शब्द यहाँ"),
                           ("silent", np.zeros(sr, np.float32), "दस अक्षर शब्द यहाँ")]:
        rel = os.path.join("tts_audio", f"{utt}.wav")
        sf.write(os.path.join(tmp, rel), wav, sr)
        append_jsonl(tts_manifest, {
            "utt_id": utt, "audio_filepath": rel, "text": text, "language": "hi",
            "speaker_id": 0, "gender": "male", "ref_id": "r", "ref_audio_path": rel,
            "duration": len(wav) / sr, "status": "synthesized",
        })

    cfg = Config(seed=1234, out_dir=tmp, raw={"quality_control": {"device": "cpu"}})
    summary = pipeline.run(cfg, get_logger("smoke-qc"), run_asr=False, run_speaker=False)
    check(summary["checked"] == 2, "2 utterances checked")

    rows = {r["utt_id"]: r for r in read_jsonl(os.path.join(tmp, "dataset_manifest.jsonl"))}
    check(rows["good"]["qc_passed"] is True, "good clip passes QC")
    check(rows["silent"]["qc_passed"] is False, "silent clip fails QC")
    check(any("silence" in reason for reason in rows["silent"]["qc_reasons"]),
          "silent clip rejected for silence")
    check("qc_checks" in rows["good"] and len(rows["good"]["qc_checks"]) >= 4,
          "per-gate check details recorded")

    s2 = pipeline.run(cfg, get_logger("smoke-qc"), run_asr=False, run_speaker=False)
    check(s2["checked"] == 0, "re-run is a resumable no-op (0 re-checked)")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        test_gates()
        test_pipeline(tmp)
    print("\nSMOKE OK: quality_control (§4.6)")


if __name__ == "__main__":
    main()
