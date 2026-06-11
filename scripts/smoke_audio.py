"""smoke_audio.py — Step 2 smoke test for audio_engineering (§4.3).

Writes a tiny reference_manifest.jsonl + raw 16 kHz / stereo clips (mimicking §4.2
output), runs the prepare stage, and asserts every prepared clip is 24 kHz mono
with the expected QC fields (resampled_up flag, manifest metrics). Also checks
resumability.

Run:  python scripts/smoke_audio.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from indic_synth.audio_engineering import pipeline  # noqa: E402
from indic_synth.common.config import Config  # noqa: E402
from indic_synth.common.logging import get_logger  # noqa: E402
from indic_synth.common.manifest import append_jsonl, read_jsonl  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def main():
    logger = get_logger("smoke-audio")
    with tempfile.TemporaryDirectory() as out:
        ref_dir = os.path.join(out, "ref_audio")
        os.makedirs(ref_dir, exist_ok=True)
        ref_manifest = os.path.join(out, "reference_manifest.jsonl")

        # Two raw clips at 16 kHz (one mono, one stereo) — like a downloaded Kathbath clip.
        for i, ch in enumerate([1, 2]):
            sr = 16000
            t = np.linspace(0, 4.0, sr * 4, endpoint=False)
            y = 0.2 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
            y = y if ch == 1 else np.stack([y, y], axis=1)
            rel = os.path.join("ref_audio", f"clip{i}.wav")
            sf.write(os.path.join(out, rel), y, sr, subtype="PCM_16")
            append_jsonl(ref_manifest, {
                "ref_id": f"clip{i}", "lang": "hindi", "speaker_id": i, "gender": "male",
                "ref_text": "reference text", "duration": 4.0,
                "local_audio_path": rel, "status": "downloaded",
            })

        cfg = Config(seed=1234, out_dir=out, raw={"audio_engineering": {
            "target_sr": 24000, "norm": "peak", "peak_dbfs": -1.0,
        }})

        summary = pipeline.run(cfg, logger)
        check(summary["prepared"] == 2 and summary["failed"] == 0, "2 prepared, 0 failed")

        rows = [r for r in read_jsonl(os.path.join(out, "prepared_manifest.jsonl"))
                if r.get("status") == "prepared"]
        check(len(rows) == 2, "prepared_manifest has 2 rows")
        for r in rows:
            wav = os.path.join(out, r["prepared_audio_path"])
            info = sf.info(wav)
            check(info.samplerate == 24000, f"{r['ref_id']}: written at 24 kHz")
            check(info.channels == 1, f"{r['ref_id']}: mono")
            check("resampled_up" in r["qc_flags"], f"{r['ref_id']}: flagged resampled_up (16k->24k)")
            check(r["out_peak_dbfs"] <= 0.0, f"{r['ref_id']}: peak-normalized (<= ceiling)")

        s2 = pipeline.run(cfg, logger)
        check(s2["prepared"] == 0, "re-run is a resumable no-op (0 reprocessed)")

    print("\nSMOKE OK: audio_engineering (§4.3)")


if __name__ == "__main__":
    main()
