"""smoke_acquire.py — Step 1 smoke test for data_acquisition (§4.2).

Builds a tiny synthetic `fake_kathbath` parquet dataset (the real Kathbath schema,
with genuine WAV bytes in the audio column), then runs the full stage against
source="local" — exercising catalog -> sample -> pull end to end with no gated
HF download. Asserts the reference manifest + downloaded clips exist and match.

Run:  python scripts/smoke_acquire.py
"""
from __future__ import annotations

import io
import os
import sys
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from indic_synth.common.config import Config  # noqa: E402
from indic_synth.common.logging import get_logger  # noqa: E402
from indic_synth.common.manifest import read_jsonl  # noqa: E402
from indic_synth.data_acquisition import pipeline  # noqa: E402


def _wav_bytes(seconds: float = 4.0, sr: int = 16000) -> bytes:
    """A short sine tone encoded as 16 kHz mono WAV (mimics a Kathbath clip)."""
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    y = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _make_fake_kathbath(root: str, langs, speakers_per_lang=4, clips_per_speaker=3):
    """Write fake_kathbath/<lang>/valid-0.parquet with the Kathbath schema."""
    for lang in langs:
        rows = {k: [] for k in
                ["fname", "text", "audio_filepath", "lang", "duration", "gender", "speaker_id"]}
        for s in range(speakers_per_lang):
            gender = "male" if s % 2 == 0 else "female"  # balanced
            for c in range(clips_per_speaker):
                rows["fname"].append(f"{lang}_{s}_{c}.wav")
                rows["text"].append(f"sample reference text {s}-{c}")
                rows["audio_filepath"].append({"bytes": _wav_bytes(), "path": f"{lang}_{s}_{c}.wav"})
                rows["lang"].append(lang)
                rows["duration"].append(4.0)
                rows["gender"].append(gender)
                rows["speaker_id"].append(s)
        schema = pa.schema([
            ("fname", pa.string()), ("text", pa.string()),
            ("audio_filepath", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
            ("lang", pa.string()), ("duration", pa.float64()),
            ("gender", pa.string()), ("speaker_id", pa.int64()),
        ])
        d = os.path.join(root, lang)
        os.makedirs(d, exist_ok=True)
        pq.write_table(pa.table(rows, schema=schema), os.path.join(d, "valid-0.parquet"))


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def main():
    logger = get_logger("smoke-acquire")
    langs = ["hindi", "tamil"]
    with tempfile.TemporaryDirectory() as tmp:
        fake = os.path.join(tmp, "fake_kathbath")
        out = os.path.join(tmp, "out")
        _make_fake_kathbath(fake, langs, speakers_per_lang=4, clips_per_speaker=3)

        cfg = Config(seed=1234, out_dir=out, raw={"data_acquisition": {
            "source": "local", "local_dir": fake, "languages": langs, "split": "valid",
            "speakers_per_language": 2, "clips_per_speaker": 1, "min_total_speakers": 4,
            "gender_balance": True, "ref_min_dur": 1.0, "ref_max_dur": 15.0,
        }})

        summary = pipeline.run(cfg, logger)

        ref_path = os.path.join(out, "reference_manifest.jsonl")
        check(os.path.exists(ref_path), "reference_manifest.jsonl written")
        rows = [r for r in read_jsonl(ref_path) if r.get("status") == "downloaded"]
        check(len(rows) == 4, f"4 clips pulled (2 spk x 2 langs x 1 clip) — got {len(rows)}")
        for r in rows:
            wav = os.path.join(out, r["local_audio_path"])
            check(os.path.exists(wav), f"audio on disk: {r['ref_id']}")
        check(summary["selection"]["total_speakers"] == 4, "4 distinct speakers selected")
        genders = {r["gender"] for r in rows}
        check(genders == {"male", "female"}, f"gender-balanced selection — got {genders}")

        # resumability: a second run pulls nothing new
        s2 = pipeline.run(cfg, logger)
        check(s2["pull"]["downloaded"] == 0, "re-run is a resumable no-op (0 new downloads)")

    print("\nSMOKE OK: data_acquisition (§4.2)")


if __name__ == "__main__":
    main()
