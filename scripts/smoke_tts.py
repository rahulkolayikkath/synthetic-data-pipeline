"""smoke_tts.py — Step 4 smoke test for tts_generation (§4.5).

CPU-only (no IndicF5 load): writes a prepared_manifest.jsonl (refs in 2 languages,
2 speakers each) + sentences.jsonl, then drives synthesize_pool with a FAKE
synthesizer. Asserts:
  - each sentence is paired with a reference in its OWN language (no cross-lingual)
  - speakers are round-robined (both speakers used) for balance
  - output wavs are 24 kHz and the manifest carries the §4.1 utterance fields
  - long text is chunked at sentence boundaries; re-run is a resumable no-op

The real `pipeline.run` (loads IndicF5) is GPU-only and exercised in Colab.

Run:  python scripts/smoke_tts.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from indic_synth.common.logging import get_logger  # noqa: E402
from indic_synth.common.manifest import append_jsonl, read_jsonl  # noqa: E402
from indic_synth.tts_generation import pipeline  # noqa: E402
from indic_synth.tts_generation.config import TTSConfig  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def fake_synth(text, ref_audio_path, ref_text):
    """A sine whose length scales with text length — stands in for IndicF5 @ 24 kHz."""
    assert os.path.exists(ref_audio_path), f"ref audio must exist: {ref_audio_path}"
    n = int(24000 * max(0.5, 0.06 * len(text)))
    t = np.linspace(0, n / 24000, n, endpoint=False)
    return (0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


def setup(out):
    os.makedirs(os.path.join(out, "prepared_audio"), exist_ok=True)
    prepared = os.path.join(out, "prepared_manifest.jsonl")
    # 2 languages x 2 speakers prepared refs
    for lang in ["hindi", "malayalam"]:
        for spk in [0, 1]:
            rel = os.path.join("prepared_audio", f"{lang}_spk{spk}.wav")
            sf.write(os.path.join(out, rel), np.zeros(24000, np.float32), 24000)
            append_jsonl(prepared, {
                "ref_id": f"{lang}_spk{spk}", "lang": lang, "speaker_id": spk,
                "gender": "male" if spk == 0 else "female", "ref_text": "ref",
                "prepared_audio_path": rel, "status": "prepared",
            })
    # sentences: 4 hi + 2 ml, one very long hi to exercise chunking
    sents = os.path.join(out, "sentences.jsonl")
    long_hi = "वाक्य एक है। वाक्य दो है। " * 30
    rows = (
        [{"id": f"hi_{i:06d}", "language": "hi", "topic": "t", "sentence_type": "declarative",
          "text": f"हिन्दी वाक्य संख्या {i}"} for i in range(3)]
        + [{"id": "hi_000003", "language": "hi", "topic": "t", "sentence_type": "declarative",
            "text": long_hi}]
        + [{"id": f"ml_{i:06d}", "language": "ml", "topic": "t", "sentence_type": "declarative",
            "text": f"മലയാളം വാക്യം {i}"} for i in range(2)]
    )
    for r in rows:
        append_jsonl(sents, r)


def main():
    logger = get_logger("smoke-tts")

    # chunker unit check
    chunks = pipeline.chunk_text("क ख ग। " * 50, max_chars=100)
    check(len(chunks) > 1 and all(len(c) <= 100 for c in chunks), "long text chunked <=max_chars")
    check(pipeline.chunk_text("छोटा वाक्य", 100) == ["छोटा वाक्य"], "short text -> single chunk")

    with tempfile.TemporaryDirectory() as out:
        setup(out)
        cfg = TTSConfig(out_dir=out, target_sr=24000, max_chars=100,
                        lang_map={"hi": "hindi", "ml": "malayalam"})
        summary = pipeline.synthesize_pool(cfg, fake_synth, logger)

        check(summary["synthesized"] == 6 and summary["failed"] == 0, "6 synthesized, 0 failed")
        rows = [r for r in read_jsonl(os.path.join(out, "tts_manifest.jsonl"))
                if r.get("status") == "synthesized"]
        check(len(rows) == 6, "tts_manifest has 6 utterances")

        # same-language pairing: hi sentences -> hindi refs, ml -> malayalam refs
        by = {r["utt_id"]: r for r in rows}
        for utt, r in by.items():
            want = "hindi" if r["language"] == "hi" else "malayalam"
            check(r["ref_id"].startswith(want), f"{utt}: paired with {want} ref ({r['ref_id']})")

        # speaker round-robin: both hindi speakers used across the 4 hi sentences
        hi_spk = {by[f"hi_{i:06d}"]["speaker_id"] for i in range(4)}
        check(hi_spk == {0, 1}, f"hindi speakers round-robined -> {hi_spk}")

        # 24 kHz outputs + §4.1 fields present
        for r in rows:
            info = sf.info(os.path.join(out, r["audio_filepath"]))
            check(info.samplerate == 24000, f"{r['utt_id']}: 24 kHz")
            for fld in ["text", "speaker_id", "gender", "language", "ref_id", "duration"]:
                check(fld in r, f"{r['utt_id']}: manifest has '{fld}'") if r["utt_id"] == "hi_000000" else None

        s2 = pipeline.synthesize_pool(cfg, fake_synth, logger)
        check(s2["synthesized"] == 0, "re-run is a resumable no-op (0 re-synthesized)")

    print("\nSMOKE OK: tts_generation (§4.5)")


if __name__ == "__main__":
    main()
