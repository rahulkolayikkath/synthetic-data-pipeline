"""smoke_e2e.py — full-chain smoke test through all six modules (§4.1–4.6).

Threads every stage through ONE shared out_dir to prove the manifest chain wires up
end to end and yields a final dataset_manifest.jsonl:

  §4.2 acquisition  (real, local fake_kathbath)  -> reference_manifest.jsonl
  §4.3 audio eng    (real DSP)                    -> prepared_manifest.jsonl
  §4.4 sentences    (fake LLM + fake langid/embed) -> sentences.jsonl
  §4.5 tts          (fake synthesizer)            -> tts_manifest.jsonl
  §4.6 qc           (real DSP gates, models off)  -> dataset_manifest.jsonl

GPU stages use injected fakes (the real Gemma/IndicF5/conformer/ECAPA paths run in
Colab); everything else is the real code. The point is the wiring + lang pairing.

Run:  python scripts/smoke_e2e.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from indic_synth.common.config import Config  # noqa: E402
from indic_synth.common.logging import get_logger  # noqa: E402
from indic_synth.common.manifest import read_jsonl  # noqa: E402
from indic_synth.data_acquisition import pipeline as acq  # noqa: E402
from indic_synth.audio_engineering import pipeline as audio  # noqa: E402
from indic_synth.sentence_generation import pipeline as gen, validation  # noqa: E402
from indic_synth.sentence_generation.config import SentenceConfig  # noqa: E402
from indic_synth.tts_generation import pipeline as tts  # noqa: E402
from indic_synth.tts_generation.config import TTSConfig  # noqa: E402
from indic_synth.quality_control import pipeline as qc  # noqa: E402

# reuse fakes / fixtures from the per-stage smokes
from smoke_acquire import _make_fake_kathbath  # noqa: E402
from smoke_sentences import FakeEmbed, FakeLangId  # noqa: E402
from smoke_tts import fake_synth  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


# bilingual fake generator: the validation gate keeps only the cell-matching language
_HI = ["आज मौसम बहुत अच्छा है", "मुझे चाय पीना पसंद है", "हम कल यात्रा पर जाएँगे"]
_ML = ["എനിക്ക് ചായ കുടിക്കാൻ ഇഷ്ടമാണ്", "ഇന്ന് കാലാവസ്ഥ വളരെ നല്ലതാണ്", "ഞങ്ങൾ നാളെ യാത്ര പോകും"]


def bilingual_generator(prompt, seed):
    return "\n".join(_HI + _ML)


def main():
    logger = get_logger("e2e")
    langs = ["hindi", "malayalam"]
    with tempfile.TemporaryDirectory() as out:
        fake = os.path.join(out, "fake_kathbath")
        _make_fake_kathbath(fake, langs, speakers_per_lang=2, clips_per_speaker=2)

        cfg = Config(seed=1234, out_dir=out, raw={
            "data_acquisition": {
                "source": "local", "local_dir": fake, "languages": langs, "split": "valid",
                "speakers_per_language": 2, "clips_per_speaker": 2, "min_total_speakers": 4,
                "gender_balance": True, "ref_min_dur": 1.0, "ref_max_dur": 15.0,
            },
            "audio_engineering": {"target_sr": 24000, "norm": "peak"},
            "quality_control": {"device": "cpu"},
        })

        # §4.2 + §4.3 (real)
        print("[e2e] §4.2 acquisition + §4.3 audio engineering")
        acq.run(cfg, logger)
        audio.run(cfg, logger)
        prepared = [r for r in read_jsonl(os.path.join(out, "prepared_manifest.jsonl"))
                    if r.get("status") == "prepared"]
        check(len(prepared) == 8, f"8 prepared refs (2 langs x 2 spk x 2 clips) — got {len(prepared)}")

        # §4.4 (fake LLM)
        print("[e2e] §4.4 sentence generation (fake LLM)")
        scfg = SentenceConfig(
            languages={"hi": ["Hindi", "Devanagari"], "ml": ["Malayalam", "Malayalam"]},
            topics=["Daily Commute"], sentence_types=["declarative"],
            per_cell_quota=2, sentences_per_call=6, max_attempts_per_cell=3, out_dir=out)
        dd = validation.Deduper(FakeEmbed(), scfg.near_dup_threshold)
        gen.generate_pool(scfg, bilingual_generator, dd, FakeLangId(), logger)
        sents = read_jsonl(os.path.join(out, "sentences.jsonl"))
        by_lang = {}
        for s in sents:
            by_lang[s["language"]] = by_lang.get(s["language"], 0) + 1
        check(by_lang.get("hi") == 2 and by_lang.get("ml") == 2,
              f"balanced sentence pool: 2 hi + 2 ml — got {by_lang}")

        # §4.5 (fake synth)
        print("[e2e] §4.5 TTS (fake synthesizer)")
        tcfg = TTSConfig(out_dir=out, target_sr=24000, max_chars=300,
                         lang_map={"hi": "hindi", "ml": "malayalam"})
        tts.synthesize_pool(tcfg, fake_synth, logger)
        utts = [r for r in read_jsonl(os.path.join(out, "tts_manifest.jsonl"))
                if r.get("status") == "synthesized"]
        check(len(utts) == 4, f"4 utterances synthesized — got {len(utts)}")
        for u in utts:
            want = "hindi" if u["language"] == "hi" else "malayalam"
            check(u["ref_id"].startswith(want), f"{u['utt_id']}: same-language pairing ({u['ref_id']})")

        # §4.6 (real DSP gates, models off)
        print("[e2e] §4.6 quality control (DSP gates)")
        summary = qc.run(cfg, logger, run_asr=False, run_speaker=False)
        final = read_jsonl(os.path.join(out, "dataset_manifest.jsonl"))
        check(len(final) == 4, "dataset_manifest has 4 validated rows")
        check(all("qc_passed" in r for r in final), "every final row carries qc_passed")
        check(all(r.get("speaker_id") is not None and r.get("text") for r in final),
              "final rows carry §4.1 schema (speaker_id, text, ...)")
        print(f"  -> final QC pass rate: {summary['pass_rate']:.0%} ({summary['passed']}/{summary['checked']})")

    print("\nSMOKE OK: end-to-end (§4.2 -> §4.6 manifest chain)")


if __name__ == "__main__":
    main()
