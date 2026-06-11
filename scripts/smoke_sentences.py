"""smoke_sentences.py — Step 3 smoke test for sentence_generation (§4.4).

Runs entirely on CPU (no Gemma / LaBSE / fastText download):
  - tests the pure validation gates (script ratio, numbers, degeneracy, norm_key)
  - drives the full grid/quota/checkpoint loop (generate_pool) with a FAKE LLM
    generator + fake langid/embed resources, asserting balance, quota, rejection
    logging, and resumability.

The real `pipeline.run` (which loads Gemma) is GPU-only and exercised in Colab.

Run:  python scripts/smoke_sentences.py
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from indic_synth.common.logging import get_logger  # noqa: E402
from indic_synth.common.manifest import read_jsonl  # noqa: E402
from indic_synth.sentence_generation import pipeline, validation  # noqa: E402
from indic_synth.sentence_generation.config import SentenceConfig  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


# ---- fakes (stand in for fastText langid + LaBSE embeddings) ----
class FakeLangId:
    """Classify by dominant script range — enough to mimic lid.176 on clean text."""
    def predict(self, text: str):
        dev = sum(0x0900 <= ord(c) <= 0x097F for c in text)
        mal = sum(0x0D00 <= ord(c) <= 0x0D7F for c in text)
        return ("hi" if dev >= mal else "ml"), 1.0


class FakeEmbed:
    """Deterministic near-orthogonal unit vectors keyed on the normalized text, so
    only exact repeats collide; distinct sentences stay well below threshold."""
    def encode(self, texts, normalize_embeddings=True):
        out = []
        for t in texts:
            h = hashlib.sha256(validation.norm_key(t).encode()).digest()
            v = np.frombuffer(h, dtype=np.uint8).astype(np.float32)[:32] - 127.5
            out.append(v / (np.linalg.norm(v) + 1e-9))
        return np.array(out, dtype=np.float32)


VALID = ["आज मौसम बहुत अच्छा है", "मुझे चाय पीना पसंद है",
         "बाज़ार में बहुत भीड़ थी", "हम कल यात्रा पर जाएँगे"]


def fake_generator(prompt, seed):
    # bad candidates first so they're seen before the quota fills
    return "\n".join(["नमस्ते", VALID[0], "मेरे पास 5 रुपये हैं", VALID[1], VALID[2], VALID[3]])


def test_validators():
    print("[validation] pure gates")
    cfg = SentenceConfig()
    check(validation.script_ratio("आज मौसम अच्छा है", "Devanagari") == 1.0, "Devanagari ratio = 1.0")
    check(validation.script_ratio("hello world", "Devanagari") == 0.0, "Latin -> 0.0 Devanagari")
    check(validation.has_unnormalized_numbers("मेरे पास 5 रुपये"), "digit detected")
    check(validation.has_unnormalized_numbers("₹ सौ रुपये"), "currency symbol detected")
    check(not validation.has_unnormalized_numbers("सौ रुपये"), "clean text -> no number")
    check(validation.is_degenerate("नमस्ते", cfg)[1] == "too_short", "1 word -> too_short")
    check(validation.is_degenerate("राम राम राम राम", cfg)[1] in ("low_diversity", "repeated_word"),
          "repetition -> degenerate")
    check(not validation.is_degenerate(VALID[0], cfg)[0], "good sentence not degenerate")
    check(validation.norm_key("आज,  मौसम!") == validation.norm_key("आज मौसम"),
          "norm_key drops punct + collapses whitespace")


def test_generate_pool(tmp):
    print("[pipeline] grid loop + checkpoint")
    cfg = SentenceConfig(
        languages={"hi": ["Hindi", "Devanagari"]},
        topics=["Daily Commute"], sentence_types=["declarative"],
        per_cell_quota=2, sentences_per_call=6, max_attempts_per_cell=3,
        out_dir=tmp,
    )
    dd = validation.Deduper(FakeEmbed(), cfg.near_dup_threshold)
    summary = pipeline.generate_pool(cfg, fake_generator, dd, FakeLangId(), get_logger("smoke-gen"))

    check(summary["total_valid"] == 2, f"quota met: 2 valid (got {summary['total_valid']})")
    rows = read_jsonl(cfg.pool_path)
    check(len(rows) == 2, "sentences.jsonl has 2 rows")
    check(all(r["language"] == "hi" and r["topic"] == "Daily Commute" for r in rows),
          "rows carry correct language/topic/type")
    rej = summary["rejection_stats"]
    check(rej.get("too_short", 0) >= 1, "too_short rejection logged")
    check(rej.get("unnormalized_number", 0) >= 1, "unnormalized_number rejection logged")

    # resume: fresh resources, same out_dir -> no new work
    dd2 = validation.Deduper(FakeEmbed(), cfg.near_dup_threshold)
    s2 = pipeline.generate_pool(cfg, fake_generator, dd2, FakeLangId(), get_logger("smoke-gen"))
    check(s2["total_valid"] == 2 and len(read_jsonl(cfg.pool_path)) == 2,
          "resume is a no-op (still 2, cell already at quota)")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        test_validators()
        test_generate_pool(tmp)
    print("\nSMOKE OK: sentence_generation (§4.4)")


if __name__ == "__main__":
    main()
