"""inspect_manifest.py — quick summary of any pipeline manifest (Colab inspect cells).

    python scripts/inspect_manifest.py <manifest.jsonl> [--n 5]

(Named inspect_manifest, not inspect, so it never shadows Python's stdlib `inspect`
when scripts/ lands on sys.path.)

Prints the row count, value counts for categorical fields (status, language,
gender, qc_passed, ...), min/mean/max for numeric fields (duration, cer,
speaker_sim, ...), and a few sample rows. Stdlib only — no pandas needed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from indic_synth.common.manifest import read_jsonl  # noqa: E402

CATEGORICAL = ["status", "lang", "language", "gender", "qc_passed",
               "sentence_type", "topic", "resample_method", "decode_backend"]
NUMERIC = ["duration", "out_duration", "source_sr", "out_peak_dbfs", "out_rms_dbfs",
           "cer", "speaker_sim", "word_count", "char_count"]
# long / noisy fields to trim out of sample-row printing
HIDE_IN_SAMPLE = ["qc_checks", "source_file", "ref_audio_path", "asr_transcript"]


def _num_summary(vals):
    vals = [float(v) for v in vals if isinstance(v, (int, float))]
    if not vals:
        return None
    return min(vals), sum(vals) / len(vals), max(vals)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--n", type=int, default=3, help="sample rows to print")
    args = ap.parse_args(argv)

    rows = read_jsonl(args.manifest)
    print(f"\n=== {args.manifest} ===")
    print(f"rows: {len(rows)}")
    if not rows:
        return

    keys = set().union(*(r.keys() for r in rows))

    print("\n-- categorical --")
    for k in CATEGORICAL:
        if k in keys:
            c = Counter(r.get(k) for r in rows if k in r)
            print(f"  {k:16s} {dict(c.most_common())}")

    print("\n-- numeric (min / mean / max) --")
    for k in NUMERIC:
        if k in keys:
            s = _num_summary([r.get(k) for r in rows if r.get(k) is not None])
            if s:
                print(f"  {k:16s} {s[0]:.3f} / {s[1]:.3f} / {s[2]:.3f}")

    # distinct speakers, if present
    if "speaker_id" in keys:
        spk = {(r.get("language", r.get("lang")), r.get("speaker_id")) for r in rows}
        print(f"\ndistinct (lang, speaker): {len(spk)}")

    print(f"\n-- {min(args.n, len(rows))} sample rows --")
    for r in rows[:args.n]:
        trimmed = {k: v for k, v in r.items() if k not in HIDE_IN_SAMPLE}
        print("  " + json.dumps(trimmed, ensure_ascii=False)[:400])
    print()


if __name__ == "__main__":
    main()
