"""
generate_report.py — one consolidated post-run report for the synthetic-speech pipeline.

Reads the per-stage `*_summary.json` files and the final `dataset_manifest.jsonl`
(all written into `out_dir`) and emits a single Markdown report covering:

    1. Datacard          — per-language / per-speaker / gender counts + duration &
                           length distributions, and the manifest field dictionary.
    2. Quality report    — QC yield + rejection breakdown (QC stage only), the list of
                           QC checks run, the separate sentence-generation text-validation
                           stats, and stage-wise pass/fail rates.
    3. Throughput & cost — per-stage utterances/min and T4 GPU-hours, plus totals.

Run from a Colab cell:
    python scripts/generate_report.py --config config.yaml
    python scripts/generate_report.py --out_dir /content/drive/MyDrive/indic_synth/out_representative_sample

The report is written to <out_dir>/REPORT.md and also printed to stdout.
Stages can be run one-by-one, so any missing summary/manifest is reported as
"unavailable" instead of crashing. Stdlib only — no pandas/numpy needed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from indic_synth.common.manifest import read_jsonl  # noqa: E402

# ---- canonical filenames each stage writes into out_dir ----
F_ACQUIRE = "data_acquisition_run_summary.json"
F_AUDIO = "prepare_refaudio_summary.json"
F_SENT = "sentences_summary.json"
F_TTS = "tts_summary.json"
F_QC = "qc_summary.json"
F_DATASET = "dataset_manifest.jsonl"

# ---- dataset_manifest field dictionary (§4.1 manifest schema) ----
FIELD_DOC = [
    ("utt_id", "unique utterance id (e.g. hi_000000)"),
    ("audio_filepath", "path to the synthesized wav, relative to out_dir"),
    ("text", "intended sentence the TTS model was asked to speak"),
    ("speaker_id", "Kathbath speaker whose voice was cloned"),
    ("gender", "speaker gender (male / female)"),
    ("language", "language code (hi / ml / ta)"),
    ("ref_id", "reference clip used for voice conditioning"),
    ("ref_audio_path", "prepared 24kHz mono reference wav, relative to out_dir"),
    ("duration", "synthesized audio duration in seconds"),
    ("topic", "sentence-generation topic cell"),
    ("sentence_type", "declarative / interrogative / imperative / exclamatory"),
    ("status", "pipeline status (validated)"),
    ("qc_passed", "True if the utterance passed all QC gates"),
    ("qc_reasons", "list of failed check names ([] when passed)"),
    ("asr_transcript", "what the ASR model heard (for the CER gate)"),
    ("cer", "character error rate vs intended text (content fidelity)"),
    ("speaker_sim", "cosine similarity of speaker embeddings synth-vs-ref"),
    ("qc_checks", "per-check detail: {name, passed, metric, detail}"),
]

# ---- human-readable description of each QC check ----
CHECK_DOC = {
    "silence": "rejects near-silent / mostly-inactive audio",
    "clipping": "rejects audio with excessive clipped samples",
    "truncation": "rejects clips that look cut off (loud tail energy)",
    "dur_per_char": "duration-to-char ratio must fall in [0.04, 0.30] s/char",
    "looping_audio": "rejects repeated/looping audio segments",
    "cer": "content fidelity — ASR CER vs intended text (gate < 0.15)",
    "looping_text": "rejects repeated tokens in the ASR transcript",
    "speaker_cos": "speaker fidelity — cosine sim synth-vs-ref (gate >= 0.60)",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _load_json(path):
    """Return parsed JSON, or None if the file is missing/unreadable."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _read_jsonl_safe(path):
    """Return rows, or None if the manifest is missing."""
    if not os.path.exists(path):
        return None
    return read_jsonl(path)


def _num(vals):
    """(min, mean, max) over numeric values, or None if empty."""
    vals = [float(v) for v in vals if isinstance(v, (int, float))]
    if not vals:
        return None
    return min(vals), sum(vals) / len(vals), max(vals)


def _table(headers, rows):
    """Render a GitHub-markdown table."""
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out


def _pct(num, den):
    return f"{100 * num / den:.1f}%" if den else "n/a"


# --------------------------------------------------------------------------- #
# section 1 — datacard
# --------------------------------------------------------------------------- #
def section_datacard(out_dir):
    L = ["## 1. Datacard — dataset statistics", ""]
    rows = _read_jsonl_safe(os.path.join(out_dir, F_DATASET))
    if not rows:
        L += [f"_Unavailable — `{F_DATASET}` not found in out_dir._", ""]
        return L

    passed = [r for r in rows if r.get("qc_passed")]
    speakers = {(r.get("language"), r.get("speaker_id")) for r in rows}
    langs = sorted({r.get("language") for r in rows})
    dur_all = _num([r.get("duration") for r in rows])
    total_sec = sum(float(r.get("duration") or 0) for r in passed)

    # headline
    L += _table(
        ["metric", "value"],
        [["utterances checked", len(rows)],
         ["validated (qc_passed=true)", f"{len(passed)}  ({_pct(len(passed), len(rows))})"],
         ["distinct languages", len(langs)],
         ["distinct (lang, speaker)", len(speakers)],
         ["validated audio", f"{total_sec:.1f} s  ({total_sec / 3600:.3f} h)"]])
    L.append("")

    # field dictionary
    present = set().union(*(r.keys() for r in rows))
    L += ["### Manifest fields (`dataset_manifest.jsonl`)", ""]
    for name, doc in FIELD_DOC:
        mark = "" if name in present else "  _(absent in this run)_"
        L.append(f"- **`{name}`** — {doc}{mark}")
    extra = sorted(present - {n for n, _ in FIELD_DOC})
    if extra:
        L.append(f"- _other fields present_: {', '.join('`%s`' % e for e in extra)}")
    L.append("")

    # per-language
    by_lang_total = Counter(r.get("language") for r in rows)
    by_lang_pass = Counter(r.get("language") for r in passed)
    lang_rows = []
    for lang in langs:
        spk = {r.get("speaker_id") for r in rows if r.get("language") == lang}
        g = {sp: None for sp in spk}
        for r in rows:
            if r.get("language") == lang:
                g[r.get("speaker_id")] = r.get("gender")
        male = sum(1 for v in g.values() if v == "male")
        female = sum(1 for v in g.values() if v == "female")
        lang_rows.append([lang, f"{by_lang_pass[lang]}/{by_lang_total[lang]}",
                          len(spk), f"{male}/{female}"])
    L += ["### Per-language", ""]
    L += _table(["language", "validated/total utts", "speakers", "M/F speakers"], lang_rows)
    L.append("")

    # per-speaker
    spk_counts = Counter((r.get("language"), r.get("speaker_id"), r.get("gender"))
                         for r in rows)
    spk_pass = Counter((r.get("language"), r.get("speaker_id")) for r in passed)
    spk_rows = []
    for (lang, sid, gen), n in sorted(spk_counts.items()):
        spk_rows.append([lang, sid, gen, f"{spk_pass[(lang, sid)]}/{n}"])
    L += ["### Per-speaker", ""]
    L += _table(["language", "speaker_id", "gender", "validated/total utts"], spk_rows)
    L.append("")

    # gender
    utt_gender = Counter(r.get("gender") for r in rows)
    spk_gender = Counter()
    seen = set()
    for r in rows:
        key = (r.get("language"), r.get("speaker_id"))
        if key not in seen:
            seen.add(key)
            spk_gender[r.get("gender")] += 1
    L += ["### Gender balance", ""]
    L += _table(["gender", "utterances", "speakers"],
                [[g, utt_gender.get(g, 0), spk_gender.get(g, 0)]
                 for g in sorted(utt_gender)])
    L.append("")

    # distributions (over validated set)
    L += ["### Distributions (min / avg / max) — over validated set", ""]
    dist_rows = []
    d = _num([r.get("duration") for r in passed])
    if d:
        dist_rows.append(["duration (s)", f"{d[0]:.2f}", f"{d[1]:.2f}", f"{d[2]:.2f}"])
    chars = _num([len(r.get("text", "")) for r in passed])
    if chars:
        dist_rows.append(["sentence length (chars)",
                          f"{chars[0]:.0f}", f"{chars[1]:.1f}", f"{chars[2]:.0f}"])
    words = _num([len(r.get("text", "").split()) for r in passed])
    if words:
        dist_rows.append(["sentence length (words)",
                          f"{words[0]:.0f}", f"{words[1]:.1f}", f"{words[2]:.0f}"])
    cer = _num([r.get("cer") for r in passed])
    if cer:
        dist_rows.append(["cer", f"{cer[0]:.3f}", f"{cer[1]:.3f}", f"{cer[2]:.3f}"])
    sim = _num([r.get("speaker_sim") for r in passed])
    if sim:
        dist_rows.append(["speaker_sim", f"{sim[0]:.3f}", f"{sim[1]:.3f}", f"{sim[2]:.3f}"])
    L += _table(["metric", "min", "avg", "max"], dist_rows)
    L += ["",
          f"_Distributions are over the {len(passed)} validated (qc_passed=true) "
          f"utterances — the deliverable. Counts tables above show validated/total so "
          f"failures stay visible. (Full-set duration min/avg/max: "
          f"{dur_all[0]:.2f}/{dur_all[1]:.2f}/{dur_all[2]:.2f} s.)_", ""]
    return L


# --------------------------------------------------------------------------- #
# section 2 — quality report
# --------------------------------------------------------------------------- #
def section_quality(out_dir):
    L = ["## 2. Quality report", ""]
    qc = _load_json(os.path.join(out_dir, F_QC))
    rows = _read_jsonl_safe(os.path.join(out_dir, F_DATASET))

    # --- QC yield (source of truth = qc_summary.json) ---
    L += ["### QC yield (audio gates)", ""]
    if qc:
        th = qc.get("thresholds", {})
        L += _table(
            ["metric", "value"],
            [["checked", qc.get("checked")],
             ["passed", qc.get("passed")],
             ["failed", qc.get("failed")],
             ["pass rate", qc.get("pass_rate")],
             ["cer_max", th.get("cer_max")],
             ["spk_cos_min", th.get("spk_cos_min")],
             ["dur_per_char", th.get("dur_per_char")]])
        L.append("")
    else:
        L += [f"_Unavailable — `{F_QC}` not found._", ""]

    # --- list of QC checks run ---
    check_names = []
    if rows:
        seen = []
        for c in rows[0].get("qc_checks", []):
            if c.get("name") not in seen:
                seen.append(c.get("name"))
        check_names = seen
    L += ["### QC checks run (3 gates + hard-failure checks)", ""]
    if check_names:
        for n in check_names:
            L.append(f"- **`{n}`** — {CHECK_DOC.get(n, 'check')}")
    else:
        L.append("_Unavailable — no qc_checks in manifest._")
    L.append("")

    # --- QC failure breakdown (QC stage only) ---
    L += ["### QC rejection breakdown (failed utterances only)", ""]
    if rows:
        failed = [r for r in rows if not r.get("qc_passed")]
        reasons = Counter()
        for r in failed:
            # qc_reasons are "<check>(<detail>)" — keep only the check name
            reasons.update(reason.split("(", 1)[0] for reason in r.get("qc_reasons", []))
        if failed:
            L += _table(["failed check", "utterances", "% of failures"],
                        [[name, n, _pct(n, len(failed))]
                         for name, n in reasons.most_common()])
            L += ["",
                  f"_{len(failed)} utterances failed; a single utterance can trip "
                  f"more than one check, so counts may sum above the failure total._", ""]
            # per-check metric stats
            metrics = defaultdict(list)
            for r in rows:
                for c in r.get("qc_checks", []):
                    if isinstance(c.get("metric"), (int, float)):
                        metrics[c["name"]].append(c["metric"])
            L += ["#### Per-check metric (min / avg / max, all utterances)", ""]
            mrows = []
            for n in check_names:
                s = _num(metrics.get(n, []))
                fails = sum(1 for r in rows for c in r.get("qc_checks", [])
                            if c.get("name") == n and not c.get("passed"))
                if s:
                    mrows.append([n, f"{s[0]:.3f}", f"{s[1]:.3f}", f"{s[2]:.3f}", fails])
            L += _table(["check", "min", "avg", "max", "# failed"], mrows)
            L.append("")
        else:
            L += ["_No QC failures — all checked utterances passed._", ""]
    else:
        L += [f"_Unavailable — `{F_DATASET}` not found._", ""]

    # --- sentence-generation text validation (separate gate) ---
    L += ["### Sentence-generation validation (text gate — separate from audio QC)", ""]
    sent = _load_json(os.path.join(out_dir, F_SENT))
    if sent:
        L += _table(
            ["metric", "value"],
            [["sentences seen", sent.get("total_seen")],
             ["sentences valid", sent.get("total_valid")],
             ["yield", sent.get("yield")]])
        L.append("")
        rej = sent.get("rejection_stats", {})
        if rej:
            total_rej = sum(rej.values())
            L += ["**Rejection breakdown:**", ""]
            L += _table(["reason", "count", "% of rejections"],
                        [[k, v, _pct(v, total_rej)]
                         for k, v in sorted(rej.items(), key=lambda x: -x[1])])
            L.append("")
    else:
        L += [f"_Unavailable — `{F_SENT}` not found._", ""]

    # --- stage-wise pass/fail rates ---
    L += ["### Stage-wise pass / fail", ""]
    acq = _load_json(os.path.join(out_dir, F_ACQUIRE))
    aud = _load_json(os.path.join(out_dir, F_AUDIO))
    tts = _load_json(os.path.join(out_dir, F_TTS))
    srows = []
    if acq:
        p = acq.get("pull", {})
        ok, bad = p.get("downloaded", 0), p.get("failed", 0)
        srows.append(["data_acquisition", "downloaded", ok, bad, _pct(ok, ok + bad)])
    if aud:
        ok, bad = aud.get("prepared", 0), aud.get("failed", 0)
        srows.append(["audio_engineering", "prepared", ok, bad, _pct(ok, ok + bad)])
    if sent:
        ok, seen = sent.get("total_valid", 0), sent.get("total_seen", 0)
        srows.append(["sentence_generation", "valid", ok, seen - ok, _pct(ok, seen)])
    if tts:
        ok, bad = tts.get("synthesized", 0), tts.get("failed", 0)
        srows.append(["tts_generation", "synthesized", ok, bad, _pct(ok, ok + bad)])
    if qc:
        ok, bad = qc.get("passed", 0), qc.get("failed", 0)
        srows.append(["quality_control", "passed", ok, bad, _pct(ok, ok + bad)])
    if srows:
        L += _table(["stage", "kept as", "pass", "fail", "pass rate"], srows)
    else:
        L.append("_No stage summaries found._")
    L.append("")
    return L


# --------------------------------------------------------------------------- #
# section 3 — throughput & cost
# --------------------------------------------------------------------------- #
def _stage_items(name, s):
    """(item label, item count) processed by a stage, for the throughput table."""
    if name == "data_acquisition":
        return "clips", s.get("pull", {}).get("downloaded", 0)
    if name == "audio_engineering":
        return "clips", s.get("prepared", 0)
    if name == "sentence_generation":
        return "sentences", s.get("total_valid", 0)
    if name == "tts_generation":
        return "utterances", s.get("synthesized", 0)
    if name == "quality_control":
        return "utterances", s.get("checked", 0)
    return "items", 0


def section_throughput(out_dir):
    L = ["## 3. Throughput & cost (T4 GPU)", ""]
    stage_files = [
        ("data_acquisition", F_ACQUIRE),
        ("audio_engineering", F_AUDIO),
        ("sentence_generation", F_SENT),
        ("tts_generation", F_TTS),
        ("quality_control", F_QC),
    ]
    rows, total_sec = [], 0.0
    for name, fname in stage_files:
        s = _load_json(os.path.join(out_dir, fname))
        if not s:
            rows.append([name, "—", "—", "—", "_missing_", "—"])
            continue
        sec = float(s.get("elapsed_sec") or 0)
        total_sec += sec
        label, n = _stage_items(name, s)
        # prefer the stage's own utterances_per_min if it reports one
        if s.get("utterances_per_min") is not None:
            rate = f"{s['utterances_per_min']} {label}/min"
        elif sec > 0:
            rate = f"{n / (sec / 60):.1f} {label}/min"
        else:
            rate = "n/a"
        rows.append([name, f"{sec:.1f}", f"{sec / 60:.2f}",
                     f"{sec / 3600:.4f}", f"{n} {label}", rate])
    L += _table(["stage", "elapsed (s)", "min", "T4 GPU-hr", "items", "throughput"], rows)
    L.append("")

    # totals
    dataset = _read_jsonl_safe(os.path.join(out_dir, F_DATASET))
    n_pass = sum(1 for r in dataset if r.get("qc_passed")) if dataset else 0
    L += ["### Totals", ""]
    tot_rows = [
        ["total wall-clock", f"{total_sec:.1f} s  ({total_sec / 60:.1f} min)"],
        ["total compute (T4 GPU-hours)", f"{total_sec / 3600:.4f}"],
    ]
    if n_pass and total_sec > 0:
        tot_rows.append(["end-to-end yield (validated utts / total time)",
                         f"{n_pass / (total_sec / 60):.2f} utts/min"])
    L += _table(["metric", "value"], tot_rows)
    L += ["",
          "_All stages ran on a single Colab T4, so total compute = total wall-clock "
          "GPU-hours. Each stage's `elapsed_sec` is wall-clock and **includes model-weight "
          "loading**, so the throughput figures are end-to-end (matching how the TTS stage "
          "reports its own utterances/min)._", ""]
    return L


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_report(out_dir):
    lines = ["# Synthetic Speech Dataset — Run Report", "",
             f"_out_dir: `{out_dir}`_", ""]
    lines += section_datacard(out_dir)
    lines += section_quality(out_dir)
    lines += section_throughput(out_dir)
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate a post-run pipeline report.")
    ap.add_argument("--config", default="config.yaml",
                    help="config.yaml to resolve out_dir (ignored if --out_dir given)")
    ap.add_argument("--out_dir", default=None,
                    help="folder holding the stage summaries + manifests (overrides config)")
    ap.add_argument("--out", default=None,
                    help="report path (default: <out_dir>/REPORT.md)")
    args = ap.parse_args(argv)

    out_dir = args.out_dir
    if out_dir is None:
        from indic_synth.common.config import load_config
        out_dir = load_config(args.config).out_dir
    out_dir = os.path.abspath(out_dir)

    report = build_report(out_dir)
    out_path = args.out or os.path.join(out_dir, "REPORT.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print(report)
    print(f"\n[report written to {out_path}]")


if __name__ == "__main__":
    main()
