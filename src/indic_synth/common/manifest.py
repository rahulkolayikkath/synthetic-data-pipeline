"""common.manifest — JSON-Lines manifest IO + the per-utterance schema (§4.1).

Every stage reads its input and writes its output as JSON Lines (one JSON object
per line). JSONL is chosen because it maps directly onto real trainers — NeMo and
ESPnet both consume a `.jsonl`/manifest of {audio_filepath, text, duration, ...}
records, and HF `datasets.load_dataset("json", ...)` reads it as-is — so the
packaged mini-dataset needs no conversion step. It also streams and appends
cheaply, which is what makes the stages resumable (append one validated line at a
time, never rewrite the whole file).

`UTTERANCE_FIELDS` documents the final per-utterance record (CLAUDE.md §4.1): the
TTS stage writes the generation fields, the QC stage adds its metrics, and the
final `dataset_manifest.jsonl` carries the full set.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, Iterator, List

# Canonical final per-utterance schema (§4.1). Stages populate it progressively:
#   TTS (§4.5)  -> the identity + generation fields
#   QC  (§4.6)  -> the qc_* metrics + verdict
UTTERANCE_FIELDS = [
    # identity / generation (written by §4.5 tts_generation)
    "utt_id",            # unique id for the synthetic utterance
    "audio_filepath",    # path to the synth wav (24 kHz), relative to out_dir
    "text",              # intended text the TTS was asked to speak
    "speaker_id",        # real Kathbath speaker whose voice was cloned
    "gender",            # speaker gender (male/female/unknown)
    "language",          # language code (e.g. hi, ml)
    "ref_id",            # reference clip used to condition IndicF5
    "ref_audio_path",    # prepared 24 kHz reference clip path
    "duration",          # synth audio duration in seconds
    # quality control (written by §4.6 quality_control)
    "qc_passed",         # bool: passed all three gates
    "qc_reasons",        # list[str]: why it failed (empty if passed)
    "cer",               # content fidelity: CER of ASR transcript vs `text`
    "asr_transcript",    # the ASR hypothesis used for CER
    "speaker_sim",       # speaker fidelity: cosine sim (synth vs ref embedding)
]


def read_jsonl(path: str) -> List[Dict]:
    """Read a JSONL file into a list of dicts. Missing file -> empty list."""
    if not os.path.exists(path):
        return []
    rows: List[Dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: str) -> Iterator[Dict]:
    """Stream a JSONL file row-by-row (for large manifests)."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _json_default(o):
    """Coerce numpy scalars/arrays (which DSP checks emit) into JSON-native types,
    without importing numpy — duck-typed so manifests never crash on a np.float32."""
    if hasattr(o, "item"):       # numpy scalar -> Python int/float
        return o.item()
    if hasattr(o, "tolist"):     # numpy array -> list
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    """Overwrite `path` with one JSON object per line."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")


def append_jsonl(path: str, rec: Dict) -> None:
    """Append a single record and flush — the unit of resumable progress."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")
        f.flush()
