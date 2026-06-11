"""common.manifest — the per-utterance manifest schema (§4.1).

Defines the canonical record written for each synthetic utterance and the IO to
read/write the manifest as JSON Lines. Per-utterance fields (per CLAUDE.md §4.1):
audio path, intended text, speaker id, gender, language, reference clip used,
duration, and the QC metrics from §4.6.

Format choice (to justify here in code): a JSONL manifest mapping cleanly to a
real trainer (e.g. NeMo / ESPnet / HF Datasets) so the packaged mini-dataset can
be consumed without a conversion step.

This schema is the contract shared across stages; each stage extends its own
slice of it (selection / reference / prepared / synthesis / QC manifests).
"""
