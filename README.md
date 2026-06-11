# Synthetic Speech Data Generation for Indic ASR

> Take real Indian-language speakers (from Kathbath) and use a voice-cloning TTS
> model (IndicF5) to make those voices say _new_ sentences — then rigorously prove
> the synthetic audio is trustworthy enough to train an ASR model on.

A reproducible, fault-tolerant pipeline that produces a packaged, validated
mini-dataset of synthetic utterances. See `CLAUDE.md` for the full spec.

## Run (Colab, T4)

The recommended path is the **stage-by-stage runbook**: `notebooks/colab_runbook.ipynb`.
It validates one stage at a time on the GPU, persists outputs to Drive (so stages can
run in separate runtime sessions), and works around the transformers version conflict
by installing per session. It uses `config.colab.yaml` (outputs to Drive) and reads the
HF token via `getpass` (never committed).

To run the whole pipeline in one go instead:

```python
!git clone https://github.com/you/synthetic-data-pipeline.git
%cd synthetic-data-pipeline
!pip install -r requirements.txt
!pip install git+https://github.com/ai4bharat/IndicF5.git   # not on PyPI
!pip install -e .
!python scripts/run.py --config config.yaml            # or --stages <name> for a subset
```

A gated `HF_TOKEN` is needed for Kathbath and Gemma-3 (`huggingface_hub` login or
`export HF_TOKEN=...`), plus accepting the Kathbath dataset terms and Gemma-3 license.
Inspect any manifest with `python scripts/inspect_manifest.py <manifest.jsonl>`.

## Pipeline stages & manifest chain

Each stage exposes `run(cfg, logger)` and is independently resumable; `scripts/run.py`
runs them in order, threading manifests through a shared `out_dir`:

| Stage | Package | Output manifest |
|---|---|---|
| Packaging / reproducibility (cross-cutting) | `indic_synth/common` | — |
| §4.2 Kathbath speaker & reference acquisition | `indic_synth/data_acquisition` | `reference_manifest.jsonl` |
| §4.3 Reference-audio engineering (→24 kHz mono) | `indic_synth/audio_engineering` | `prepared_manifest.jsonl` |
| §4.4 Synthetic sentence generation (Gemma-3) | `indic_synth/sentence_generation` | `sentences.jsonl` |
| §4.5 TTS generation (IndicF5, same-language pairing) | `indic_synth/tts_generation` | `tts_manifest.jsonl` |
| §4.6 Quality control (3 gates) | `indic_synth/quality_control` | `dataset_manifest.jsonl` (final) |

```
acquisition ─> reference_manifest ─┐
audio eng   ─> prepared_manifest ──┤
                                   ├─> TTS (pairs by language) ─> tts_manifest
sentences   ─> sentences ──────────┘                                  │
                                   QC reads tts_manifest ─> dataset_manifest (+ QC metrics)
```

The final `dataset_manifest.jsonl` is JSON-Lines (NeMo / ESPnet / HF-Datasets
compatible), one row per utterance: audio path, intended text, speaker id, gender,
language, reference clip, duration, and the QC metrics (`qc_passed`, `cer`,
`speaker_sim`, per-gate results).

## Smoke tests (no GPU required)

Each stage has a CPU smoke test under `scripts/`; GPU stages (Gemma / IndicF5 /
conformer / ECAPA) are driven with injected fakes so the wiring, manifests, and
pairing/validation logic can be verified locally. The real models run in Colab.

```bash
python scripts/smoke_common.py     # §4.1 manifest IO, config, determinism, checkpoint
python scripts/smoke_acquire.py    # §4.2 catalog -> sample -> pull (local fake_kathbath)
python scripts/smoke_audio.py      # §4.3 24 kHz mono resample + normalize
python scripts/smoke_sentences.py  # §4.4 validation gates + grid/quota/checkpoint loop
python scripts/smoke_tts.py        # §4.5 same-language pairing + chunking + resume
python scripts/smoke_qc.py         # §4.6 DSP/CER gates + orchestration
python scripts/smoke_e2e.py        # full §4.2 -> §4.6 chain through one out_dir
```

## Known integration risk

IndicF5's model card pins `transformers==4.49.0`, while Gemma-3 (§4.4) needs
`transformers>=4.50`. They can conflict in one environment. `requirements.txt`
defaults to `>=4.50`; if IndicF5 fails to load, run §4.4 (sentences) in a separate
session from §4.5/§4.6 — the manifests make that safe (`--stages` selects a subset).
