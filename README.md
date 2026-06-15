# Synthetic Speech Data Generation for Indic ASR

ASR for Indian languages is bottlenecked by data: real labelled speech is scarce, expensive, and
demographically skewed. Synthetic augmentation is one lever — take a zero-shot voice- cloning TTS
model, condition it on real speakers, and have it speak new, controlled text. 

> Take real Indian-language speakers (from Kathbath) and use a voice-cloning TTS
> model (IndicF5) to make those voices say _new_ sentences — then rigorously prove
> the synthetic audio is trustworthy enough to train an ASR model on.

A reproducible, fault-tolerant pipeline that produces a packaged, validated
mini-dataset of synthetic utterances.

## Run (Colab, T4)

The recommended path is the **stage-by-stage runbook**: `notebooks/Colab_stage1_2_3_5.ipynb` and `notebooks/Colab_stage4.ipynb`
It validates one stage at a time on the GPU, persists outputs to Drive (so stages can
run in separate runtime sessions), and works around the transformers version conflict
by using differnt runtimes in sepearte notebooks. It uses `config.colab.yaml` (outputs to Drive) and reads the
HF token via `getpass` .

To run the whole pipeline in stage by satge:

```python
!git clone https://github.com/you/synthetic-data-pipeline.git
%cd synthetic-data-pipeline
!pip install -r requirements.txt
!pip install -e .
!python scripts/run.py --config config.colab.yaml --stages data_acquisition            # or --stages <name> for a subset
```

A gated `HF_TOKEN` is needed for Kathbath and Gemma-3 (`huggingface_hub` login or
`export HF_TOKEN=...`), plus accepting the Kathbath dataset terms and Gemma-3 license.
Inspect any manifest with `python scripts/inspect_manifest.py <manifest.jsonl>`.

## Installable CLI & config validation

`pip install -e .` registers an `indic-synth` console command (entry point in
`pyproject.toml`), equivalent to `python scripts/run.py`:

```bash
indic-synth --config config.yaml                              # all stages, in order
indic-synth --config config.yaml --stages tts_generation quality_control
indic-synth --config config.yaml --validate-only             # check config, run nothing
indic-synth --config config.yaml --no-validate               # skip validation (not advised)
```

Before any stage runs, the config is checked against a declarative **schema**
(`CONFIG_SCHEMA` in `indic_synth/common/config.py`, validated by `validate_config`):

- **Types / ranges / choices** — e.g. `cer_max` ∈ [0, 1], `audio_engineering.norm` ∈
  {`peak`,`rms`,`lufs`}, `speakers_per_language` a positive int.
- **Cross-field rules** — `ref_min_dur < ref_max_dur`, `dur_per_char_min <
  dur_per_char_max`, `min_words ≤ max_words`.
- **Unknown keys** are reported as warnings (likely typos) — never silently dropped.

An invalid config fails fast with every problem listed and a non-zero exit code, e.g.:

```
invalid config:
  - audio_engineering.norm: 'pek' not one of ['peak', 'rms', 'lufs']
  - quality_control.cer_max: 5 is above maximum 1
  - quality_control.dur_per_char_min (0.5) must be < quality_control.dur_per_char_max (0.3)
```

## Pipeline stages & manifest chain
The system is a five-stage pipeline over a shared common/ layer (logging, config, manifest IO, checkpointing). Each stage is a Python package exposing a single run(cfg, logger) entry point, so any stage runs on its own and is independently resumable.
1) Data Acquisition — treats Kathbath as a voice bank, not training data. Builds a cheap metadata catalog of speakers, then selectively downloads a small, gender-balanced set of reference clips.
2) Audio Engineering — turns each raw reference clip into exactly what IndicF5 expects: 24 kHz mono peak-normalized WAV. Resampling and loudness are made explicit and verified so no silent format bug leaks downstream.
3) Sentence Generation — Gemma-3-12B (4-bit) walks a (language × topic × sentence_type) grid so coverage is balanced by construction, and runs every candidate through a programmatic validation gate (script, language-ID, normalization, degeneracy, dedup).
4) TTS Generation — IndicF5 is conditioned on a real reference clip + transcript and made to speak a generated sentence in that speaker's own language. Long text is chunked at sentence boundaries and re-joined.
5) Quality Control — Every clip passes three independent gates — content fidelity (ASR + CER), speaker fidelity (embedding cosine), and hard-failure DSP checks — before it enters the final dataset.


Each stage exposes `run(cfg, logger)` and is independently resumable; `scripts/run.py`
runs them in order, threading manifests through a shared `out_dir`:

| Stage | Package | Output manifest |
|---|---|---|
| Packaging / reproducibility (cross-cutting) | `indic_synth/common` | — |
| Kathbath speaker & reference acquisition | `indic_synth/data_acquisition` | `reference_manifest.jsonl` |
| Reference-audio engineering (→24 kHz mono) | `indic_synth/audio_engineering` | `prepared_manifest.jsonl` |
| Synthetic sentence generation (Gemma-3) | `indic_synth/sentence_generation` | `sentences.jsonl` |
| TTS generation (IndicF5, same-language pairing) | `indic_synth/tts_generation` | `tts_manifest.jsonl` |
| Quality control (3 gates) | `indic_synth/quality_control` | `dataset_manifest.jsonl` (final) |

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

## Ethics and Licensing
Kathbath is gated on Hugging Face (you must accept conditions and share contact information to
access it). Its HF metadata tag is cc-by-4.0 , while the dataset card itself states that AI4Bharat
licenses the packaging and annotations under CC0 ("no rights reserved") and notes the raw text
originates from IndicCorp, a crawl of publicly available websites. I used Kathbath purely as a voice
bank — reference audio + transcript to condition the TTS — and do not redistribute its raw audio.

IndicF5 is released under the MIT license (also gated on access) and ships an explicit voice- cloning
term: by using the model you agree to only clone voices for which you have explicit permission;
unauthorized voice cloning is strictly prohibited; any misuse is the user's responsibility. It is built on
F5-TTS and trained on AI4Bharat's IndicVoices-R and Rasa.

### The intended-use boundary of what I produced
The output is intended for ASR training-data augmentation and research only. The sharp ethical
tension worth naming: Kathbath speakers consented to having their speech collected for an ASR
dataset, which is not the same as consenting to have their voice cloned. IndicF5's terms put the
permission burden on the user. So within a research/benchmark context this is defensible, but these
clips should not be used to impersonate identifiable individuals, are paired same-language only (no
cross-lingual voice transfer), carry anonymized speaker IDs, and every clip is labelled synthetic in the
manifest so it can never be mistaken for a real recording. Producing them is augmentation; using
them to make a named person appear to say something would cross the line the IndicF5 terms draw.

## Known integration risk

IndicF5's model needs to pins transfomers version to `transformers==4.49.0`, while Gemma-3 needs
`transformers>=4.50`. They can conflict in one environment. `requirements.txt`
defaults to `>=4.50`; if IndicF5 fails to load, run stage 4 in a separate
runtime.

## References of Models used
1. ai4bharat/Kathbath — voice bank. huggingface.co/datasets/ai4bharat/Kathbath
(IndicSUPERB, arXiv:2208.11761)
2. ai4bharat/IndicF5 — zero-shot voice-cloning TTS. huggingface.co/ai4bharat/IndicF5 ·
github.com/ AI4Bharat/IndicF5
3. ai4bharat/indic-conformer-600m-multilingual — ASR for the CER gate
4. google/gemma-3-12b-it — sentence generation LLM
5. sentence-transformers/LaBSE — semantic dedup embeddings
6. speechbrain/spkrec-ecapa-voxceleb — speaker-verification embeddings
7. fastText lid.176 — language identification · F5-TTS (upstream of IndicF5)