# Synthetic Speech Data Generation for Indic ASR
The one-line summary for what the code does: 
Build a pipeline that takes real Indian-language speakers (from Kathbath) and uses a voice-cloning TTS model (IndicF5) to make those voices say _new_ sentences — then rigorously prove the synthetic audio you generated is trustworthy enough to train an ASR model on.
## 1. Context

We build ASR systems for Indian languages. Real labelled speech is scarce, expensive, and demographically skewed. One lever is synthetic speech augmentation: take a multi-speaker zero-shot TTS model, condition it on real speakers, and have it speak new, controlled text. Done well, this expands lexical and acoustic coverage cheaply. Done badly, it injects hallucinations, distribution shift, and silent label noise that degrades the very models it was meant to help.

Your task is to build a small but production-minded synthetic-speech generation pipeline and, just as importantly, to prove the output is trustworthy. More than it runs the system should focus far more about the judgment and the validation.

## 2. The components 

- IndicF5 (ai4bharat/IndicF5 on Hugging Face) — a zero-shot, voice-cloning TTS model. 
- Kathbath (ai4bharat/Kathbath, part of IndicSUPERB) — ~1,684 hours of read speech across 12 languages. (ASR Dataset)
- ai4bharat/indic-conformer-600m-multilingual - ASR model, This model can be used to transcribe speech in various Indian languages.
- speechbrain/spkrec-ecapa-voxceleb - speaker Verification model
- Gema 3 12 b - 4 bit Quantized - Multilingual LLM, This model can be used to Generate text in Indic Languages
---

## 3. What you must build

A reproducible, fault-tolerant pipeline that produces a packaged, validated mini-dataset of synthetic utterances — each one IndicF5 speaking a newly generated sentence in the voice of a real Kathbath speaker — implementing the §4 spec end-to-end.

1000 validated utterances spanning 2 languages and 20 distinct speakers, with balanced gender representation, produced and QC'd within your compute budget.

The pipeline must produce the quality, and demonstrate it with metrics.
  
---

## 4. Required Modules 
### 4.1  General - Output packaging and Reproducibility
Make sure all the code we write inside creates an output that is Packaged & reproducible
- Design a manifest schema (per-utterance: audio path, intended text, speaker id, gender, language, reference clip used, duration, and your QC metrics). Pick a format compatible with a real trainer (e.g., NeMo / ESPnet / HF Datasets) and justify.
- Pin the environment, fix seeds, and make the run deterministic enough to reproduce your headline numbers.
- Throughput: batching/precision choices; report utterances/min and total compute used; estimate total wall-clock on a T4 before you run.
- Fault-tolerance: checkpointing and resumability so a dropped Colab session doesn't cost the run; idempotent restart; partial-failure handling and retries; structured logging/observability so you can see progress and diagnose failures.

### 4.2 Data acquisition & subsetting

Kathbath two-stage speaker acquisition
Stage 1 of the synthetic-speech pipeline: pull a small, **diverse, gender-balanced**
bank of real speakers and reference clips from
[`ai4bharat/Kathbath`](https://huggingface.co/datasets/ai4bharat/Kathbath) — without downloading the ~170 GB dataset.
#### The idea
Kathbath is used here purely as a *voice bank* (reference audio + reference text to
condition a zero-shot TTS model), not as training/eval data. That reframing drives
every choice:
- **Split = `valid`.** We don't train or evaluate an ASR model on Kathbath, so the
train/known/unknown-speaker distinction is irrelevant. `valid` is ~30× smaller than
`train` (~0.5 GB vs ~16 GB per language) and still holds dozens of gender-labelled
speakers. (The HF parquet build only exposes `train` and `valid`; the known/unknown test splits live in the raw IndicSUPERB distribution.)
- **Catalog cheaply, pull selectively.** Two stages with a hard cost boundary:
```

valid-*.parquet (gated)

│ column projection — audio column NEVER fetched

▼

Stage A: metadata catalog ── speaker_id, gender, lang, duration, text (KBs)

▼

seeded stratified sampler ── balance language / speaker / gender

▼

selection_manifest.jsonl ── reproducible list of exact rows

┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ cost boundary

▼ read only the row groups containing chosen rows

Stage B: pull audio ── ref_audio/*.wav + reference_manifest.jsonl

```

The Kathbath per-language schema is
`fname, text, audio_filepath (audio), lang, duration, gender, speaker_id`.
Because `speaker_id` and `gender` are real columns, Stage A can profile every speaker by reading only those columns — parquet stores each column separately, so the heavy `audio_filepath` chunks are never transferred.
#### Diversity, honestly
- **Speakers/gender**: the sampler groups by `(lang, speaker_id)`, then picks a
balanced male/female quota per language from the *whole* pool — not the first N
rows. Clips per speaker are filtered to a sane reference duration window.
- **Regional**: the HF parquet has **no district/state column**, so regional balance
cannot be guaranteed from this metadata. We maximize distinct `speaker_id`s as a
proxy (1,218 contributors span 203 districts). Recovering district means going to
the raw IndicSUPERB filenames, which forces an m4a download — out of scope here.
#### Outputs (`out_dir/`)

| File | What 
|---|---|

| `catalog.parquet` | every utterance's metadata + `source_file`, `row_index` (cached) |
| `selection_manifest.jsonl` | the chosen rows (deterministic given catalog + seed) |
| `reference_manifest.jsonl` | per clip: `local_audio_path`, `ref_text`, speaker, gender, duration, `status` |
| `ref_audio/<ref_id>.<ext>` | the downloaded reference clips |
| `run_summary.json` | counts, per-language gender split, row groups touched, elapsed |
`reference_manifest.jsonl` is the input to Stage 4.3 (decode/resample/normalize).

Audio bytes are written **verbatim** with the source extension — no decode happens
here, so resampling decisions stay in the stage that owns them.

#### Reproducibility, fault-tolerance, observability
- **Deterministic**: catalog is sorted into a stable order; the sampler is fully
seeded. Same inputs + seed ⇒ same speakers and clips.
- **Idempotent / resumable**: the catalog is cached; the puller appends to
`reference_manifest.jsonl` and skips any `ref_id` already downloaded and present on
disk, so a dropped session resumes without re-sampling.
- **Partial-failure isolation**: a bad row or unreadable row group is logged, marked
`failed` in the manifest, and the run continues. HF/IO reads use exponential-backoff
retries.
- **Observability**: structured logs per stage + a `run_summary.json` reporting
selection counts, gender balance, and how many row groups / approx bytes Stage B touched.

### 4.3 Reference-audio engineering 
Kathbath audio will not be in the format or sample rate IndicF5 expects. So we have  decode/resample/normalization and choose the target sample rate, and avoid silent resampling bugs
#### idea: 
Turns each raw Kathbath clip into what IndicF5 expects. IndicF5 (F5-TTS-based) takes
a target text, a reference **audio path**, and the reference transcript, and its output is written at **24 kHz**. So references are normalized to **24 kHz, mono, WAV**.
#### Output:
`prepared_audio/<ref_id>.wav` + `prepared_manifest.jsonl` (extends the input
manifest with `prepared_audio_path`, `source_sr`, `target_sr`, `resample_method`,
`channels_in`, `gain_db`, `out_duration`, `out_peak_dbfs`, `out_rms_dbfs`, `qc_flags`)
+ `prepare_summary.json`. This manifest is the input to Stage 4.5 (TTS generation).
#### Decisions
- **Target SR 24000 Hz** matches F5-TTS's front-end, so the model's internal
preprocessing does no implicit resample. Kathbath is ~16 kHz, so we usually
**upsample** - flagged `resampled_up` because it cannot create real content above
the source Nyquist (an honest limitation, not a defect).
- **Mono**, downmixing by channel-average if a clip is ever stereo (`channels_in`
recorded).
- **Normalization** defaults to **peak to -1 dBFS**: deterministic and cannot clip
by construction. `rms` (target dBFS) and `lufs` (ITU-R BS.1770, needs pyloudnorm)
are available, both with a peak guard.

How silent resampling bugs are avoided
1. Decode at the file's **native** rate - the loader is never asked to resample.
2. Read the real header rate; refuse if it's missing/zero.
3. Resample **explicitly** with soxr (HQ) and assert the output length matches the
rate ratio.
4. Assert post-conditions: `out_sr == target`, mono, finite, peak <= 1.0.
5. Read the written file back and check its header rate (catches a silent writer misconfig).
6. Record source/target rate, backend, gain and QC flags per clip, so a wrong rate
or near-silent clip surfaces in the manifest instead of poisoning the TTS run.
Near-silence is measured **before** normalization, since peak/RMS gain would
otherwise amplify a silent clip to full scale and mask it.

### 4.4 Synthetic sentence generation
Generates validated, diverse Indic sentences that will later be spoken by IndicF5.
**Design**
- Loop over a **grid of `(language × topic × sentence_type)` cells**; keep generating per cell until each hits its valid quota. This *guarantees* topic/type/language balance instead of hoping for it.
- Small batches per LLM call (≈12) to avoid long-generation degeneracy.
- A **programmatic validation gate** is the source of truth, not the prompt: script check → language-ID → normalization → degeneracy → dedup. Every rejection is logged with a reason (this becomes the QC-yield table in the report).
- **Per-call seeds** derived from a base seed give reproducibility *and* diversity. The validated pool is **checkpointed to disk** so a dropped Colab session resumes without re-doing work.
**Model:** `google/gemma-3-12b-it` (4-bit) — instruction-tuned, strong Indic script coverage, fits a T4. Swap to Sarvam-M (24B) on an L4/A100, or Gemma 3 4B-it for more headroom. 

### 4.5 TTS generation

Model: ASR model Indic F5 
- Using the model card's default inference parameters. Chunk any input over the length limit at sentence boundaries.
- A reference speaker is paired only with sentences in that speaker's own language (no cross-lingual conditioning in the main run).
- Output format, sample rate, and loudness normalization for the final files.

### 4.6 Quality control & validation 

`Implements three independent gates per utterance:`
`1. Content fidelity -> ASR transcribe + CER vs intended text; reject CER > 0.15`
`2. Speaker fidelity -> cosine sim of speaker embeddings (synth vs ref); reject < 0.60`
`3. Hard failures -> silence, clipping, truncation, looping/repetition,duration-to-char ratio outside [0.04, 0.30] s/char`


## Design of REPO
The design of the repo should be such that it should be runnable in Colab notebook Connected to a T4 GPU. 

**Repo structure** — organize as an installable package so imports resolve cleanly:

```
synthetic-data-pipeline/
├── src/
│   └── data-acquisition/ # module 1, similarly module 2 Audio eng
│       ├── __init__.py
│       ├── data.py
│       ├── models.py
│       └── pipeline.py 
├── scripts/
│   └── run.py
├── requirements.txt
├── pyproject.toml        # makes it pip-installable
├── config.yaml
└── README.md
```

**The Colab notebook** is then just a few cells:
python

```python
# Cell 1 — get the code
!git clone https://github.com/you/my-pipeline.git
%cd my-pipeline
```
python

```python
# Cell 2 — install dependencies (and the package itself)
!pip install -r requirements.txt
!pip install -e .          # -e makes `import mypipeline` work anywhere
```
python

```python
# Cell 3 — run
!python scripts/run.py --config config.yaml
```


!!! General Rules is Use the simplest possible approach, don't over engineer 
Most of the code will be copy pasted, your job is to Stitch together the pieces one-by-one!!! 