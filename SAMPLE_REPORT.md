# Synthetic Speech Dataset — Run Report

_out_dir: `/content/drive/MyDrive/indic_synth/out_representative_sample`_

## 1. Datacard — dataset statistics

| metric | value |
|---|---|
| utterances checked | 96 |
| validated (qc_passed=true) | 80  (83.3%) |
| distinct languages | 3 |
| distinct (lang, speaker) | 6 |
| validated audio | 353.1 s  (0.098 h) |

### Manifest fields (`dataset_manifest.jsonl`)

- **`utt_id`** — unique utterance id (e.g. hi_000000)
- **`audio_filepath`** — path to the synthesized wav, relative to out_dir
- **`text`** — intended sentence the TTS model was asked to speak
- **`speaker_id`** — Kathbath speaker whose voice was cloned
- **`gender`** — speaker gender (male / female)
- **`language`** — language code (hi / ml / ta)
- **`ref_id`** — reference clip used for voice conditioning
- **`ref_audio_path`** — prepared 24kHz mono reference wav, relative to out_dir
- **`duration`** — synthesized audio duration in seconds
- **`topic`** — sentence-generation topic cell
- **`sentence_type`** — declarative / interrogative / imperative / exclamatory
- **`status`** — pipeline status (validated)
- **`qc_passed`** — True if the utterance passed all QC gates
- **`qc_reasons`** — list of failed check names ([] when passed)
- **`asr_transcript`** — what the ASR model heard (for the CER gate)
- **`cer`** — character error rate vs intended text (content fidelity)
- **`speaker_sim`** — cosine similarity of speaker embeddings synth-vs-ref
- **`qc_checks`** — per-check detail: {name, passed, metric, detail}

### Per-language

| language | validated/total utts | speakers | M/F speakers |
|---|---|---|---|
| hi | 32/32 | 2 | 1/1 |
| ml | 31/32 | 2 | 1/1 |
| ta | 17/32 | 2 | 1/1 |

### Per-speaker

| language | speaker_id | gender | validated/total utts |
|---|---|---|---|
| hi | 229 | female | 16/16 |
| hi | 934 | male | 16/16 |
| ml | 187 | female | 15/16 |
| ml | 1145 | male | 16/16 |
| ta | 21 | male | 15/16 |
| ta | 67 | female | 2/16 |

### Gender balance

| gender | utterances | speakers |
|---|---|---|
| female | 48 | 3 |
| male | 48 | 3 |

### Distributions (min / avg / max) — over validated set

| metric | min | avg | max |
|---|---|---|---|
| duration (s) | 2.02 | 4.41 | 8.83 |
| sentence length (chars) | 31 | 53.4 | 85 |
| sentence length (words) | 5 | 8.2 | 14 |
| cer | 0.000 | 0.022 | 0.120 |
| speaker_sim | 0.648 | 0.793 | 0.890 |

_Distributions are over the 80 validated (qc_passed=true) utterances — the deliverable. Counts tables above show validated/total so failures stay visible. (Full-set duration min/avg/max: 2.02/4.16/8.83 s.)_

## 2. Quality report

### QC yield (audio gates)

| metric | value |
|---|---|
| checked | 96 |
| passed | 80 |
| failed | 16 |
| pass rate | 0.8333 |
| cer_max | 0.15 |
| spk_cos_min | 0.6 |
| dur_per_char | [0.04, 0.3] |

### QC checks run (3 gates + hard-failure checks)

- **`silence`** — rejects near-silent / mostly-inactive audio
- **`clipping`** — rejects audio with excessive clipped samples
- **`truncation`** — rejects clips that look cut off (loud tail energy)
- **`dur_per_char`** — duration-to-char ratio must fall in [0.04, 0.30] s/char
- **`looping_audio`** — rejects repeated/looping audio segments
- **`cer`** — content fidelity — ASR CER vs intended text (gate < 0.15)
- **`looping_text`** — rejects repeated tokens in the ASR transcript
- **`speaker_cos`** — speaker fidelity — cosine sim synth-vs-ref (gate >= 0.60)

### QC rejection breakdown (failed utterances only)

| failed check | utterances | % of failures |
|---|---|---|
| truncation | 14 | 87.5% |
| speaker_cos | 1 | 6.2% |
| cer | 1 | 6.2% |
| looping_audio | 1 | 6.2% |

_16 utterances failed; a single utterance can trip more than one check, so counts may sum above the failure total._

#### Per-check metric (min / avg / max, all utterances)

| check | min | avg | max | # failed |
|---|---|---|---|---|
| silence | -20.000 | -20.000 | -19.990 | 0 |
| clipping | 0.000 | 0.000 | 0.000 | 0 |
| truncation | 0.000 | 0.120 | 0.802 | 14 |
| dur_per_char | 0.046 | 0.078 | 0.117 | 0 |
| looping_audio | 0.171 | 0.435 | 0.933 | 1 |
| cer | 0.000 | 0.024 | 0.160 | 1 |
| looping_text | 1.000 | 1.021 | 2.000 | 0 |
| speaker_cos | 0.573 | 0.794 | 0.890 | 1 |

### Sentence-generation validation (text gate — separate from audio QC)

| metric | value |
|---|---|
| sentences seen | 109 |
| sentences valid | 96 |
| yield | 0.8807 |

**Rejection breakdown:**

| reason | count | % of rejections |
|---|---|---|
| near_duplicate | 6 | 46.2% |
| wrong_script | 4 | 30.8% |
| too_short | 3 | 23.1% |

### Stage-wise pass / fail

| stage | kept as | pass | fail | pass rate |
|---|---|---|---|---|
| data_acquisition | downloaded | 12 | 0 | 100.0% |
| audio_engineering | prepared | 12 | 0 | 100.0% |
| sentence_generation | valid | 96 | 13 | 88.1% |
| tts_generation | synthesized | 96 | 0 | 100.0% |
| quality_control | passed | 80 | 16 | 83.3% |

## 3. Throughput & cost (T4 GPU)

| stage | elapsed (s) | min | T4 GPU-hr | items | throughput |
|---|---|---|---|---|---|
| data_acquisition | 134.5 | 2.24 | 0.0374 | 12 clips | 5.4 clips/min |
| audio_engineering | 0.4 | 0.01 | 0.0001 | 12 clips | 2000.0 clips/min |
| sentence_generation | 359.8 | 6.00 | 0.0999 | 96 sentences | 16.0 sentences/min |
| tts_generation | 1192.3 | 19.87 | 0.3312 | 96 utterances | 4.8 utterances/min |
| quality_control | 186.0 | 3.10 | 0.0517 | 96 utterances | 31.0 utterances/min |

### Totals

| metric | value |
|---|---|
| total wall-clock | 1873.0 s  (31.2 min) |
| total compute (T4 GPU-hours) | 0.5203 |
| end-to-end yield (validated utts / total time) | 2.56 utts/min |

_All stages ran on a single Colab T4, so total compute = total wall-clock GPU-hours. Each stage's `elapsed_sec` is wall-clock and **includes model-weight loading**, so the throughput figures are end-to-end (matching how the TTS stage reports its own utterances/min)._

