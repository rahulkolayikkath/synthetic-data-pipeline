# Synthetic Speech Data Generation for Indic ASR

> Take real Indian-language speakers (from Kathbath) and use a voice-cloning TTS
> model (IndicF5) to make those voices say _new_ sentences — then rigorously prove
> the synthetic audio is trustworthy enough to train an ASR model on.

_Stub README — scaffold only. See `CLAUDE.md` for the full spec._

## Run (Colab, T4)

```bash
# Cell 1 — get the code
!git clone https://github.com/you/synthetic-data-pipeline.git
%cd synthetic-data-pipeline

# Cell 2 — install dependencies (and the package itself)
!pip install -r requirements.txt
!pip install -e .

# Cell 3 — run
!python scripts/run.py --config config.yaml
```

## Pipeline stages

| Stage | Package | Spec |
|---|---|---|
| Packaging / reproducibility (cross-cutting) | `indic_synth/common` | §4.1 |
| Kathbath speaker & reference acquisition | `indic_synth/data_acquisition` | §4.2 |
| Reference-audio engineering | `indic_synth/audio_engineering` | §4.3 |
| Synthetic sentence generation | `indic_synth/sentence_generation` | §4.4 |
| TTS generation (IndicF5) | `indic_synth/tts_generation` | §4.5 |
| Quality control & validation | `indic_synth/quality_control` | §4.6 |

<!-- TODO: fill in design justifications, metrics, and QC results once implemented. -->
