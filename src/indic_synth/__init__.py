"""indic_synth — Synthetic speech data generation pipeline for Indic ASR.

Top-level package. Takes real Kathbath speakers and uses the IndicF5 zero-shot
voice-cloning TTS model to make those voices speak newly generated sentences,
then validates the synthetic audio is trustworthy enough to train an ASR model on.

Stage subpackages (each maps to a CLAUDE.md §4 module):
    common              — §4.1 packaging, reproducibility, observability (cross-cutting)
    data_acquisition    — §4.2 Kathbath two-stage speaker/reference acquisition
    audio_engineering   — §4.3 decode / resample / normalize reference audio
    sentence_generation — §4.4 LLM grid generation of validated Indic sentences
    tts_generation      — §4.5 IndicF5 synthesis of speaker-conditioned utterances
    quality_control     — §4.6 content / speaker / hard-failure validation gates

Target deliverable: 1000 validated utterances across 2 languages, 20 speakers,
gender-balanced, QC'd within a T4 compute budget.
"""
