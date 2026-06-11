"""audio_engineering — §4.3 reference-audio engineering.

Turns each raw Kathbath clip into what IndicF5 expects: 24 kHz, mono, WAV.
IndicF5 (F5-TTS based) writes its output at 24 kHz, so references are normalized
to 24 kHz to match the model's front-end and avoid implicit resampling.

Modules:
    data     — prepared_manifest schema + QC flag fields
    models   — explicit resampler (soxr HQ) + loudness backends (peak/rms/lufs)
    pipeline — decode(native sr) -> resample(24k mono) -> normalize -> verify
"""
