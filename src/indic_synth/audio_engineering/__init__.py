"""audio_engineering —

Turns each raw Kathbath clip into what IndicF5 expects: 24 kHz, mono, WAV.
IndicF5 (F5-TTS based) writes its output at 24 kHz, so references are normalized
to 24 kHz to match the model's front-end and avoid implicit resampling.

Modules:
    config   — stage config dataclass (AudioConfig.from_dict)
    prepare  — DSP: decode(native sr) -> mono -> resample(24k) -> normalize -> verify
    pipeline — run(cfg, logger): drive prepare over the reference manifest
"""
