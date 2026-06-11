"""tts_generation — §4.5 TTS generation with IndicF5.

Synthesizes each utterance: IndicF5 (ai4bharat/IndicF5) speaking a generated
sentence in the voice of a real Kathbath speaker, conditioned on that speaker's
prepared reference clip + reference transcript. A reference speaker is paired
ONLY with sentences in that speaker's own language (no cross-lingual conditioning
in the main run).

Modules:
    data     — synthesis job + output utterance schema
    models   — IndicF5 wrapper (target text + reference audio path + ref text)
    pipeline — pair speaker<->sentence, chunk long inputs, synth, write 24 kHz
"""
