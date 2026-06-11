"""tts_generation.data — synthesis job & output utterance schemas (§4.5).

Defines:
    - the synthesis job (intended text + paired speaker/reference clip + language)
    - the output utterance record (audio path, intended text, speaker id, gender,
      language, reference clip used, duration), written 24 kHz with the chosen
      output format and loudness normalization.

This output manifest is the input to Stage 4.6 (quality_control).
"""
