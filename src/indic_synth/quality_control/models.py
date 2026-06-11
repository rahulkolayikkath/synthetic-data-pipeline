"""quality_control.models — ASR & speaker-verification wrappers (§4.6).

Wraps the two validation models:
    - ai4bharat/indic-conformer-600m-multilingual — ASR used to transcribe each
      synthetic utterance for the content-fidelity (CER) gate.
    - speechbrain/spkrec-ecapa-voxceleb — speaker-verification model used to
      embed synth and reference audio for the speaker-fidelity (cosine-sim) gate.
"""
