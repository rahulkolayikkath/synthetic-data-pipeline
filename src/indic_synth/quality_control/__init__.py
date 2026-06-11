"""quality_control — §4.6 quality control & validation.

Implements three independent gates per synthetic utterance:
    1. Content fidelity — ASR transcribe + CER vs intended text; reject CER > 0.15
    2. Speaker fidelity — cosine sim of speaker embeddings (synth vs ref);
       reject < 0.60
    3. Hard failures   — silence, clipping, truncation, looping/repetition, and
       duration-to-char ratio outside [0.04, 0.30] s/char

Modules:
    data     — per-utterance QC result schema + reject thresholds
    models   — indic-conformer-600m ASR + ECAPA speaker-verification wrappers
    pipeline — run the three gates and produce the validated, packaged dataset
"""
