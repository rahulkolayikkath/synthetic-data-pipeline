"""quality_control.pipeline — runs the three validation gates (§4.6).

For each synthetic utterance, runs three independent gates and records the result:
    1. Content fidelity — transcribe with indic-conformer, compute CER vs the
       intended text, reject CER > 0.15.
    2. Speaker fidelity — embed synth + reference with ECAPA, compute cosine
       similarity, reject < 0.60.
    3. Hard failures — detect silence, clipping, truncation, looping/repetition,
       and duration-to-char ratio outside [0.04, 0.30] s/char.

Merges QC metrics into the per-utterance manifest and packages the validated
mini-dataset. Emits the headline QC metrics that prove the synthetic audio is
trustworthy to train on. Resumable / idempotent per utterance.
"""
