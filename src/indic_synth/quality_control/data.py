"""quality_control.data — QC result schema & thresholds (§4.6).

Defines the per-utterance QC record (CER, speaker cosine similarity, hard-failure
flags, per-gate pass/fail, overall verdict) and the reject thresholds:
    CER > 0.15, speaker-sim < 0.60, duration-to-char ratio outside [0.04, 0.30].

These QC metrics are merged back into the per-utterance manifest (common.manifest)
so the final packaged mini-dataset carries its own quality evidence.
"""
