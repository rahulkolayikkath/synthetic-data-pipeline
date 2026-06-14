"""quality_control.config — stage config dataclass

The QCConfig the tested checks expect: model ids, sample rates, the three spec
thresholds (CER<=0.15, speaker cos>=0.60, dur/char in [0.04,0.30]), and the
hard-failure heuristic knobs
"""
from __future__ import annotations

import dataclasses as dc
from dataclasses import dataclass


@dataclass
class QCConfig:
    # --- model identifiers ---
    asr_model_id: str = "ai4bharat/indic-conformer-600m-multilingual"
    asr_decoding: str = "ctc"                       # "ctc" or "rnnt"
    spk_model_id: str = "speechbrain/spkrec-ecapa-voxceleb"

    # --- sample rates ---
    asr_sr: int = 16_000                            # IndicConformer expects 16 kHz
    spk_sr: int = 16_000                            # ECAPA expects 16 kHz

    # --- thresholds from the spec ---
    cer_max: float = 0.15
    spk_cos_min: float = 0.60
    dur_per_char_min: float = 0.04                  # s/char
    dur_per_char_max: float = 0.30                  # s/char

    # --- hard-failure tuning (heuristics; document any you change) ---
    silence_rms_dbfs: float = -45.0
    silence_active_frac_min: float = 0.10
    clip_sample_thresh: float = 0.99
    clip_frac_max: float = 1e-3
    clip_run_max: int = 8
    trunc_tail_ms: float = 60.0
    trunc_tail_ratio: float = 0.60                   #0.25 was too strict, 
    loop_min_lag_s: float = 0.30
    loop_pair_sim: float = 0.99                      # near-impossible per-pair match
    # looping_audio gate effectively DISABLED: best_frac is a fraction in [0, 1],
    # so a threshold > 1.0 can never be exceeded -> no clip is ever flagged.
    # The MFCC self-similarity heuristic produced ~75% of all QC failures and was
    # judged too fragile (false positives on non-looping audio). Pending a better
    # detector; re-tighten to ~0.85 once the check is improved.
    loop_band_frac: float = 1.01
    text_repeat_ngram: int = 3
    device: str = "cuda"                             # "cpu" or "cuda" if available, Going with cuda as running on gpu

    log_every: int = 25                              # heartbeat: log progress every N utterances
    out_dir: str = "out"

    @classmethod
    def from_dict(cls, d: dict) -> "QCConfig":
        known = {f.name for f in dc.fields(cls)}
        data = {k: v for k, v in (d or {}).items() if k in known and v is not None}
        return cls(**data)
