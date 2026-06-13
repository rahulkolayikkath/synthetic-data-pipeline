"""tts_generation.config — stage config dataclass (§4.5).

`lang_map` bridges the two naming conventions in the pipeline: sentences generation use
ISO codes (hi, ml) while prepared references audios manifest carry Kathbath folder names
(hindi, malayalam). Pairing groups references by the mapped name so a speaker is
only ever paired with sentences in their own language.
"""
from __future__ import annotations

import dataclasses as dc
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TTSConfig:
    model_id: str = "ai4bharat/IndicF5"
    target_sr: int = 24000
    max_chars: int = 300                 # chunk longer text at sentence boundaries
    device: str = "cuda"
    out_dir: str = "out"
    seed: int = 1234
    log_every: int = 25                  # heartbeat: log progress every N utterances
    # sentence language code -> reference-manifest language name
    lang_map: Dict[str, str] = field(default_factory=lambda: {
        "hi": "hindi", "ml": "malayalam", "ta": "tamil"
    })

    @classmethod
    def from_dict(cls, d: dict) -> "TTSConfig":
        known = {f.name for f in dc.fields(cls)}
        data = {k: v for k, v in (d or {}).items() if k in known and v is not None}
        return cls(**data)
