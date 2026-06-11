"""audio_engineering.config — stage config dataclass (§4.3).

Holds the attributes process_clip + the driver expect (target_sr, norm, etc.).
`in_manifest` defaults to reference_manifest.jsonl inside out_dir — that's where
§4.2 writes it, and since every stage shares out_dir the default just works.
"""
from __future__ import annotations

import dataclasses as dc
import os
from dataclasses import dataclass
from typing import Optional

from .prepare import TARGET_SR


@dataclass
class AudioConfig:
    out_dir: str = "out"
    in_manifest: Optional[str] = None       # default: <out_dir>/reference_manifest.jsonl
    target_sr: int = TARGET_SR
    norm: str = "peak"                       # peak | rms | lufs
    peak_dbfs: float = -1.0                  # peak ceiling
    rms_dbfs: float = -20.0                  # target for norm rms/lufs
    trim: bool = False
    max_ref_sec: float = 15.0                # hard cap on reference length (0 = off)
    subtype: str = "PCM_16"
    seed: int = 1234

    def __post_init__(self):
        if self.in_manifest is None:
            self.in_manifest = os.path.join(self.out_dir, "reference_manifest.jsonl")

    @classmethod
    def from_dict(cls, d: dict) -> "AudioConfig":
        known = {f.name for f in dc.fields(cls)}
        data = {k: v for k, v in (d or {}).items() if k in known and v is not None}
        return cls(**data)
