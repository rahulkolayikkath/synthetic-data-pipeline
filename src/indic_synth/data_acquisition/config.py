"""data_acquisition.config — stage config dataclass

The tested acquisition code (catalog/sampler/puller) expects a `cfg` object with
these attributes. We keep that dataclass intact and add
`from_dict`, so the orchestrator can build it from the unified config subsection
(`cfg.stage("data_acquisition")`) without changing any stage logic.
"""
from __future__ import annotations

import dataclasses as dc
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AcquireConfig:
    source: str = "hf" # Source: "hf" reads from huggingface.co; "local"
    repo_id: str = "ai4bharat/Kathbath"
    local_dir: str = "fake_kathbath" # reads a directory laid out like <local_dir>/<lang>/<split>-*.parquet.

    languages: List[str] = field(default_factory=lambda: ["hindi", "tamil"])
    split: str = "valid"

    speakers_per_language: int = 10
    clips_per_speaker: int = 4
    min_total_speakers: int = 15
    gender_balance: bool = True

    ref_min_dur: float = 3.0
    ref_max_dur: float = 15.0

    seed: int = 1234
    out_dir: str = "out"

    hf_token: Optional[str] = None      # falls back to $HF_TOKEN / cached login
    max_retries: int = 4
    retry_backoff: float = 2.0
    force_catalog: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "AcquireConfig":
        """Build from a config subsection dict, ignoring unknown keys and falling
        back to $HF_TOKEN when no token is given."""
        known = {f.name for f in dc.fields(cls)}
        data = {k: v for k, v in (d or {}).items() if k in known and v is not None}
        if not data.get("hf_token"):
            data["hf_token"] = os.environ.get("HF_TOKEN")
        return cls(**data)

    def public_dict(self) -> dict:
        """asdict() but with the token redacted to a boolean (safe to log)."""
        out = dc.asdict(self)
        out["hf_token"] = bool(self.hf_token)
        return out
