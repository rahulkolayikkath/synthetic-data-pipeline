"""sentence_generation.config — stage config dataclass (§4.4).

Ported from the notebook's Config. `languages` maps a language code to
[display_name, script_name] (the YAML lists deserialize as lists, indexed [0]/[1]
exactly like the notebook's tuples). pool/state paths live under the shared out_dir.
"""
from __future__ import annotations

import dataclasses as dc
import os
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SentenceConfig:
    # ---- model ----
    model_id: str = "google/gemma-3-12b-it"
    load_in_4bit: bool = True

    # ---- generation grid: code -> [display name, script in SCRIPT_RANGES] ----
    languages: Dict[str, List[str]] = field(default_factory=lambda: {
        "hi": ["Hindi", "Devanagari"],
        "ml": ["Malayalam", "Malayalam"],
    })
    topics: List[str] = field(default_factory=lambda: [
        "Daily Commute", "Local Cuisine", "Tech Troubleshooting", "Weather Reports",
        "Health and Wellness", "Shopping and Markets", "Festivals and Culture",
        "Travel and Tourism", "Education and School", "Banking and Money",
    ])
    sentence_types: List[str] = field(default_factory=lambda: [
        "declarative", "interrogative", "imperative", "exclamatory",
    ])

    per_cell_quota: int = 13
    sentences_per_call: int = 12
    max_attempts_per_cell: int = 8

    # ---- sampling ----
    temperature: float = 0.9
    top_p: float = 0.95
    max_new_tokens: int = 512

    # ---- validation thresholds ----
    min_words: int = 3
    max_words: int = 20
    script_min_ratio: float = 0.90
    min_type_token_ratio: float = 0.55
    near_dup_threshold: float = 0.92
    langid_backend: str = "fasttext"
    langid_min_conf: float = 0.50
    reject_unnormalized_numbers: bool = True

    embed_model_id: str = "sentence-transformers/LaBSE"

    base_seed: int = 1234
    out_dir: str = "out"

    @property
    def pool_path(self) -> str:
        return os.path.join(self.out_dir, "sentences.jsonl")

    @property
    def state_path(self) -> str:
        return os.path.join(self.out_dir, "sentences_state.json")

    @classmethod
    def from_dict(cls, d: dict) -> "SentenceConfig":
        d = dict(d or {})
        # the unified config uses the global `seed`; map it onto base_seed.
        if "base_seed" not in d and "seed" in d:
            d["base_seed"] = d["seed"]
        known = {f.name for f in dc.fields(cls)}
        data = {k: v for k, v in d.items() if k in known and v is not None}
        return cls(**data)
