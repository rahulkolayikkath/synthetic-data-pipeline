"""common.config — unified config loading, seeds & determinism (§4.1).

One `config.yaml` drives the whole run. `load_config` parses it into a `Config`
with a global section (`seed`, `out_dir`) plus one subsection per stage; each
stage builds its own (existing, tested) dataclass from `cfg.stage("<name>")`,
so the unified config stays light-touch and doesn't force a rewrite of the
migrated modules.

`set_determinism` / `derive_seed` are ported from the sentence-generation
notebook. Heavy deps (numpy/torch) are imported lazily so importing `common`
on a CPU-only box (e.g. the smoke tests) never pulls in torch.
"""
from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Config:
    """Parsed config.yaml: global knobs + raw per-stage subsections."""
    seed: int = 1234
    out_dir: str = "out"
    raw: Dict[str, Any] = field(default_factory=dict)

    def stage(self, name: str) -> Dict[str, Any]:
        """Return the subsection dict for a stage (e.g. 'data_acquisition').

        The global seed/out_dir are injected as defaults so a stage adapter can
        splat the dict straight into its dataclass without re-reading globals.
        """
        sub = dict(self.raw.get(name, {}) or {})
        sub.setdefault("seed", self.seed)
        sub.setdefault("out_dir", self.out_dir)
        return sub


def load_config(path: str) -> Config:
    """Load config.yaml into a Config. Requires pyyaml."""
    import yaml  # lazy: only needed when a real run loads config

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config(
        seed=int(raw.get("seed", 1234)),
        out_dir=str(raw.get("out_dir", "out")),
        raw=raw,
    )


def set_determinism(seed: int) -> None:
    """Pin every RNG + ask Torch for deterministic kernels.

    Note: GPU sampling is not guaranteed bit-exact across batch sizes / hardware;
    the reproducibility guarantee is the *checkpointed manifests* + pinned config,
    not identical token rolls. numpy/torch imported lazily (CPU smokes need neither).
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def derive_seed(base: int, *parts) -> int:
    """Per-call seed: reproducible, but different per (lang, topic, type, attempt)
    so calls don't regenerate identical content."""
    key = "|".join([str(base)] + [str(p) for p in parts])
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
