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


# --------------------------------------------------------------------------- #
# config schema + validation (§ optional: installable CLI w/ schema + validation)
# --------------------------------------------------------------------------- #
class ConfigError(ValueError):
    """Raised when config.yaml fails schema validation (with all problems listed)."""


_NUM = (int, float)

# Declarative schema: one section per stage (+ "_global" for top-level scalars).
# Each field spec may set:  type (python type or the _NUM tuple), min/max (inclusive,
# numeric), choices (allowed values), non_empty (for list/dict). Keys present in a
# section but absent from its schema are reported as warnings (likely typos) — never
# fatal — so adding a new dataclass field can't be silently dropped without notice.
CONFIG_SCHEMA = {
    "_global": {
        "seed": {"type": int, "min": 0},
        "out_dir": {"type": str},
    },
    "data_acquisition": {
        "source": {"type": str, "choices": ["hf", "local"]},
        "repo_id": {"type": str},
        "local_dir": {"type": str},
        "languages": {"type": list, "non_empty": True},
        "split": {"type": str},
        "speakers_per_language": {"type": int, "min": 1},
        "clips_per_speaker": {"type": int, "min": 1},
        "min_total_speakers": {"type": int, "min": 1},
        "gender_balance": {"type": bool},
        "ref_min_dur": {"type": _NUM, "min": 0},
        "ref_max_dur": {"type": _NUM, "min": 0},
        "hf_token": {"type": str},
        "max_retries": {"type": int, "min": 0},
        "retry_backoff": {"type": _NUM, "min": 0},
        "force_catalog": {"type": bool},
    },
    "audio_engineering": {
        "in_manifest": {"type": str},
        "target_sr": {"type": int, "min": 1},
        "norm": {"type": str, "choices": ["peak", "rms", "lufs"]},
        "peak_dbfs": {"type": _NUM, "max": 0},
        "rms_dbfs": {"type": _NUM, "max": 0},
        "trim": {"type": bool},
        "max_ref_sec": {"type": _NUM, "min": 0},
        "subtype": {"type": str},
    },
    "sentence_generation": {
        "model_id": {"type": str},
        "load_in_4bit": {"type": bool},
        "languages": {"type": dict, "non_empty": True},
        "topics": {"type": list, "non_empty": True},
        "sentence_types": {"type": list, "non_empty": True},
        "per_cell_quota": {"type": int, "min": 1},
        "sentences_per_call": {"type": int, "min": 1},
        "max_attempts_per_cell": {"type": int, "min": 1},
        "temperature": {"type": _NUM, "min": 0},
        "top_p": {"type": _NUM, "min": 0, "max": 1},
        "max_new_tokens": {"type": int, "min": 1},
        "min_words": {"type": int, "min": 1},
        "max_words": {"type": int, "min": 1},
        "script_min_ratio": {"type": _NUM, "min": 0, "max": 1},
        "min_type_token_ratio": {"type": _NUM, "min": 0, "max": 1},
        "near_dup_threshold": {"type": _NUM, "min": 0, "max": 1},
        "langid_backend": {"type": str},
        "langid_min_conf": {"type": _NUM, "min": 0, "max": 1},
        "reject_unnormalized_numbers": {"type": bool},
        "embed_model_id": {"type": str},
        "base_seed": {"type": int, "min": 0},
    },
    "tts_generation": {
        "model_id": {"type": str},
        "target_sr": {"type": int, "min": 1},
        "max_chars": {"type": int, "min": 1},
        "device": {"type": str},
        "log_every": {"type": int, "min": 1},
        "lang_map": {"type": dict, "non_empty": True},
    },
    "quality_control": {
        "asr_model_id": {"type": str},
        "asr_decoding": {"type": str, "choices": ["ctc", "rnnt"]},
        "spk_model_id": {"type": str},
        "asr_sr": {"type": int, "min": 1},
        "spk_sr": {"type": int, "min": 1},
        "cer_max": {"type": _NUM, "min": 0, "max": 1},
        "spk_cos_min": {"type": _NUM, "min": -1, "max": 1},
        "dur_per_char_min": {"type": _NUM, "min": 0},
        "dur_per_char_max": {"type": _NUM, "min": 0},
        "silence_rms_dbfs": {"type": _NUM, "max": 0},
        "silence_active_frac_min": {"type": _NUM, "min": 0, "max": 1},
        "clip_sample_thresh": {"type": _NUM, "min": 0, "max": 1},
        "clip_frac_max": {"type": _NUM, "min": 0, "max": 1},
        "clip_run_max": {"type": int, "min": 0},
        "trunc_tail_ms": {"type": _NUM, "min": 0},
        "trunc_tail_ratio": {"type": _NUM, "min": 0},
        "loop_min_lag_s": {"type": _NUM, "min": 0},
        "loop_pair_sim": {"type": _NUM, "min": 0, "max": 1},
        "loop_band_frac": {"type": _NUM, "min": 0, "max": 1},
        "text_repeat_ngram": {"type": int, "min": 1},
        "device": {"type": str},
        "log_every": {"type": int, "min": 1},
    },
}

# seed/out_dir are injected into every stage subsection by Config.stage(), so they're
# always allowed inside a stage section without triggering an "unknown key" warning.
_STAGE_COMMON = {"seed", "out_dir"}


def _check_value(path, v, spec):
    """Return a list of error strings for one value against its field spec."""
    errs = []
    t = spec.get("type")
    if t is bool:
        if not isinstance(v, bool):
            return [f"{path}: expected bool, got {type(v).__name__}"]
    elif t is int:
        if isinstance(v, bool) or not isinstance(v, int):
            return [f"{path}: expected int, got {type(v).__name__}"]
    elif t is _NUM:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return [f"{path}: expected number, got {type(v).__name__}"]
    elif t is not None and not isinstance(v, t):
        return [f"{path}: expected {t.__name__}, got {type(v).__name__}"]

    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if "min" in spec and v < spec["min"]:
            errs.append(f"{path}: {v} is below minimum {spec['min']}")
        if "max" in spec and v > spec["max"]:
            errs.append(f"{path}: {v} is above maximum {spec['max']}")
    if "choices" in spec and v not in spec["choices"]:
        errs.append(f"{path}: {v!r} not one of {spec['choices']}")
    if spec.get("non_empty") and hasattr(v, "__len__") and len(v) == 0:
        errs.append(f"{path}: must not be empty")
    return errs


def _cross_field_errors(section, d):
    """Relational checks that span two fields within a section."""
    errs = []
    pairs = {
        "data_acquisition": ("ref_min_dur", "ref_max_dur", "<"),
        "quality_control": ("dur_per_char_min", "dur_per_char_max", "<"),
        "sentence_generation": ("min_words", "max_words", "<="),
    }
    if section in pairs:
        lo_k, hi_k, op = pairs[section]
        lo, hi = d.get(lo_k), d.get(hi_k)
        if isinstance(lo, _NUM) and isinstance(hi, _NUM) and not (isinstance(lo, bool) or isinstance(hi, bool)):
            bad = (lo >= hi) if op == "<" else (lo > hi)
            if bad:
                errs.append(f"{section}.{lo_k} ({lo}) must be {op} {section}.{hi_k} ({hi})")
    return errs


def validate_config(cfg: "Config", logger=None) -> None:
    """Validate a loaded Config against CONFIG_SCHEMA.

    Raises ConfigError listing every problem if any value is the wrong type / out of
    range / not an allowed choice, or if two related fields are inconsistent. Unknown
    keys (likely typos) are logged as warnings, not errors — so a valid config never
    breaks, but mistakes surface before any model loads.
    """
    from .logging import get_logger
    log = logger or get_logger("config")
    raw = cfg.raw or {}
    errors, warnings = [], []

    for k, v in raw.items():
        if k in CONFIG_SCHEMA["_global"]:
            errors += _check_value(k, v, CONFIG_SCHEMA["_global"][k])
            continue
        if k not in CONFIG_SCHEMA:
            warnings.append(f"unknown top-level key {k!r} (ignored)")
            continue
        section = v or {}
        if not isinstance(section, dict):
            errors.append(f"{k}: expected a mapping/section, got {type(section).__name__}")
            continue
        sec_schema = CONFIG_SCHEMA[k]
        for fk, fv in section.items():
            if fk in sec_schema:
                errors += _check_value(f"{k}.{fk}", fv, sec_schema[fk])
            elif fk not in _STAGE_COMMON:
                warnings.append(f"unknown key {k}.{fk!r} (ignored — possible typo)")
        errors += _cross_field_errors(k, section)

    for w in warnings:
        log.warning("config: %s", w)
    if errors:
        raise ConfigError("invalid config:\n  - " + "\n  - ".join(errors))
    n_sections = sum(1 for k in raw if k in CONFIG_SCHEMA)
    log.info("config OK: %d section(s) validated, %d warning(s).", n_sections, len(warnings))
