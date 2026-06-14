"""quality_control.models — ASR & speaker-verification wrappers

Lazy-loaded, cached wrappers for the two model-backed gates:
  - indic-conformer-600m  -> transcribe (content fidelity / CER)
  - ECAPA (speechbrain)   -> speaker_cosine (speaker fidelity)

Loaded via lru_cache so importing the package is cheap and the DSP checks in
checks.py run with no model download. 
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _disable_jit_fuser() -> None:
    """Disable TorchScript's tensor-expression / nvFuser JIT fusion.

    The STFT front-end inside ECAPA and IndicConformer produces complex tensors;
    on some torch/CUDA combos the JIT fuser emits invalid CUDA for the complex
    `abs` ("name followed by '::' must be a class or namespace name" referencing
    c10::complex<float>) and the model call dies. Turning the fuser off forces the
    eager path. Each call is guarded so it is a no-op on torch versions that lack
    a given knob. Cached so it runs exactly once, before any model executes.
    """
    import torch

    for fn, args in [
        (getattr(torch._C, "_jit_set_texpr_fuser_enabled", None), (False,)),
        (getattr(torch._C, "_jit_override_can_fuse_on_cpu", None), (False,)),
        (getattr(torch._C, "_jit_override_can_fuse_on_gpu", None), (False,)),
        (getattr(torch._C, "_jit_set_nvfuser_enabled", None), (False,)),
        (getattr(torch._C, "_jit_set_profiling_mode", None), (False,)),
        (getattr(torch._C, "_jit_set_profiling_executor", None), (False,)),
    ]:
        if fn is None:
            continue
        try:
            fn(*args)
        except Exception:
            pass


@lru_cache(maxsize=1)
def _load_asr(model_id: str, device: str):
    _disable_jit_fuser()
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    try:
        model = model.to(device)
    except Exception:
        pass
    return model


@lru_cache(maxsize=1)
def _load_spk(model_id: str, device: str):
    _disable_jit_fuser()
    from speechbrain.inference.speaker import EncoderClassifier
    return EncoderClassifier.from_hparams(source=model_id, run_opts={"device": device})


def warmup(cfg, run_asr: bool = True, run_speaker: bool = True) -> None:
    """Eagerly load the cached models so their load time can be measured separately
    from per-utterance processing (otherwise the lazy load lands on the first clip)."""
    if run_asr:
        _load_asr(cfg.asr_model_id, cfg.device)
    if run_speaker:
        _load_spk(cfg.spk_model_id, cfg.device)


def transcribe(wav: np.ndarray, sr: int, lang_code: str, cfg) -> str:
    """Transcribe with IndicConformer. wav must already be at cfg.asr_sr (16k)."""
    import torch
    model = _load_asr(cfg.asr_model_id, cfg.device)
    t = torch.from_numpy(wav).float().unsqueeze(0)          # (1, n)
    out = model(t, lang_code, cfg.asr_decoding)
    return out[0] if isinstance(out, (list, tuple)) else str(out)


def speaker_cosine(synth: np.ndarray, ref: np.ndarray, sr: int, cfg) -> float:
    """Cosine similarity between ECAPA embeddings of synth and reference clips."""
    import torch
    enc = _load_spk(cfg.spk_model_id, cfg.device)

    def embed(x: np.ndarray) -> np.ndarray:
        e = enc.encode_batch(torch.from_numpy(x).float().unsqueeze(0))
        return e.squeeze().detach().cpu().numpy()

    a, b = embed(synth), embed(ref)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
