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
def _load_asr(model_id: str, device: str):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    try:
        model = model.to(device)
    except Exception:
        pass
    return model


@lru_cache(maxsize=1)
def _load_spk(model_id: str, device: str):
    from speechbrain.inference.speaker import EncoderClassifier
    return EncoderClassifier.from_hparams(source=model_id, run_opts={"device": device})


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
