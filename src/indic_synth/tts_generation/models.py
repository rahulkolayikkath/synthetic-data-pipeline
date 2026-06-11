"""tts_generation.models — the IndicF5 voice-cloning wrapper (§4.5).

Wraps ai4bharat/IndicF5 (loaded via AutoModel + trust_remote_code, as on the model
card). `synthesize(text, ref_audio_path, ref_text)` returns float32 audio at 24 kHz,
handling the int16 -> float32/32768 conversion the model card / notebook show. Heavy
deps (torch/transformers) are imported lazily so the package imports on CPU boxes.
"""
from __future__ import annotations

import numpy as np


class IndicF5:
    def __init__(self, cfg, logger=None):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(cfg.model_id, trust_remote_code=True)
        self.model = self.model.to(cfg.device)
        self.target_sr = cfg.target_sr
        if logger:
            logger.info("IndicF5 loaded: %s on %s", cfg.model_id, cfg.device)

    def synthesize(self, text: str, ref_audio_path: str, ref_text: str) -> np.ndarray:
        """Generate `text` in the reference voice. Returns float32 @ target_sr."""
        audio = self.model(text, ref_audio_path=ref_audio_path, ref_text=ref_text)
        audio = np.asarray(audio)
        if audio.dtype == np.int16:                  # model may emit PCM16
            audio = audio.astype(np.float32) / 32768.0
        return audio.astype(np.float32)
