"""sentence_generation.models — the Indic LLM wrapper (§4.4).

Loads google/gemma-3-12b-it in 4-bit NF4 (fits a T4) and returns a seeded
`generate(prompt, seed) -> str` callable. Heavy deps (torch/transformers) are
imported lazily so the rest of the module (validation/prompting) stays importable
on a CPU-only box. Ported from the notebook's model cell.
"""
from __future__ import annotations


def load_generator(cfg, logger=None):
    """Return a `generate(prompt, seed) -> str` closure over a loaded Gemma model."""
    import torch
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Gemma3ForConditionalGeneration, set_seed)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ) if cfg.load_in_4bit else None

    processor = AutoProcessor.from_pretrained(cfg.model_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        cfg.model_id,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    if logger:
        logger.info("Model loaded: %s", cfg.model_id)

    @torch.inference_mode()
    def generate(prompt: str, seed: int) -> str:
        """One chat completion. Seeded so the same (prompt, seed) is reproducible."""
        set_seed(seed)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        in_len = inputs["input_ids"].shape[-1]
        out = model.generate(
            **inputs,
            do_sample=True,                      # sampling (not greedy) -> diverse sentences
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_new_tokens=cfg.max_new_tokens,
        )
        return processor.decode(out[0][in_len:], skip_special_tokens=True)

    return generate
