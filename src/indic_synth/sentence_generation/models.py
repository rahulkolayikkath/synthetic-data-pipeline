"""sentence_generation.models — the Indic LLM wrapper (§4.4).

Wraps google/gemma-3-12b-it (4-bit): instruction-tuned, strong Indic script
coverage, fits a T4. Generates in small batches (~12 per call) to avoid
long-generation degeneracy, with per-call seeds derived from a base seed for
reproducibility + diversity.

Swap targets noted in spec: Sarvam-M (24B) on an L4/A100, or Gemma 3 4B-it for
more headroom.
"""
