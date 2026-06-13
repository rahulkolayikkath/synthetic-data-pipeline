"""sentence_generation — synthetic sentence generation using LLM

Generates validated, diverse Indic sentences that IndicF5 will later speak.
Loops over a grid of (language x topic x sentence_type) cells, generating per
cell until each hits its valid quota, so topic/type/language balance is
guaranteed rather than hoped for.

Modules:
    config     — stage config dataclass (grid, quotas, thresholds)
    prompting  — build_prompt / parse_output + SENTENCE_TYPE_GUIDE
    validation — the gate: script / langid / degeneracy / numbers / dedup
    models     — google/gemma-3-12b-it (4-bit) LLM wrapper (load_generator)
    pipeline   — run(cfg, logger): grid loop + checkpointed pool (generate_pool)
"""
