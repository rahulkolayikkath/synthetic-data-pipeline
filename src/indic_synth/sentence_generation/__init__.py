"""sentence_generation — §4.4 synthetic sentence generation.

Generates validated, diverse Indic sentences that IndicF5 will later speak.
Loops over a grid of (language x topic x sentence_type) cells, generating per
cell until each hits its valid quota, so topic/type/language balance is
guaranteed rather than hoped for.

Modules:
    data     — grid cells, sentence records, rejection-reason log
    models   — google/gemma-3-12b-it (4-bit) LLM wrapper
    pipeline — grid loop + programmatic validation gate + checkpointed pool
"""
