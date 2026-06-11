"""sentence_generation.pipeline — grid loop + validation gate (§4.4).

Orchestration:
    for each (language x topic x sentence_type) cell, keep generating small
    batches until the cell hits its valid quota.

A programmatic validation gate — not the prompt — is the source of truth:
    script check -> language-ID -> normalization -> degeneracy -> dedup.
Every rejection is logged with a reason (feeds the QC-yield table).

The validated pool is checkpointed to disk so a dropped session resumes without
re-doing work. Per-call seeds give reproducibility and diversity.
"""
