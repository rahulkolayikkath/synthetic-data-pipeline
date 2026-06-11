"""sentence_generation.data — grid, sentence & rejection schemas (§4.4).

Defines:
    - the (language x topic x sentence_type) grid cells and per-cell quotas
    - the validated sentence record (text, language, topic, type, seed, cell)
    - the rejection log entry (text + reason), which becomes the QC-yield table
      in the report

Also owns the checkpointed validated-pool format so a dropped Colab session
resumes without re-doing work.
"""
