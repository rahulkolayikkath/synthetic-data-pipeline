"""data_acquisition — §4.2 Kathbath two-stage speaker acquisition.

Stage 1 of the pipeline: pull a small, diverse, gender-balanced bank of real
speakers and reference clips from ai4bharat/Kathbath WITHOUT downloading the
~170 GB dataset. Kathbath is used purely as a voice bank (reference audio +
reference text to condition the zero-shot TTS), not as train/eval data.

Modules:
    data     — catalog / selection / reference manifest schemas
    models   — gated-parquet column-projection + selective row-group reader
    pipeline — Stage A (metadata catalog → seeded stratified sampler) then,
               across the cost boundary, Stage B (pull only chosen row groups)
"""
