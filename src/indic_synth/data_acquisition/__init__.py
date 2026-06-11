"""data_acquisition — §4.2 Kathbath two-stage speaker acquisition.

Stage 1 of the pipeline: pull a small, diverse, gender-balanced bank of real
speakers and reference clips from ai4bharat/Kathbath WITHOUT downloading the
~170 GB dataset. Kathbath is used purely as a voice bank (reference audio +
reference text to condition the zero-shot TTS), not as train/eval data.

Modules:
    config   — stage config dataclass (AcquireConfig.from_dict)
    hf_io    — gated-parquet column-projection + selective row-group reader
    catalog  — Stage A metadata catalog (audio column never fetched)
    sampler  — seeded, gender-balanced stratified speaker/clip selection
    puller   — Stage B selective audio pull (row group read once each)
    pipeline — run(cfg, logger): catalog -> sample -> (cost boundary) -> pull
"""
