"""data_acquisition — Kathbath two-stage speaker acquisition.

Approach:Catalog first and pull selectively 

Stage 1: Build a metadata catalog

Stage 2: Selective Pull


Modules:
    config   — stage config dataclass (AcquireConfig.from_dict)
    hf_io    — gated-parquet column-projection + selective row-group reader
    catalog  — Stage A metadata catalog (audio column never fetched)
    sampler  — seeded, gender-balanced stratified speaker/clip selection
    puller   — Stage B selective audio pull (row group read once each)
    pipeline — run(cfg, logger): catalog -> sample -> (cost boundary) -> pull
"""
