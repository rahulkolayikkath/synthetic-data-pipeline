"""data_acquisition.catalog — Stage A metadata catalog.

Reads only metadata columns (audio skipped) into out_dir/catalog.parquet, sorted
into a stable order so the seeded sampler is fully reproducible. 
"""
from __future__ import annotations

import os

import pandas as pd

from indic_synth.common.logging import with_retries

from .hf_io import META_COLS, make_source, open_parquet


def _norm_gender(g: str) -> str:
    g = (g or "").strip().lower()
    if g in ("m", "male"):
        return "male"
    if g in ("f", "female"):
        return "female"
    return g or "unknown"


def build_catalog(cfg, logger) -> pd.DataFrame:
    """Stage A. Read only metadata columns (no audio) and build a catalog of
    every utterance with its source file + row index, so Stage B knows exactly
    which rows to fetch. Cached to out_dir/catalog.parquet."""
    cat_path = os.path.join(cfg.out_dir, "catalog.parquet")
    if os.path.exists(cat_path) and not cfg.force_catalog:
        logger.info("Loading cached catalog: %s", cat_path)
        return pd.read_parquet(cat_path)

    fs, list_files = make_source(cfg, logger)
    frames = []
    for lang in cfg.languages:
        files = list_files(lang)
        if not files:
            logger.warning("No %s files found for language %r at the source", cfg.split, lang)
            continue
        logger.info("[%s] cataloging %d parquet file(s) (audio column skipped)", lang, len(files))
        for fp in files:
            pf = with_retries(open_parquet, fs, fp, logger=logger, what=f"open {fp}",
                              max_retries=cfg.max_retries, backoff=cfg.retry_backoff)
            tbl = with_retries(pf.read, columns=META_COLS, logger=logger,
                               what=f"read meta {os.path.basename(fp)}",
                               max_retries=cfg.max_retries, backoff=cfg.retry_backoff)
            df = tbl.to_pandas()
            df["row_index"] = range(len(df))   # global row index within this file
            df["source_file"] = fp
            df["lang"] = lang                  # normalize to the config's language name
            frames.append(df)
            logger.info("[%s] %s -> %d rows", lang, os.path.basename(fp), len(df))

    if not frames:
        raise RuntimeError("Catalog is empty. Check languages/split/source settings.")

    catalog = pd.concat(frames, ignore_index=True)
    catalog["gender"] = catalog["gender"].astype(str).map(_norm_gender)
    # Deterministic order so the seeded sampler is fully reproducible.
    catalog = (catalog.sort_values(["lang", "speaker_id", "source_file", "row_index"])
                       .reset_index(drop=True))

    os.makedirs(cfg.out_dir, exist_ok=True)
    catalog.to_parquet(cat_path, index=False)
    logger.info("Catalog: %d rows, %d distinct speakers across %d language(s); saved %s",
                len(catalog), catalog["speaker_id"].nunique(),
                catalog["lang"].nunique(), cat_path)
    return catalog
