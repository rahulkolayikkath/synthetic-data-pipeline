"""data_acquisition.hf_io — cheap gated-parquet access (§4.2).

Column-projection + row-group reading so Stage A never fetches the audio column
and Stage B reads only the row groups holding the chosen rows. Ported verbatim
from the tested kathbath-extraction module (no internal imports to rewire).
"""
from __future__ import annotations

import bisect
import os

import pyarrow.parquet as pq

# Kathbath per-language schema:
#   fname, text, audio_filepath (audio), lang, duration, gender, speaker_id
# We catalog everything EXCEPT the audio column. That single omission is what
# keeps Stage A cheap: parquet stores each column separately, so projecting
# these never fetches the multi-hundred-MB audio chunks.
META_COLS = ["fname", "text", "lang", "duration", "gender", "speaker_id"]
AUDIO_COL = "audio_filepath"


def make_source(cfg, logger):
    """Return (fs, list_files) where list_files(lang) -> sorted parquet paths."""
    if cfg.source == "hf":
        from huggingface_hub import HfFileSystem  # imported lazily; not needed for local runs
        fs = HfFileSystem(token=cfg.hf_token)
        base = f"datasets/{cfg.repo_id}"
    elif cfg.source == "local":
        import fsspec
        fs = fsspec.filesystem("local")
        base = os.path.abspath(cfg.local_dir)
    else:
        raise ValueError(f"Unknown source {cfg.source!r} (use 'hf' or 'local')")

    def list_files(lang: str):
        return sorted(fs.glob(f"{base}/{lang}/{cfg.split}-*.parquet"))

    return fs, list_files


def open_parquet(fs, path: str) -> pq.ParquetFile:
    # fs.open gives a seekable file object; pyarrow reads the footer first, then
    # only the requested column chunks via range reads.
    return pq.ParquetFile(fs.open(path, "rb"))


def row_group_offsets(pf: pq.ParquetFile):
    """Start row index of each row group (parallel to read_row_group order)."""
    starts, running = [], 0
    md = pf.metadata
    for i in range(md.num_row_groups):
        starts.append(running)
        running += md.row_group(i).num_rows
    return starts


def row_group_for(starts, idx: int) -> int:
    return bisect.bisect_right(starts, idx) - 1
