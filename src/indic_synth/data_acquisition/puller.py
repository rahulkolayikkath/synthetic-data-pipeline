"""data_acquisition.puller — Stage B selective audio pull (§4.2).

Groups selected rows by file then row group, reads each needed row group exactly
once (audio column only), and writes ref_audio/<ref_id>.<ext> + appends
reference_manifest.jsonl. Resumable/idempotent and partial-failure isolated.
Ported from the tested module; imports rewired to package hf_io + common.logging.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

from indic_synth.common.logging import with_retries

from .hf_io import AUDIO_COL, make_source, open_parquet, row_group_for, row_group_offsets


def _extract_audio(value):
    """HF Audio is stored as struct{bytes, path}. Return (bytes, path)."""
    if isinstance(value, dict):
        return value.get("bytes"), value.get("path") or ""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value), ""
    return None, ""


def _sniff_ext(b: bytes, path: str) -> str:
    if path and "." in os.path.basename(path):
        return os.path.splitext(path)[1]
    if len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WAVE":
        return ".wav"
    if len(b) >= 8 and b[4:8] == b"ftyp":
        return ".m4a"
    if b[:3] == b"ID3" or b[:2] == b"\xff\xfb":
        return ".mp3"
    if b[:4] == b"OggS":
        return ".ogg"
    if b[:4] == b"fLaC":
        return ".flac"
    return ".bin"


def _load_done(out_manifest: str, out_dir: str) -> set:
    """ref_ids already downloaded AND still present on disk (idempotent restart)."""
    done = set()
    if not os.path.exists(out_manifest):
        return done
    with open(out_manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") == "downloaded" and rec.get("local_audio_path"):
                if os.path.exists(os.path.join(out_dir, rec["local_audio_path"])):
                    done.add(rec["ref_id"])
    return done


def _write(f, rec):
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    f.flush()


def pull_audio(records, cfg, logger):
    """Stage B. Group selected rows by file, then by row group, and read each
    needed row group exactly once (audio column only). Resumable and idempotent:
    already-downloaded clips are skipped; per-clip failures are isolated."""
    fs, _ = make_source(cfg, logger)
    audio_dir = os.path.join(cfg.out_dir, "ref_audio")
    os.makedirs(audio_dir, exist_ok=True)
    out_manifest = os.path.join(cfg.out_dir, "reference_manifest.jsonl")

    done = _load_done(out_manifest, cfg.out_dir)
    pending = [r for r in records if r["ref_id"] not in done]
    logger.info("%d selected, %d already done, %d to pull", len(records), len(done), len(pending))

    by_file = defaultdict(list)
    for r in pending:
        by_file[r["source_file"]].append(r)

    stats = {"downloaded": 0, "failed": 0, "row_groups_touched": 0, "approx_bytes_touched": 0}
    fout = open(out_manifest, "a", encoding="utf-8")
    try:
        for fp, recs in by_file.items():
            try:
                pf = with_retries(open_parquet, fs, fp, logger=logger, what=f"open {fp}",
                                  max_retries=cfg.max_retries, backoff=cfg.retry_backoff)
            except Exception as exc:
                logger.error("Could not open %s: %s; marking %d clip(s) failed", fp, exc, len(recs))
                for r in recs:
                    _write(fout, {**r, "status": "failed", "error": str(exc)})
                    stats["failed"] += 1
                continue

            starts = row_group_offsets(pf)
            rg_rows = defaultdict(list)
            for r in recs:
                rg_rows[row_group_for(starts, r["row_index"])].append(r)

            for rg, items in rg_rows.items():
                try:
                    tbl = with_retries(pf.read_row_group, rg, columns=[AUDIO_COL], logger=logger,
                                       what=f"read row group {rg} of {os.path.basename(fp)}",
                                       max_retries=cfg.max_retries, backoff=cfg.retry_backoff)
                except Exception as exc:
                    logger.error("Row group %d read failed: %s; %d clip(s) failed", rg, exc, len(items))
                    for r in items:
                        _write(fout, {**r, "status": "failed", "error": str(exc)})
                        stats["failed"] += 1
                    continue

                stats["row_groups_touched"] += 1
                stats["approx_bytes_touched"] += int(pf.metadata.row_group(rg).total_byte_size)
                col = tbl.column(AUDIO_COL)
                for r in items:
                    local = r["row_index"] - starts[rg]
                    try:
                        audio_bytes, path = _extract_audio(col[local].as_py())
                        if not audio_bytes:
                            raise ValueError("empty audio bytes in row")
                        dest = os.path.join(audio_dir, r["ref_id"] + _sniff_ext(audio_bytes, path))
                        with open(dest, "wb") as af:
                            af.write(audio_bytes)
                        _write(fout, {**r, "local_audio_path": os.path.relpath(dest, cfg.out_dir),
                                      "status": "downloaded"})
                        stats["downloaded"] += 1
                    except Exception as exc:
                        _write(fout, {**r, "status": "failed", "error": str(exc)})
                        stats["failed"] += 1
                        logger.warning("%s failed: %s", r["ref_id"], exc)
    finally:
        fout.close()

    logger.info("Pull complete: %s", stats)
    return stats
