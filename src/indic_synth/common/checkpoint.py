"""common.checkpoint — idempotent resume helpers (§4.1).

Generalizes the identical "skip already-done" logic that puller.py and
prepare_refs.py each hand-rolled: read an output manifest, collect the ids that
already succeeded (and whose audio is still on disk), so a dropped Colab session
resumes without redoing work. Stages append one validated line at a time
(common.manifest.append_jsonl), so the manifest itself *is* the checkpoint.
"""
from __future__ import annotations

import os
from typing import Optional, Set

from .manifest import iter_jsonl


def load_done(
    manifest_path: str,
    key: str = "ref_id",
    *,
    status: Optional[str] = None,
    audio_field: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> Set[str]:
    """Return the set of `key` values already completed in `manifest_path`.

    Optional guards make the resume honest rather than just "a line exists":
      - status:      only count rows whose "status" equals this (e.g. "downloaded").
      - audio_field: only count rows whose audio file is still present on disk
                     (resolved against out_dir when the path is relative).
    """
    done: Set[str] = set()
    for rec in iter_jsonl(manifest_path):
        if key not in rec:
            continue
        if status is not None and rec.get("status") != status:
            continue
        if audio_field is not None:
            rel = rec.get(audio_field)
            if not rel:
                continue
            path = rel if os.path.isabs(rel) or out_dir is None else os.path.join(out_dir, rel)
            if not os.path.exists(path):
                continue
        done.add(rec[key])
    return done


def pending(records, done: Set[str], key: str = "ref_id"):
    """Filter input records down to those whose `key` is not already done."""
    return [r for r in records if r.get(key) not in done]
