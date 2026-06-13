"""data_acquisition.sampler — seeded stratified speaker/clip selection.

Groups by (lang, speaker_id), picks a gender-balanced speaker quota per language
from the whole pool, and selects clips within the reference-duration window —
fully determined by catalog order + cfg.seed. Writes selection_manifest.jsonl.
"""
from __future__ import annotations

import json
import os
import random

import pandas as pd


def _choose_speakers(males, females, others, cfg, rng, logger, lang):
    """Pick speakers for one language, balancing gender where possible."""
    per = cfg.speakers_per_language
    if not cfg.gender_balance:
        pool = males + females + others
        return rng.sample(pool, min(per, len(pool)))

    #equal spliting between male and female
    n_f = per // 2 
    n_m = per - n_f
    chosen = rng.sample(males, min(n_m, len(males))) + rng.sample(females, min(n_f, len(females)))
    deficit = per - len(chosen)
    if deficit > 0:
        chosen_set = set(chosen)
        pool = [s for s in (males + females + others) if s not in chosen_set]
        backfill = rng.sample(pool, min(deficit, len(pool)))
        if backfill:
            logger.info("[%s] gender pool too small for a clean split; backfilled %d speaker(s)",
                        lang, len(backfill))
        chosen += backfill
    return chosen


def build_selection(catalog: pd.DataFrame, cfg, logger):
    """Stratify by (language, speaker, gender), then pick clips per
    speaker. Fully determined by the catalog order + cfg.seed. Writes
    out_dir/selection_manifest.jsonl."""
    rng = random.Random(cfg.seed)

    eligible = catalog[
        (catalog["duration"] >= cfg.ref_min_dur)
        & (catalog["duration"] <= cfg.ref_max_dur)
        & (catalog["text"].astype(str).str.strip().str.len() > 0)
    ].copy()

    records = []
    per_language = {}
    for lang in cfg.languages:
        lang_df = eligible[eligible["lang"] == lang]
        if lang_df.empty:
            logger.warning("[%s] no eligible reference clips in [%.1f, %.1f]s window",
                           lang, cfg.ref_min_dur, cfg.ref_max_dur)
            continue

        spk = (lang_df.groupby("speaker_id")["gender"].first()
                      .reset_index().sort_values("speaker_id"))
        males = list(spk.loc[spk["gender"] == "male", "speaker_id"])
        females = list(spk.loc[spk["gender"] == "female", "speaker_id"])
        others = list(spk.loc[~spk["gender"].isin(["male", "female"]), "speaker_id"])

        chosen = _choose_speakers(males, females, others, cfg, rng, logger, lang)
        male_set, female_set = set(males), set(females)
        for s in chosen:
            clips = lang_df[lang_df["speaker_id"] == s].sort_values(["source_file", "row_index"])
            n = min(cfg.clips_per_speaker, len(clips))
            for idx in rng.sample(list(clips.index), n):
                row = lang_df.loc[idx]
                records.append({
                    "ref_id": f"{lang}_spk{int(row.speaker_id)}_{int(row.row_index)}",
                    "lang": lang,
                    "speaker_id": int(row.speaker_id),
                    "gender": row.gender,
                    "source_repo": cfg.repo_id if cfg.source == "hf" else cfg.local_dir,
                    "source_split": cfg.split,
                    "source_file": row.source_file,
                    "row_index": int(row.row_index),
                    "fname": str(row.get("fname", "")),
                    "ref_text": str(row.text),
                    "duration": float(row.duration),
                    "local_audio_path": None,
                    "status": "selected",
                })
        per_language[lang] = {
            "chosen_speakers": len(chosen),
            "male": sum(1 for s in chosen if s in male_set),
            "female": sum(1 for s in chosen if s in female_set),
            "available_speakers": int(len(spk)),
        }

    total_speakers = len({(r["lang"], r["speaker_id"]) for r in records})
    if total_speakers < cfg.min_total_speakers:
        logger.warning(
            "Only %d distinct speakers selected (< min %d). Raise speakers_per_language, "
            "add a language, or fall back to split='train' for more speakers.",
            total_speakers, cfg.min_total_speakers,
        )

    os.makedirs(cfg.out_dir, exist_ok=True)
    man_path = os.path.join(cfg.out_dir, "selection_manifest.jsonl")
    with open(man_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {"total_clips": len(records), "total_speakers": total_speakers,
               "per_language": per_language}
    logger.info("Selected %d clips across %d speakers; saved %s",
                len(records), total_speakers, man_path)
    return records, summary
