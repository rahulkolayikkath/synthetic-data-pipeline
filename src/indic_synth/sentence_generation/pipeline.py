"""sentence_generation.pipeline

`run(cfg, logger)` generates a balanced, validated pool of Indic sentences by
looping over a grid of (language x topic x sentence_type) cells until each hits
its quota. Writes the checkpointed sentences.jsonl (the TTS generation input) + a state file of rejection counters, and returns a QC-yield summary.

`generate_pool` is split out and takes the LLM generator + validation resources as
arguments, so the grid/checkpoint/validation machinery can be tested on CPU with a
fake generator (see scripts/smoke_sentences.py) without loading Gemma.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter

from indic_synth.common.config import derive_seed
from indic_synth.common.logging import get_logger
from indic_synth.common.manifest import append_jsonl, iter_jsonl

from .config import SentenceConfig
from .prompting import build_prompt, parse_output
from .validation import validate_sentence


def _cell_id(lang, topic, st):
    return f"{lang}|{topic}|{st}"


def generate_pool(cfg: SentenceConfig, generate, deduper, langid, logger=None,
                  model_load_sec: float = 0.0) -> dict:
    """Run the grid loop with injected resources. `generate(prompt, seed)->str`.

    `model_load_sec` is the time `run()` spent loading the LLM + validation resources;
    it's folded into the reported `elapsed_sec` so the summary reflects the actual
    stage wall-clock, while `process_sec` keeps the generation-only time.
    """
    logger = logger or get_logger("gen")
    os.makedirs(cfg.out_dir, exist_ok=True)

    pool, cell_counts, rejection_stats = [], Counter(), Counter()

    # ---- resume from checkpoint ----
    for rec in iter_jsonl(cfg.pool_path):
        pool.append(rec)
        deduper.add(rec["text"], rec["language"])
        cell_counts[_cell_id(rec["language"], rec["topic"], rec["sentence_type"])] += 1
    if pool:
        logger.info("Resumed %d sentences from checkpoint.", len(pool))
    if os.path.exists(cfg.state_path):
        with open(cfg.state_path, encoding="utf-8") as f:
            rejection_stats = Counter(json.load(f).get("rejection_stats", {}))

    def save_state():
        with open(cfg.state_path, "w", encoding="utf-8") as f:
            json.dump({"rejection_stats": dict(rejection_stats)}, f, ensure_ascii=False)

    cells = [(lc, t, st) for lc in cfg.languages for t in cfg.topics for st in cfg.sentence_types]
    t0 = time.time()
    logger.info("Grid: %d cells x quota %d (~%d target sentences); already have %d.",
                len(cells), cfg.per_cell_quota, len(cells) * cfg.per_cell_quota, len(pool))

    for ci, (lang_code, topic, stype) in enumerate(cells, 1):
        name, script = cfg.languages[lang_code][0], cfg.languages[lang_code][1]
        cid = _cell_id(lang_code, topic, stype)
        attempts = 0

        while cell_counts[cid] < cfg.per_cell_quota and attempts < cfg.max_attempts_per_cell:
            need = cfg.per_cell_quota - cell_counts[cid]
            n = min(cfg.sentences_per_call, max(need, 4))
            seed = derive_seed(cfg.base_seed, lang_code, topic, stype, attempts)
            attempts += 1

            try:
                raw = generate(build_prompt(name, script, topic, stype, n), seed)
            except Exception as e:                       # keep the run alive
                logger.warning("gen failed %s attempt %d: %s", cid, attempts, e)
                continue

            for cand in parse_output(raw):
                if cell_counts[cid] >= cfg.per_cell_quota:
                    break
                ok, reason = validate_sentence(cand, lang_code, script, deduper, langid, cfg)
                if ok:
                    deduper.add(cand, lang_code)
                    rec = {
                        "id": f"{lang_code}_{len(pool):06d}",
                        "language": lang_code, "language_name": name,
                        "topic": topic, "sentence_type": stype, "text": cand,
                        "word_count": len(cand.split()), "char_count": len(cand), "seed": seed,
                    }
                    pool.append(rec)
                    cell_counts[cid] += 1
                    append_jsonl(cfg.pool_path, rec)
                else:
                    rejection_stats[reason] += 1
            save_state()

        logger.info("[%d/%d] %s -> %d/%d valid (%d attempts, %d total, %.0fs elapsed)",
                    ci, len(cells), cid, cell_counts[cid], cfg.per_cell_quota,
                    attempts, len(pool), time.time() - t0)
        if cell_counts[cid] < cfg.per_cell_quota:
            logger.warning("cell under quota: %s -> %d/%d", cid, cell_counts[cid], cfg.per_cell_quota)

    process_sec = round(time.time() - t0, 2)
    total_rej = sum(rejection_stats.values())
    total_seen = total_rej + len(pool)
    summary = {
        "stage": "sentence_generation",
        "elapsed_sec": round(model_load_sec + process_sec, 2),  # actual wall-clock
        "process_sec": process_sec,                             # generation only
        "model_load_sec": round(model_load_sec, 2),             # Gemma + LaBSE + fastText
        "total_valid": len(pool),
        "total_seen": total_seen,
        "yield": round(len(pool) / max(total_seen, 1), 4),
        "rejection_stats": dict(rejection_stats),
        "sentences": cfg.pool_path,
    }
    with open(os.path.join(cfg.out_dir, "sentences_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Sentence generation done: %d valid, yield %.1f%%",
                len(pool), 100 * summary["yield"])
    return summary


def run(cfg, logger=None) -> dict:
    """Run §4.4 end-to-end. Loads Gemma + LaBSE + fastText, then generate_pool."""
    logger = logger or get_logger("gen")
    scfg = SentenceConfig.from_dict(cfg.stage("sentence_generation"))

    from sentence_transformers import SentenceTransformer

    from .models import load_generator
    from .validation import Deduper, LanguageIdentifier

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    t_load = time.time()
    generate = load_generator(scfg, logger)
    embed_model = SentenceTransformer(scfg.embed_model_id, device=device)
    deduper = Deduper(embed_model, scfg.near_dup_threshold)
    langid = LanguageIdentifier(scfg.langid_backend, model_dir=scfg.out_dir)
    model_load_sec = time.time() - t_load

    return generate_pool(scfg, generate, deduper, langid, logger,
                         model_load_sec=model_load_sec)
