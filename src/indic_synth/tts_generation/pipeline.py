"""tts_generation.pipeline — Stage 4.5 entrypoint (the join point).

`run(cfg, logger)` reads prepared_manifest.jsonl (§4.3 references) + sentences.jsonl
(§4.4 validated text), pairs each sentence with a same-language prepared reference
(round-robin over that language's speakers for balance), chunks long text at
sentence boundaries, synthesizes with IndicF5, writes tts_audio/<utt_id>.wav at
24 kHz, and appends tts_manifest.jsonl in the §4.1 utterance schema (the §4.6 input).

`synthesize_pool` takes the synth callable as an argument so the pairing / chunking /
manifest machinery is testable on CPU with a fake synthesizer (no IndicF5 load).
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict

import numpy as np
import soundfile as sf

from indic_synth.common.checkpoint import load_done
from indic_synth.common.logging import get_logger
from indic_synth.common.manifest import append_jsonl, read_jsonl

from .config import TTSConfig

_SENT_SPLIT = re.compile(r'(?<=[।!?\.])\s+')


def chunk_text(text: str, max_chars: int):
    """Split text into <=max_chars chunks at sentence boundaries (greedy)."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks, cur = [], ""
    for part in _SENT_SPLIT.split(text):
        if not part:
            continue
        if cur and len(cur) + 1 + len(part) > max_chars:
            chunks.append(cur)
            cur = part
        else:
            cur = f"{cur} {part}".strip()
    if cur:
        chunks.append(cur)
    return chunks or [text[:max_chars]]


def _refs_by_lang(prepared, lang_map):
    """Group prepared references by their sentence-language code (via lang_map)."""
    inv = {v: k for k, v in lang_map.items()}    # kathbath name -> code
    by_code = defaultdict(list)
    for r in prepared:
        code = inv.get(r.get("lang"), r.get("lang"))
        by_code[code].append(r)
    for code in by_code:
        by_code[code].sort(key=lambda r: r["ref_id"])   # deterministic order
    return by_code


def synthesize_pool(cfg: TTSConfig, synth, logger=None) -> dict:
    """Pair + synthesize with an injected `synth(text, ref_audio_path, ref_text)`."""
    logger = logger or get_logger("tts")
    out_dir = os.path.abspath(cfg.out_dir)
    audio_dir = os.path.join(out_dir, "tts_audio")
    os.makedirs(audio_dir, exist_ok=True)
    out_manifest = os.path.join(out_dir, "tts_manifest.jsonl")

    prepared = [r for r in read_jsonl(os.path.join(out_dir, "prepared_manifest.jsonl"))
                if r.get("status") == "prepared" and r.get("prepared_audio_path")]
    sentences = read_jsonl(os.path.join(out_dir, "sentences.jsonl"))
    refs = _refs_by_lang(prepared, cfg.lang_map)
    logger.info("%d prepared refs, %d sentences, languages=%s",
                len(prepared), len(sentences), {k: len(v) for k, v in refs.items()})

    # deterministic round-robin pairing: index sentences within their language
    lang_seen = defaultdict(int)
    jobs = []
    for s in sentences:
        lang = s["language"]
        pool = refs.get(lang)
        if not pool:
            logger.warning("no prepared reference for language %r; skipping %s", lang, s["id"])
            continue
        ref = pool[lang_seen[lang] % len(pool)]
        lang_seen[lang] += 1
        jobs.append((s, ref))

    done = load_done(out_manifest, key="utt_id", status="synthesized",
                     audio_field="audio_filepath", out_dir=out_dir)
    pending = [(s, r) for (s, r) in jobs if f"{s['id']}" not in done]
    logger.info("%d jobs, %d already synthesized, %d to do", len(jobs), len(done), len(pending))

    t0 = time.time()
    stats = {"synthesized": 0, "failed": 0}
    for s, ref in pending:
        utt_id = s["id"]
        dst = os.path.join(audio_dir, utt_id + ".wav")
        ref_audio = os.path.join(out_dir, ref["prepared_audio_path"])
        try:
            chunks = chunk_text(s["text"], cfg.max_chars)
            parts = [synth(c, ref_audio, ref.get("ref_text", "")) for c in chunks]
            audio = np.concatenate(parts) if len(parts) > 1 else parts[0]
            sf.write(dst, np.asarray(audio, dtype=np.float32), samplerate=cfg.target_sr)
            duration = round(len(audio) / cfg.target_sr, 3)
        except Exception as exc:
            logger.warning("%s failed: %s", utt_id, exc)
            append_jsonl(out_manifest, {"utt_id": utt_id, "status": "failed", "error": str(exc)})
            stats["failed"] += 1
            continue

        append_jsonl(out_manifest, {
            "utt_id": utt_id,
            "audio_filepath": os.path.relpath(dst, out_dir),
            "text": s["text"],
            "speaker_id": ref.get("speaker_id"),
            "gender": ref.get("gender"),
            "language": s["language"],
            "ref_id": ref["ref_id"],
            "ref_audio_path": ref["prepared_audio_path"],
            "duration": duration,
            "topic": s.get("topic"),
            "sentence_type": s.get("sentence_type"),
            "status": "synthesized",
        })
        stats["synthesized"] += 1

    elapsed = time.time() - t0
    summary = {
        "stage": "tts_generation",
        "elapsed_sec": round(elapsed, 2),
        "utterances_per_min": round(stats["synthesized"] / max(elapsed / 60, 1e-9), 1),
        **stats,
        "tts_manifest": out_manifest,
    }
    with open(os.path.join(out_dir, "tts_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("TTS done: %s", json.dumps(summary, ensure_ascii=False))
    return summary


def run(cfg, logger=None) -> dict:
    """Run §4.5 end-to-end. Loads IndicF5, then synthesize_pool."""
    logger = logger or get_logger("tts")
    tcfg = TTSConfig.from_dict(cfg.stage("tts_generation"))

    from .models import IndicF5
    model = IndicF5(tcfg, logger)
    return synthesize_pool(tcfg, model.synthesize, logger)
