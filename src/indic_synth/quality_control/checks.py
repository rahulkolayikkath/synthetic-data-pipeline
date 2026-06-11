"""quality_control.checks — model-free gates + result containers (§4.6).

The hard-failure DSP checks (silence/clipping/truncation/looping/dur-per-char),
the text normalization + CER, and the CheckResult/UtteranceQC containers. All
pure DSP/text, so they run with no model download. Ported verbatim from the
tested qc_validation module.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ----------------------------- result containers --------------------------- #
@dataclass
class CheckResult:
    name: str
    passed: bool
    metric: Optional[float] = None
    detail: str = ""


@dataclass
class UtteranceQC:
    utt_id: str
    passed: bool
    reasons: list = field(default_factory=list)     # why it failed (empty if passed)
    checks: list = field(default_factory=list)
    duration_s: Optional[float] = None
    transcript: Optional[str] = None

    def to_row(self) -> dict:
        """Flatten into a manifest-friendly dict."""
        row = {
            "utt_id": self.utt_id,
            "qc_passed": self.passed,
            "qc_reasons": ";".join(self.reasons),
            "duration_s": self.duration_s,
            "asr_transcript": self.transcript,
        }
        for c in self.checks:
            row[f"{c.name}_pass"] = c.passed
            if c.metric is not None:
                row[f"{c.name}_metric"] = round(float(c.metric), 4)
        return row


# ----------------------------- audio I/O ----------------------------------- #
def load_audio(path: str, target_sr: int):
    """Load -> mono -> float32 in [-1, 1] -> resample to target_sr.

    Resampling is the #1 source of *silent* bugs, so we always read the real sr,
    refuse to 'resample' by reinterpreting, and return the actual sr produced.
    """
    import librosa
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:                       # (n, ch) -> mono
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return np.ascontiguousarray(wav, dtype=np.float32), sr


def _frame_rms(wav, sr, frame_ms=25.0, hop_ms=10.0):
    frame = max(1, int(sr * frame_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    if len(wav) < frame:
        return np.array([np.sqrt(np.mean(wav ** 2) + 1e-12)])
    n = 1 + (len(wav) - frame) // hop
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        seg = wav[i * hop: i * hop + frame]
        out[i] = np.sqrt(np.mean(seg ** 2) + 1e-12)
    return out


def _dbfs(x: float) -> float:
    return 20.0 * np.log10(max(x, 1e-12))


# ----------------------------- hard-failure checks ------------------------- #
def check_silence(wav, sr, cfg) -> CheckResult:
    rms = np.sqrt(np.mean(wav ** 2) + 1e-12)
    dbfs = _dbfs(rms)
    frames = _frame_rms(wav, sr)
    floor = max(frames.max() * 0.1, 10 ** (cfg.silence_rms_dbfs / 20))
    active_frac = float(np.mean(frames > floor)) if frames.size else 0.0
    is_silent = (dbfs < cfg.silence_rms_dbfs) or (active_frac < cfg.silence_active_frac_min)
    return CheckResult("silence", not is_silent, metric=round(dbfs, 2),
                       detail=f"rms={dbfs:.1f}dBFS active_frac={active_frac:.2f}")


def check_clipping(wav, sr, cfg) -> CheckResult:
    clipped = np.abs(wav) >= cfg.clip_sample_thresh
    frac = float(np.mean(clipped))
    run = best = 0
    for c in clipped:
        run = run + 1 if c else 0
        best = max(best, run)
    bad = (frac > cfg.clip_frac_max) or (best >= cfg.clip_run_max)
    return CheckResult("clipping", not bad, metric=round(frac, 6),
                       detail=f"clipped_frac={frac:.4%} max_run={best}")


def check_truncation(wav, sr, cfg) -> CheckResult:
    """Natural utterances decay into trailing silence. If the final window still
    carries near-peak energy, the clip was likely cut off mid-word (heuristic)."""
    frames = _frame_rms(wav, sr)
    if frames.size < 3:
        return CheckResult("truncation", True, metric=0.0, detail="too short to assess")
    peak = float(np.percentile(frames, 95))
    tail_n = max(1, int((cfg.trunc_tail_ms / 1000) / 0.010))   # 10 ms hop
    tail = float(np.mean(frames[-tail_n:]))
    ratio = tail / (peak + 1e-12)
    truncated = ratio > cfg.trunc_tail_ratio
    return CheckResult("truncation", not truncated, metric=round(ratio, 3),
                       detail=f"tail/peak={ratio:.2f} (>{cfg.trunc_tail_ratio} => suspect cut-off)")


def check_looping_audio(wav, sr, cfg) -> CheckResult:
    """Acoustic loop detection via an MFCC self-similarity matrix. A repeated
    segment shows as a strong diagonal at some lag. Complements the text check."""
    import librosa
    if len(wav) < int(sr * (cfg.loop_min_lag_s + 0.2)):
        return CheckResult("looping_audio", True, metric=0.0, detail="too short to assess")
    hop = int(sr * 0.01)
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=20, hop_length=hop)
    rms = librosa.feature.rms(y=wav, frame_length=int(sr * 0.025), hop_length=hop)[0]
    L = min(mfcc.shape[1], rms.shape[0])
    mfcc, rms = mfcc[:, :L], rms[:L]
    active = rms > 0.15 * rms.max()                      # voiced-frame mask
    mfcc = mfcc / (np.linalg.norm(mfcc, axis=0, keepdims=True) + 1e-9)
    sim = mfcc.T @ mfcc                                  # (L, L) cosine self-similarity
    min_lag = int(cfg.loop_min_lag_s / 0.01)
    best_frac, best_lag = 0.0, 0
    for lag in range(min_lag, L - min_lag):
        diag = np.diagonal(sim, offset=lag)              # pairs (i, i+lag)
        pair_active = active[:L - lag] & active[lag:]
        if pair_active.sum() < min_lag:                  # need enough voiced overlap
            continue
        frac = float(np.mean(diag[pair_active] > cfg.loop_pair_sim))
        if frac > best_frac:
            best_frac, best_lag = frac, lag
    looping = best_frac > cfg.loop_band_frac
    return CheckResult("looping_audio", not looping, metric=round(best_frac, 3),
                       detail=f"max_voiced_repeat={best_frac:.2f} @lag={best_lag * 0.01:.2f}s "
                              f"(loop if >{cfg.loop_band_frac})")


def detect_text_repetition(text: str, cfg) -> CheckResult:
    """Catch ASR transcripts where an n-gram repeats consecutively (looping)."""
    toks = text.split()
    n = cfg.text_repeat_ngram
    worst = 1
    if len(toks) >= n:
        for size in range(1, 4):                         # 1..3-token units
            i = 0
            while i + size <= len(toks):
                unit = toks[i:i + size]
                reps, j = 1, i + size
                while toks[j:j + size] == unit:
                    reps += 1
                    j += size
                worst = max(worst, reps)
                i = max(j, i + 1)
    looping = worst >= n
    return CheckResult("looping_text", not looping, metric=float(worst),
                       detail=f"max_consecutive_repeat={worst}")


def check_duration_per_char(wav, sr, text: str, cfg) -> CheckResult:
    n_chars = len(text.strip())
    if n_chars == 0:
        return CheckResult("dur_per_char", False, metric=None, detail="empty intended text")
    dur = len(wav) / sr
    ratio = dur / n_chars
    ok = cfg.dur_per_char_min <= ratio <= cfg.dur_per_char_max
    return CheckResult("dur_per_char", ok, metric=round(ratio, 4),
                       detail=f"{dur:.2f}s / {n_chars} chars = {ratio:.3f} s/char "
                              f"[{cfg.dur_per_char_min}, {cfg.dur_per_char_max}]")


# ----------------------------- text normalization + CER -------------------- #
def normalize_text(text: str) -> str:
    """NFC-normalize (critical for Indic combining chars), collapse whitespace,
    drop common punctuation. Swap in AI4Bharat IndicNormalizer for production."""
    text = unicodedata.normalize("NFC", text)
    for ch in "।.,?!\"'()[]{}:;-—…":
        text = text.replace(ch, " ")
    return " ".join(text.split()).strip()


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character error rate after normalization. Uses jiwer if available."""
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    if not ref:
        return 1.0 if hyp else 0.0
    try:
        import jiwer
        return float(jiwer.cer(ref, hyp))
    except Exception:
        return _levenshtein(ref, hyp) / len(ref)


def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
