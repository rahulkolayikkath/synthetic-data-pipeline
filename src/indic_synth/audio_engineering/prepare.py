"""audio_engineering.prepare — DSP for reference-audio engineering (§4.3).

The per-clip signal chain that turns a raw Kathbath clip into what IndicF5 expects
(24 kHz, mono, normalized WAV), and the metering/normalization helpers it uses.
Ported verbatim from the tested prepare_refs.py; the only change is that the driver
loop and argparse/logging live in pipeline.py / common now.

Silent-resampling-bug guards (the point of this stage) live in `process_clip`:
decode at native rate, resample explicitly, assert post-conditions, read back the
written header. See process_clip for the step-by-step.
"""
from __future__ import annotations

import math
import os
import subprocess
import tempfile

import numpy as np

TARGET_SR = 24000
NEAR_SILENCE_DBFS = -50.0   # out RMS below this -> flag
INPUT_CLIP_PEAK = 0.999     # input peak at/above this -> flag (source already clipped)


# ----------------------------- audio I/O ----------------------------------- #
def load_audio_native(path):
    """Decode at the source's own sample rate. Returns
    (samples[n, ch] float32, native_sr, channels, backend, src_format).
    Tries soundfile (wav/flac/ogg), then librosa, then ffmpeg (mp3/m4a/aac)."""
    src_format = os.path.splitext(path)[1].lstrip(".").lower() or "unknown"

    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)  # (frames, channels)
        return data, int(sr), data.shape[1], "soundfile", src_format
    except Exception as e_sf:
        last = f"soundfile: {e_sf}"

    try:
        import librosa  # handles mp3/m4a via audioread/ffmpeg
        y, sr = librosa.load(path, sr=None, mono=False)   # sr=None => NATIVE rate
        y = y[None, :] if y.ndim == 1 else y              # (channels, frames)
        return y.T.astype(np.float32), int(sr), y.shape[0], "librosa", src_format
    except Exception as e_lib:
        last += f" | librosa: {e_lib}"

    data, sr = _decode_ffmpeg(path)   # raises on failure
    return data, sr, data.shape[1], "ffmpeg", src_format


def _decode_ffmpeg(path):
    """Decode to PCM WAV preserving native rate + channels (no -ar/-ac, so ffmpeg
    does NOT resample). Then read with soundfile."""
    import soundfile as sf
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-c:a", "pcm_s16le", tmp.name],
            check=True, capture_output=True,
        )
        data, sr = sf.read(tmp.name, dtype="float32", always_2d=True)
        return data, int(sr)
    except FileNotFoundError as e:
        raise RuntimeError(f"Cannot decode {path}: ffmpeg not installed (needed for "
                           f"{os.path.splitext(path)[1]} files). Install ffmpeg or librosa.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed to decode {path}: {e.stderr.decode(errors='ignore')[:200]}") from e
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


def to_mono(data):
    """(frames, channels) -> (frames,). Downmix by averaging channels."""
    return data[:, 0] if data.shape[1] == 1 else data.mean(axis=1)


def resample(y, sr_in, sr_out):
    """Explicit, high-quality resample. Returns (y_out, method)."""
    if sr_in == sr_out:
        return y, "none"
    try:
        import soxr
        out = soxr.resample(y, sr_in, sr_out, quality="HQ")
        method = "soxr_hq"
    except ImportError:
        try:
            import librosa
            out = librosa.resample(y, orig_sr=sr_in, target_sr=sr_out, res_type="kaiser_best")
            method = "librosa_kaiser_best"
        except ImportError:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr_in, sr_out)
            out = resample_poly(y, sr_out // g, sr_in // g)
            method = "scipy_resample_poly"
    expected = round(len(y) * sr_out / sr_in)
    if abs(len(out) - expected) > 4:
        raise AssertionError(f"resample length sanity failed: got {len(out)}, expected ~{expected}")
    return out.astype(np.float32), method


# ----------------------------- metering ------------------------------------ #
def peak_dbfs(y):
    p = float(np.max(np.abs(y))) if y.size else 0.0
    return 20 * math.log10(p) if p > 0 else float("-inf")


def rms_dbfs(y):
    if y.size == 0:
        return float("-inf")
    r = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
    return 20 * math.log10(r) if r > 0 else float("-inf")


# ----------------------------- normalization -------------------------------- #
def normalize(y, method, target_dbfs, ceiling_dbfs):
    """Returns (y_out, gain_db). Peak method can never clip; rms/lufs apply a peak
    guard so they can't either."""
    if method == "peak":
        peak = float(np.max(np.abs(y)))
        if peak <= 0:
            return y, 0.0
        gain = (10 ** (ceiling_dbfs / 20)) / peak
        return y * gain, 20 * math.log10(gain)

    if method == "rms":
        cur = rms_dbfs(y)
        if not math.isfinite(cur):
            return y, 0.0
        gain_db = target_dbfs - cur
        return _apply_gain_with_guard(y, gain_db, ceiling_dbfs)

    if method == "lufs":
        try:
            import pyloudnorm as pyln
        except ImportError as e:
            raise RuntimeError("norm=lufs requires `pip install pyloudnorm`") from e
        meter = pyln.Meter(TARGET_SR)
        loudness = meter.integrated_loudness(y)
        if not math.isfinite(loudness):
            return y, 0.0
        gain_db = target_dbfs - loudness
        return _apply_gain_with_guard(y, gain_db, ceiling_dbfs)

    raise ValueError(f"unknown norm method {method!r}")


def _apply_gain_with_guard(y, gain_db, ceiling_dbfs):
    y2 = y * (10 ** (gain_db / 20))
    pk = float(np.max(np.abs(y2)))
    ceil = 10 ** (ceiling_dbfs / 20)
    if pk > ceil:                       # pull back so we never clip
        extra = ceil / pk
        y2 *= extra
        gain_db += 20 * math.log10(extra)
    return y2, gain_db


def trim_silence(y, top_db=40.0, frame=1024, hop=256):
    """Conservative energy-based trim of leading/trailing silence. No-op if it
    would remove everything."""
    if y.size < frame:
        return y
    n_frames = 1 + (len(y) - frame) // hop
    energies = np.empty(n_frames)
    for i in range(n_frames):
        seg = y[i * hop: i * hop + frame]
        energies[i] = np.sqrt(np.mean(seg ** 2)) if seg.size else 0.0
    peak = energies.max()
    if peak <= 0:
        return y
    thresh = peak * (10 ** (-top_db / 20))
    keep = np.where(energies >= thresh)[0]
    if keep.size == 0:
        return y
    start = keep[0] * hop
    end = min(len(y), (keep[-1]) * hop + frame)
    out = y[start:end]
    return out if out.size > 0 else y


# ----------------------------- per-clip processing ------------------------- #
def process_clip(src_path, dst_path, cfg, logger):
    """Decode -> mono -> explicit resample -> (optional trim) -> normalize ->
    assert -> write -> verify. Returns a dict of metrics/flags. Raises on hard
    failure."""
    flags = []

    data, src_sr, channels_in, backend, src_format = load_audio_native(src_path)
    if not src_sr or src_sr <= 0:
        raise ValueError(f"missing/invalid source sample rate ({src_sr}) for {src_path}")
    if data.size == 0:
        raise ValueError("decoded zero samples")

    in_peak = float(np.max(np.abs(data)))
    if in_peak >= INPUT_CLIP_PEAK:
        flags.append("input_clipped")

    y = to_mono(data).astype(np.float32)
    y = y - float(np.mean(y))                      # remove DC offset

    y, resample_method = resample(y, src_sr, cfg.target_sr)
    if src_sr < cfg.target_sr:
        flags.append("resampled_up")               # honest: can't add real >src_sr/2 content
    elif src_sr > cfg.target_sr:
        flags.append("resampled_down")

    if cfg.trim:
        y = trim_silence(y)

    if cfg.max_ref_sec and len(y) > cfg.max_ref_sec * cfg.target_sr:
        y = y[: int(cfg.max_ref_sec * cfg.target_sr)]
        flags.append("truncated_to_max")

    # Measure silence BEFORE normalization: peak/RMS norm would otherwise amplify a
    # near-silent clip up to full scale and hide that it's essentially noise.
    pre_norm_rms = rms_dbfs(y)
    if pre_norm_rms < NEAR_SILENCE_DBFS:
        flags.append("near_silence")

    y, gain_db = normalize(y, cfg.norm, cfg.rms_dbfs, cfg.peak_dbfs)

    out_rms = rms_dbfs(y)

    # Hard post-conditions -- fail loud rather than write garbage.
    assert np.isfinite(y).all(), "non-finite samples after processing"
    assert y.size > 0, "empty output"
    assert float(np.max(np.abs(y))) <= 1.0 + 1e-6, "output exceeds full scale"

    import soundfile as sf
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    sf.write(dst_path, y.astype(np.float32), samplerate=cfg.target_sr, subtype=cfg.subtype)

    info = sf.info(dst_path)                        # read back: catch a silent writer misconfig
    if info.samplerate != cfg.target_sr:
        raise AssertionError(f"written file SR {info.samplerate} != target {cfg.target_sr}")
    if info.channels != 1:
        raise AssertionError(f"written file has {info.channels} channels, expected mono")

    return {
        "source_format": src_format,
        "decode_backend": backend,
        "source_sr": src_sr,
        "target_sr": cfg.target_sr,
        "resample_method": resample_method,
        "channels_in": int(channels_in),
        "norm_method": cfg.norm,
        "gain_db": round(gain_db, 3),
        "out_duration": round(info.frames / info.samplerate, 3),
        "out_peak_dbfs": round(peak_dbfs(y), 2),
        "out_rms_dbfs": round(out_rms, 2),
        "qc_flags": flags,
    }
