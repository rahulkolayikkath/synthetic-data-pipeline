"""audio_engineering.models — resampling & loudness backends (§4.3).

Thin wrappers over the DSP backends used by the prepare pipeline:
    - Resampler: explicit soxr (HQ) resampling; asserts the output length
      matches the rate ratio. Kathbath is ~16 kHz so this usually upsamples
      (flagged `resampled_up` — it cannot create real content above the source
      Nyquist; an honest limitation, not a defect).
    - Mono downmix: channel-average if a clip is ever stereo (channels_in recorded).
    - Normalization: default peak-to -1 dBFS (deterministic, cannot clip by
      construction); `rms` (target dBFS) and `lufs` (ITU-R BS.1770, pyloudnorm)
      available, both with a peak guard.
"""
