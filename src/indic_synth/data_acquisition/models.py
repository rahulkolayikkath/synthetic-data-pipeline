"""data_acquisition.models — cheap gated-parquet access (§4.2).

Wrappers around reading the gated Kathbath valid-*.parquet files such that the
hard cost boundary is enforced:
    - Stage A: column projection that reads ONLY speaker_id, gender, lang,
      duration, text — the heavy audio_filepath column is never fetched
      (parquet stores columns separately, so audio chunks aren't transferred).
    - Stage B: read only the row groups containing the chosen rows to pull
      ref_audio/*.wav.

Uses split=`valid` (~0.5 GB/lang vs ~16 GB train) since Kathbath here is a voice
bank, not ASR train/eval data. Reads use exponential-backoff retries.
"""
