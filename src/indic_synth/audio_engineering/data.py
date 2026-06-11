"""audio_engineering.data — prepared-manifest schema for the audio stage (§4.3).

Extends the input reference_manifest with the fields this stage records per clip:
    prepared_audio_path, source_sr, target_sr, resample_method, channels_in,
    gain_db, out_duration, out_peak_dbfs, out_rms_dbfs, qc_flags

Outputs owned here:
    prepared_audio/<ref_id>.wav
    prepared_manifest.jsonl   — input to Stage 4.5 (tts_generation)
    prepare_summary.json
"""
