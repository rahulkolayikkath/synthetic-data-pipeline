"""data_acquisition.data — schemas & IO for the acquisition stage (§4.2).

Defines the records and writers for this stage's outputs:
    catalog.parquet            — every utterance's metadata + source_file, row_index
    selection_manifest.jsonl   — the chosen rows (deterministic given catalog + seed)
    reference_manifest.jsonl   — per clip: local_audio_path, ref_text, speaker,
                                 gender, duration, status
    run_summary.json           — counts, per-language gender split, row groups
                                 touched, elapsed

Kathbath per-language source schema profiled here:
    fname, text, audio_filepath (audio), lang, duration, gender, speaker_id

reference_manifest.jsonl is the input to Stage 4.3 (audio_engineering). Audio
bytes are written verbatim with the source extension — no decode happens here.
"""
