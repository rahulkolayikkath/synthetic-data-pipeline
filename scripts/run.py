"""run — CLI entrypoint for the synthetic speech data pipeline.

Usage (the Colab Cell 3):
    python scripts/run.py --config config.yaml

Drives the stages end-to-end in order, each reading the previous stage's manifest:
    1. data_acquisition    (§4.2)  -> reference_manifest.jsonl
    2. audio_engineering   (§4.3)  -> prepared_manifest.jsonl
    3. sentence_generation (§4.4)  -> validated sentence pool
    4. tts_generation      (§4.5)  -> synthesized utterances + output manifest
    5. quality_control     (§4.6)  -> validated, packaged mini-dataset

Reproducible (seeded), fault-tolerant (checkpointed / resumable), and
observable (structured logs + per-stage summary JSON), per common (§4.1).
"""
