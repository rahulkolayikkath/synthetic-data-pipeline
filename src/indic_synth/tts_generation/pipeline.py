"""tts_generation.pipeline — orchestrates speaker-conditioned synthesis (§4.5).

Flow:
    - pair each validated sentence with a reference speaker in the SAME language
      (no cross-lingual conditioning in the main run)
    - chunk any input over the model's length limit at sentence boundaries
    - run IndicF5 with the model card's default inference parameters
    - write the output at 24 kHz with the chosen format + loudness normalization

Reads prepared_manifest.jsonl (§4.3) + the validated sentence pool (§4.4); emits
the synthesized utterances + output manifest for §4.6. Throughput choices
(batching/precision) are reported as utterances/min. Resumable / idempotent per
utterance.
"""
