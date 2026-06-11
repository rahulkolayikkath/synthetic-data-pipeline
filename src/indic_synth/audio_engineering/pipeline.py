"""audio_engineering.pipeline — decode / resample / normalize per clip (§4.3).

Per-clip flow, designed to make silent resampling bugs impossible:
    1. Decode at the file's NATIVE rate — the loader is never asked to resample.
    2. Read the real header rate; refuse if it's missing/zero.
    3. Resample EXPLICITLY (soxr HQ) and assert output length == rate ratio.
    4. Assert post-conditions: out_sr == 24000, mono, finite, peak <= 1.0.
    5. Read the written WAV back and check its header rate (catches a silent
       writer misconfig).
    6. Record source/target rate, backend, gain and QC flags per clip so a wrong
       rate or near-silent clip surfaces in the manifest instead of poisoning the
       TTS run.

Near-silence is measured BEFORE normalization, since peak/RMS gain would
otherwise amplify a silent clip to full scale and mask it.

Reads reference_manifest.jsonl; writes prepared_audio/ + prepared_manifest.jsonl
+ prepare_summary.json. Resumable / idempotent per ref_id.
"""
