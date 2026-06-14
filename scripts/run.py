"""run — CLI entrypoint for the synthetic speech data pipeline.

Thin wrapper around `indic_synth.cli.main`, kept for the documented Colab flow:

    Run all stages in order: python scripts/run.py --config config.yaml
    Run selected stages:     python scripts/run.py --config config.yaml --stages tts_generation quality_control
    Validate config only:    python scripts/run.py --config config.yaml --validate-only

The same orchestration is also installed as the `indic-synth` command after
`pip install -e .` (see [project.scripts] in pyproject.toml); both call
indic_synth.cli.main. The config is validated against the schema in
indic_synth.common.config before any stage runs.

    data_acquisition    (§4.2) -> reference_manifest.jsonl
    audio_engineering   (§4.3) -> prepared_manifest.jsonl
    sentence_generation (§4.4) -> sentences.jsonl
    tts_generation      (§4.5) -> tts_manifest.jsonl
    quality_control     (§4.6) -> dataset_manifest.jsonl   (final, packaged + QC'd)

Each stage is independently resumable, so a dropped session is restarted by simply
re-running this command. Writes a top-level pipeline_summary.json.
"""
from __future__ import annotations

import os
import sys

# Allow running straight from a clone (python scripts/run.py ...) without pip install.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from indic_synth.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
