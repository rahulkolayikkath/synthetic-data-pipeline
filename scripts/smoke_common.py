"""smoke_common.py — Step 0 smoke test for indic_synth.common (§4.1).

Exercises the cross-cutting infra with no heavy deps:
  - JSONL manifest round-trip (write / read / append)
  - checkpoint.load_done resume logic (with status + on-disk audio guards)
  - config.yaml loads with global + per-stage subsections
  - set_determinism makes RNG draws reproducible across two calls

Run:  python scripts/smoke_common.py
"""
from __future__ import annotations

import os
import random
import sys
import tempfile

# Make `import indic_synth` work before `pip install -e .`
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from indic_synth.common import checkpoint, config, manifest  # noqa: E402


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def test_manifest_roundtrip(tmp: str) -> None:
    print("[manifest] round-trip")
    path = os.path.join(tmp, "m.jsonl")
    rows = [{"ref_id": "a", "x": 1}, {"ref_id": "b", "x": 2, "lang": "हिन्दी"}]
    manifest.write_jsonl(path, rows)
    check(manifest.read_jsonl(path) == rows, "read == written (unicode preserved)")
    manifest.append_jsonl(path, {"ref_id": "c", "x": 3})
    check(len(manifest.read_jsonl(path)) == 3, "append adds one row")
    check(list(manifest.iter_jsonl(path))[2]["ref_id"] == "c", "iter_jsonl streams in order")
    check(manifest.read_jsonl(os.path.join(tmp, "missing.jsonl")) == [], "missing file -> []")
    check("audio_filepath" in manifest.UTTERANCE_FIELDS, "UTTERANCE_FIELDS exposed")


def test_checkpoint(tmp: str) -> None:
    print("[checkpoint] load_done")
    path = os.path.join(tmp, "ref.jsonl")
    wav = os.path.join(tmp, "a.wav")
    open(wav, "wb").close()  # exists on disk
    manifest.write_jsonl(path, [
        {"ref_id": "a", "status": "downloaded", "local_audio_path": "a.wav"},
        {"ref_id": "b", "status": "downloaded", "local_audio_path": "gone.wav"},
        {"ref_id": "c", "status": "failed"},
    ])
    done = checkpoint.load_done(path, key="ref_id", status="downloaded",
                               audio_field="local_audio_path", out_dir=tmp)
    check(done == {"a"}, "only on-disk + downloaded counts as done (a; not b/c)")
    pend = checkpoint.pending([{"ref_id": "a"}, {"ref_id": "z"}], done)
    check([r["ref_id"] for r in pend] == ["z"], "pending filters out done ids")


def test_config() -> None:
    print("[config] load_config + stage()")
    cfg = config.load_config(os.path.join(ROOT, "config.yaml"))
    check(cfg.seed == 1234, "global seed parsed")
    check(cfg.out_dir == "out", "global out_dir parsed")
    for name in ["data_acquisition", "audio_engineering", "sentence_generation",
                 "tts_generation", "quality_control"]:
        sub = cfg.stage(name)
        check(sub.get("seed") == 1234 and "out_dir" in sub, f"{name}: globals injected")
    check(cfg.stage("data_acquisition")["speakers_per_language"] == 10,
          "stage subsection values parsed")


def test_determinism() -> None:
    print("[config] set_determinism reproducibility")
    config.set_determinism(7)
    a = [random.random() for _ in range(5)]
    config.set_determinism(7)
    b = [random.random() for _ in range(5)]
    check(a == b, "same seed -> identical RNG draws")
    check(config.derive_seed(1234, "hi", "x") == config.derive_seed(1234, "hi", "x"),
          "derive_seed is deterministic")
    check(config.derive_seed(1234, "hi", "x") != config.derive_seed(1234, "ml", "x"),
          "derive_seed varies by parts")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        test_manifest_roundtrip(tmp)
        test_checkpoint(tmp)
        test_config()
        test_determinism()
    print("\nSMOKE OK: common (§4.1)")


if __name__ == "__main__":
    main()
