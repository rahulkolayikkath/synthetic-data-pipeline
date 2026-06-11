"""common.config — config loading, seeds and determinism (§4.1).

Loads config.yaml into a typed config object and pins everything needed to make
a run reproducible: global seeds, per-call derived seeds, and the deterministic
flags (torch / numpy / cudnn) so headline numbers can be reproduced.

Also the home for environment pinning helpers and the seed-derivation scheme
that gives reproducibility *and* diversity across LLM/TTS calls.
"""
