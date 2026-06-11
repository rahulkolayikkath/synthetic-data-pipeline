"""data_acquisition.pipeline — orchestrates the two-stage acquisition (§4.2).

Stage A (metadata, cheap):
    profile every speaker from projected columns -> catalog.parquet, then run a
    seeded stratified sampler that groups by (lang, speaker_id) and picks a
    balanced male/female quota per language from the whole pool (not the first N
    rows), filtering clips to a sane reference-duration window -> selection_manifest.jsonl.

--- cost boundary ---

Stage B (audio, selective):
    read only the row groups holding the chosen rows, write ref_audio/<ref_id>.<ext>
    verbatim and append reference_manifest.jsonl, skipping any ref_id already
    downloaded and present on disk (idempotent resume).

Reproducible: catalog sorted into a stable order, sampler fully seeded.
Partial-failure isolation: a bad row / unreadable row group is logged, marked
`failed`, and the run continues. Emits run_summary.json.

Diversity caveat (documented honestly): the HF parquet has no district/state
column, so regional balance can't be guaranteed; distinct speaker_ids are used
as a proxy.
"""
