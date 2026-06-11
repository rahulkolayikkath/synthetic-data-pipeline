"""common.checkpoint — idempotent resume & partial-failure handling (§4.1).

Fault-tolerance primitives so a dropped Colab session does not cost the run:
    - checkpoint the validated/produced work to disk as it completes
    - idempotent restart: skip items already done and present on disk
    - partial-failure isolation: mark a bad item `failed` and continue
    - exponential-backoff retries for HF / IO reads

Used by every stage that appends to a manifest and must be resumable.
"""
