# CMA-ES Ask/Tell Checkpoints Design

**Date:** 2026-05-20
**Status:** Draft approved for specification
**Scope:** Stable CMA-ES ask/tell checkpoint and resume using the shared checkpoint envelope and Rust-backed CMA-ES state snapshots

## Summary

EvoCore should add stable ask/tell checkpoint-resume support for
`CMAESOptimizer`.

This is the next layer after Rust-backed `PyCMAESState` snapshots. The Rust
state snapshot preserves CMA-ES adaptation state, while the optimizer checkpoint
preserves the Python ask/tell runtime ledgers needed to continue external
evaluation workflows.

The user model should match GA ask/tell checkpoints:

```python
optimizer = CMAESOptimizer(space, population_size=8, seed=42)
candidates = optimizer.ask()

checkpoint = optimizer.ask_tell_checkpoint()
optimizer.save_checkpoint("cmaes.evocore-checkpoint.json", checkpoint)

restored = CMAESOptimizer(space, population_size=8, seed=42)
summary = restored.resume_ask_tell_checkpoint("cmaes.evocore-checkpoint.json")
```

After resume, callers should be able to submit records for pending candidates,
complete partially evaluated batches, and continue with deterministic future
`ask()` calls.

## Current Context

GA already supports stable ask/tell checkpoints through the shared
`CheckpointSnapshot` envelope. It serializes candidate ledgers, batch ledgers,
telemetry, event history, best-candidate identity, and GA-specific trusted
population state.

CMA-ES has the same Python ask/tell ledger concepts:

- `_event_index`
- `_candidates_by_id`
- `_batches_by_id`
- `best_candidate`
- `vnext_telemetry`
- `events`

CMA-ES also owns Rust-backed adaptation state through
`self._state: _core.PyCMAESState | None`. That state is now exportable and
restorable with `PyCMAESState.to_dict()` and `PyCMAESState.from_dict(...)`,
including the lazy eigendecomposition cache needed for exact continuation.

The current `CMAESOptimizer.run()` path uses a local Rust state rather than
`self._state`, so this design is intentionally limited to manual ask/tell
workflows.

## Goals

- Add stable CMA-ES ask/tell checkpoints using the existing
  `CheckpointSnapshot` envelope.
- Preserve Rust CMA-ES state through `PyCMAESState.to_dict()`.
- Restore Rust CMA-ES state through `PyCMAESState.from_dict(...)`.
- Preserve pending batches and partial tells as valid checkpoint state.
- Preserve continuous samples by candidate id so completed batches can still
  update CMA-ES state after resume.
- Restore telemetry, event history, event index, candidate ledgers, batch
  ledgers, and best-candidate state.
- Validate optimizer identity before mutating the receiving optimizer.
- Return `OptimizerStateSummary` after resume.
- Prove that the next `ask()` after resume matches uninterrupted execution.

## Non-Goals

- Do not make `CMAESOptimizer.run()` resumable.
- Do not support policy-driven mid-loop `run(evaluator, policy=...)` resume.
- Do not replay events to rebuild optimizer state.
- Do not refactor GA and CMA-ES into a shared checkpoint base in this slice.
- Do not checkpoint evaluator callables, callbacks, thread pools, process pools,
  wall-clock timers, open files, or external job handles.
- Do not introduce checkpoint schema v2 unless implementation discovers a true
  incompatibility with the existing envelope.

## Public API

`CMAESOptimizer` should expose the same manual checkpoint API shape as GA:

```python
snapshot = optimizer.ask_tell_checkpoint(metadata={...})
optimizer.save_checkpoint(path, snapshot)

restored = CMAESOptimizer(...)
summary = restored.resume_ask_tell_checkpoint(path)
```

`ask_tell_checkpoint(...)` should return a `CheckpointSnapshot`.

`resume_ask_tell_checkpoint(...)` should accept a file path or an already loaded
checkpoint mapping. It should validate identity, restore state, and return
`state_summary()`.

`save_checkpoint(...)` and `load_checkpoint(...)` should be available as static
methods on `CMAESOptimizer` for parity with GA, delegating to the shared result
checkpoint helpers.

## Checkpoint Payload

Use the existing stable checkpoint envelope. The outer snapshot should use:

- `optimizer_type = "CMAESOptimizer"`
- current `config_signature()`
- current `config_hash()`
- `gene_space.signature()`
- `gene_space.hash()`
- `direction`
- `seed`

The `position` payload should describe ask/tell continuation:

```python
{
    "mode": "ask_tell",
    "event_index": self._event_index,
    "pending_batch_ids": list(self._pending_batch_ids()),
    "best_candidate_id": best_candidate_id,
}
```

The `state` envelope should contain:

```python
{
    "optimizer_type": "CMAESOptimizer",
    "schema_version": 1,
    "payload": {
        "state_kind": "cmaes_ask_tell",
        "event_index": self._event_index,
        "cmaes_state": self._ensure_state().to_dict(),
        "candidates_by_id": {
            candidate_id: candidate_to_checkpoint(candidate),
        },
        "batches_by_id": {
            batch_id: batch_to_checkpoint(batch),
        },
        "best_candidate_id": best_candidate_id,
        "telemetry": telemetry_to_checkpoint(self.vnext_telemetry),
        "events": event_history_to_checkpoint(self.events),
    },
}
```

The `audit` payload should also include serialized events and telemetry for
inspection continuity, matching the GA ask/tell checkpoint shape.

`batch_to_checkpoint(...)` already serializes `continuous_samples_by_id`. That is
critical for CMA-ES because `_consume_complete_batch(...)` needs the original
continuous samples, in ask order, when a pending batch receives its remaining
state-update records after resume.

## Restore Contract

Restore should load a checkpoint from a path or mapping, then validate identity
before mutating the optimizer.

Identity validation should require:

- `optimizer_type = "CMAESOptimizer"`
- matching gene-space hash
- matching optimizer config hash
- matching seed
- matching direction

State validation should require:

- `state.schema_version == 1`
- `state.payload.state_kind == "cmaes_ask_tell"`
- `state.payload.cmaes_state` is present and accepted by
  `PyCMAESState.from_dict(...)`

After validation, restore should set:

- `_state`
- `_event_index`
- `_candidates_by_id`
- `_batches_by_id`
- `best_candidate`
- `vnext_telemetry`
- `events`

Integrity checks should reject malformed ledgers:

- candidate map keys must match each restored candidate's `candidate_id`.
- batch map keys must match each restored batch's `batch_id`.
- every batch candidate id must exist in `candidates_by_id`.
- every non-consumed batch must retain continuous samples for every candidate in
  the batch.
- `best_candidate_id`, when present, must reference a known candidate.
- consumed batches may be restored, but they must not be re-consumed.

Resume should return `state_summary()` so callers can inspect pending batches and
best state immediately.

## Determinism Contract

The authoritative continuation state is the checkpoint payload, not event
history.

A restored CMA-ES ask/tell checkpoint must continue like an uninterrupted
optimizer when given the same remaining records. In particular:

1. If a checkpoint is taken immediately after `ask()`, a restored optimizer
   should accept the same trusted records for that pending batch.
2. If a checkpoint is taken after a partial `tell()`, a restored optimizer
   should accept the remaining records and consume the batch exactly once.
3. If a checkpoint is taken after a consumed batch, a restored optimizer's next
   `ask()` should match the uninterrupted optimizer's next `ask()`.

Events remain audit data. They should be restored for continuity, but they
should never be replayed to rebuild candidates, batches, telemetry, best state,
or Rust CMA-ES state.

## Implementation Shape

Add a focused CMA-ES checkpointing module:

```text
evocore/optimizers/cmaes/checkpointing.py
```

It should define:

- `CMAES_ASK_TELL_STATE_KIND = "cmaes_ask_tell"`
- `CMAES_CHECKPOINT_STATE_SCHEMA_VERSION = 1`
- `CMAESCheckpointingMixin`
- `ask_tell_checkpoint(...)`
- `resume_ask_tell_checkpoint(...)`
- `_validate_stable_checkpoint_identity(...)`
- `_ask_tell_payload_from_checkpoint(...)`
- `_restore_ask_tell_state(...)`
- static `save_checkpoint(...)`
- static `load_checkpoint(...)`

`CMAESOptimizer` should inherit `CMAESCheckpointingMixin` in addition to
`CMAESAskTellMixin`. The generation-loop `run()` implementation should remain
unchanged.

`evocore/optimizers/cmaes/__init__.py` should export
`CMAESCheckpointingMixin`, matching the existing package export style for
`CMAESAskTellMixin`.

## Test Plan

Add a focused test module:

```text
tests/unit/test_cmaes_ask_tell_checkpointing.py
```

Coverage should include:

- checkpoint immediately after `ask()` contains pending batch state and
  `cmaes_state`.
- resume after `ask()` accepts trusted records for the pending batch.
- checkpoint after partial `tell()` restores accepted records and accepts the
  missing records.
- checkpoint after a full consumed batch restores Rust state and the next
  `ask()` matches uninterrupted execution.
- resume rejects optimizer config mismatch.
- resume rejects seed mismatch.
- resume rejects direction mismatch.
- resume rejects wrong state kind.
- resume rejects unknown `best_candidate_id`.
- resume rejects a batch referencing an unknown candidate.
- resume rejects missing or malformed `cmaes_state`.
- restored events remain audit data and are not replayed.

Existing GA checkpoint tests should remain unchanged unless a shared lifecycle
serializer bug is discovered.

## Documentation And Changelog

Update `docs/site/cmaes.md` with a CMA-ES ask/tell checkpoint example.

Update `docs/site/callbacks-checkpointing.md` so CMA-ES ask/tell checkpoints are
no longer listed as unsupported. The docs should still state that CMA-ES
generation-loop and policy-run resume remain unsupported.

Update `CHANGELOG.md` because this adds user-visible checkpoint capability.

Do not commit generated `site/` output.

## Verification

Implementation should run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_cmaes_ask_tell_checkpointing.py -v
.\.venv\Scripts\python.exe -m pytest tests/unit/test_cmaes_ask_tell_vnext.py tests/unit/test_ask_tell_checkpointing.py tests/unit/test_checkpointing.py -v
.\.venv\Scripts\python.exe -m pytest tests/unit/ tests/integration/ -v
.\.venv\Scripts\python.exe -m ruff format --check
.\.venv\Scripts\python.exe -m ruff check
.\.venv\Scripts\python.exe -m mkdocs build
```

Because this slice should be Python-only and uses an existing Rust state
snapshot API, Rust rebuild is not required unless implementation touches
`src/`, `evocore/_core.pyi`, `pyproject.toml`, or Rust-facing behavior.
