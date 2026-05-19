# Ask/Tell Checkpoints Design

## Summary

Ask/tell checkpoints extend EvoCore's stable checkpoint envelope to external
evaluation workflows. They preserve optimizer runtime ledgers so a process can
restart after candidates have been proposed, after some evaluation records have
arrived, or after a full state update.

The first implementation target is GA ask/tell. CMA-ES should use the same
envelope shape later, but stable CMA-ES resume requires explicit Rust/PyO3 state
export and import before it can be promised.

## Goals

- Save and restore GA ask/tell runtime state using the existing stable JSON
  checkpoint envelope.
- Preserve pending batches and partial tells as valid checkpoint state.
- Resume by restoring structured ledgers directly, not by replaying event
  history.
- Keep normal `ask()` and `tell()` semantics after resume, including duplicate
  and conflict rejection.
- Validate optimizer identity through optimizer type, gene-space hash, config
  hash, seed, direction, and schema version.
- Return a useful `OptimizerStateSummary` after resume so callers can inspect
  pending work immediately.

## Non-Goals

- Do not implement mid-loop policy-driven `run(...)` resume in this step.
- Do not implement CMA-ES stable resume until Rust/PyO3 CMA state can be
  exported and restored.
- Do not treat event history as replay input.
- Do not checkpoint evaluator callables, callbacks, thread/process pools,
  wall-clock timers, or file handles.
- Do not add a new top-level checkpoint format; reuse the shared
  `CheckpointSnapshot` envelope.

## Public API

The public surface should stay small and optimizer-oriented:

```python
snapshot = optimizer.ask_tell_checkpoint(metadata={...})
optimizer.save_checkpoint(path, snapshot)

restored = GeneticAlgorithmOptimizer(...)
summary = restored.resume_ask_tell_checkpoint(path)
```

`ask_tell_checkpoint(...)` returns a `CheckpointSnapshot`. The existing stable
`save_checkpoint(...)` and `load_checkpoint(...)` helpers remain the file I/O
boundary.

`resume_ask_tell_checkpoint(...)` accepts a path or already-loaded checkpoint
mapping. It validates identity, restores runtime state, and returns
`OptimizerStateSummary`.

## Checkpoint Payload

The shared envelope remains:

- `checkpoint_schema_version`
- `checkpoint_kind`
- `created_by`
- `optimizer`
- `position`
- `state`
- `audit`
- `metadata`

For GA ask/tell, `state` should contain:

- `optimizer_type = "GeneticAlgorithmOptimizer"`
- `schema_version = 1`
- `state_kind = "ga_ask_tell"`
- `payload`

The `payload` should contain:

- `event_index`: next ask event index.
- `candidates_by_id`: candidate ledger keyed by candidate ID.
- `batches_by_id`: batch ledger keyed by batch ID.
- `trusted_candidate_ids`: ordered trusted population candidate IDs.
- `best_candidate_id`: current best candidate ID, or `None`.
- `telemetry`: serialized optimization telemetry.
- `events`: event-history rows for audit continuity.

`position` should describe where the optimizer can continue:

- `mode = "ask_tell"`
- `event_index`
- `pending_batch_ids`
- `best_candidate_id`

## Serialization Units

Candidate serialization should include:

- candidate ID
- genes
- params
- batch ID
- origin
- parents
- event index
- generation
- stage
- status
- confidence
- cost
- score observations
- metadata

Evaluation record serialization should include:

- candidate ID
- batch ID
- score
- confidence
- stage
- cost
- metrics
- metadata

Batch serialization should include:

- batch ID
- candidate ID order
- accepted records
- consumed flag
- optimizer-specific extras such as CMA-ES continuous samples

GA trusted state should store candidate IDs instead of duplicating candidate
payloads. Restoring trusted state should resolve those IDs through the candidate
ledger.

Best state should store `best_candidate_id` and resolve it through the same
candidate ledger.

Telemetry should round-trip from a stable public dictionary shape. If the
current telemetry object lacks a restore helper, add one rather than rebuilding
telemetry from events.

Events should round-trip for diagnostics and audit continuity only.

## Resume Semantics

Resume should follow this order:

1. Load and validate the shared checkpoint envelope.
2. Validate optimizer identity.
3. Validate ask/tell state kind, schema version, and optimizer type.
4. Reconstruct candidates.
5. Reconstruct batches and their accepted records.
6. Reconstruct telemetry.
7. Reconstruct event history.
8. Resolve trusted and best candidate references.
9. Set `_event_index` to the saved next ask index.
10. Return `state_summary()`.

After resume:

- `ask()` should produce the same next batch ID and candidate IDs as
  uninterrupted execution.
- `tell()` should accept missing records for restored pending batches.
- `tell()` should reject duplicates and conflicts exactly as before resume.
- Partial pending batches should remain pending until enough state-update
  records arrive.
- Fully consumed GA batches may remain in the audit ledger. CMA-ES consumed
  batches must continue rejecting additional state-update records once CMA-ES
  resume exists.

## Event History

Event history is append-only audit data. It helps users diagnose what happened
before a checkpoint, and it should continue appending after resume.

Event history must not be replayed to rebuild optimizer state. Replaying would
make resume behavior depend on historical event interpretation and would blur
the line between audit data and state.

## Policy-Driven Run Resume

Policy-driven `run(evaluator, policy=...)` resume is intentionally deferred.

The current ask/tell runtime has candidate, batch, event, telemetry, best, and
trusted ledgers. A mid-loop policy run also needs explicit scheduler/run-position
state: current stage, assigned candidates, promoted candidates, evaluation
counts, and loop stop conditions.

This design should not promise seamless policy-run resume until that scheduler
state is represented as a serializable contract.

## Error Handling

Resume should reject:

- unsupported checkpoint schema versions
- wrong checkpoint kind
- wrong optimizer type
- mismatched gene-space hash
- mismatched optimizer-config hash
- mismatched seed
- mismatched direction
- unknown state kind
- malformed candidates, records, batches, telemetry, or events
- trusted candidate IDs that are missing from the candidate ledger
- best candidate IDs that are missing from the candidate ledger
- batch records whose candidate IDs are not part of the batch
- duplicate records for the same candidate and stage
- duplicate state-update records for the same candidate in one batch

Incomplete pending batches are valid state and should not raise during resume.

## Testing

The first implementation should add tests for:

- checkpoint after `ask()`, resume, then `tell()` all records.
- checkpoint after partial `tell()`, resume, then submit missing records.
- checkpoint after full trusted `tell()`, resume, then next `ask()` matches an
  uninterrupted optimizer.
- duplicate `tell()` after resume is rejected.
- config, gene-space, seed, and direction mismatches are rejected.
- pending batch IDs survive resume.
- `state_summary()` reports best candidate, event index, trusted count,
  telemetry, and pending batches after resume.
- event history is restored for audit continuity, without relying on replay.
- checkpoint JSON round-trips deterministically.
- CMA-ES stable ask/tell resume is explicitly unsupported until state export and
  import exist.

## Implementation Order

1. Add serialization helpers for `Candidate`, `ScoreObservation`,
   `EvaluationRecord`, and `CandidateBatch`.
2. Add telemetry and event-history restore helpers if they do not already exist.
3. Add GA ask/tell checkpoint snapshot creation.
4. Add GA ask/tell checkpoint resume.
5. Add public exports or docs only where they clarify the stable contract.
6. Add docs, changelog, and focused tests.

## Compatibility

This is an additive contract on top of checkpoint schema version 1. It should not
change generation-loop checkpoint semantics or legacy pickle compatibility.

Within `state_kind = "ga_ask_tell"` and `schema_version = 1`, compatible EvoCore
patch releases should continue to load checkpoints or fail with explicit
diagnostics when a checkpoint is unsupported.
