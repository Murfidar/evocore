# EvoCore Checkpoint Resume Contract Design

**Date:** 2026-05-19
**Status:** Draft approved for specification
**Scope:** Long-term checkpoint and resume contract for optimizer continuation, starting with GA and deferring CMA-ES state serialization until Rust state export/import is explicit

## Summary

EvoCore should stabilize checkpointing around a clear separation of concerns:

- `OptimizationResult` is for completed-run analysis and deterministic export.
- `EventHistory` is append-only audit data.
- Checkpoints are authoritative optimizer continuation snapshots.

The long-term contract should be optimizer-agnostic, schema-versioned, and
validated before resume. The first implementation slice should support
`GeneticAlgorithmOptimizer` because its continuation state can be represented in
Python data structures today. `CMAESOptimizer` should use the same checkpoint
envelope later, but only after the Rust-backed CMA-ES state exposes stable
serializable fields such as mean, sigma, covariance, evolution paths, generation,
and any mixed-variable distribution state.

This contract intentionally avoids replaying event history to rebuild optimizer
state. Replay looks attractive, but it becomes fragile with cached records,
partial-fidelity stages, callbacks, custom operators, asynchronous tell order,
and optimizer-specific state such as CMA covariance updates. Checkpoints should
store the state needed to continue, while events remain useful for inspection and
audit.

## Current Context

Current checkpointing is GA generation-loop specific. `CheckpointCallback` writes
pickle files named `checkpoint_gen_{n}.pkl` containing a small payload:

- `population`
- `generation`
- `seed`

`GeneticAlgorithmOptimizer.resume(...)` loads that payload and continues through
the classic generation-loop path. It validates the checkpoint seed and accepts the
legacy `population` key. This path is useful, but it is not yet a general
framework contract.

Newer EvoCore surfaces already separate lifecycle and export concerns:

- `EventHistory` records ask, tell, generation, and run-stop events.
- `OptimizationTelemetry` summarizes proposed, trusted, cached, partial, and
  surrogate work.
- `ReproducibilityMetadata` records optimizer type, seed, direction, gene-space
  signature/hash, optimizer config, config hash, runtime hooks, and
  reproducibility status.
- `OptimizationResult.to_dict()` exports schema-versioned JSON-safe run output
  and prior specs explicitly avoid resume-from-result semantics.

The checkpoint contract should build on those decisions instead of blurring them.

## Product Direction

Users should be able to stop a long optimization run, persist a checkpoint, and
resume it later with confidence that EvoCore either continues the same optimizer
trajectory or fails fast with an actionable incompatibility error.

The user model should become:

```python
optimizer = GeneticAlgorithmOptimizer(space, seed=42, ...)
result = optimizer.run(evaluator, policy=policy)

# Separate continuation path:
checkpoint = optimizer.checkpoint()
optimizer.save_checkpoint("run-42.evocore-checkpoint.json", checkpoint)

resumed = GeneticAlgorithmOptimizer(space, seed=42, ...).resume_from_checkpoint(
    evaluator,
    "run-42.evocore-checkpoint.json",
)
```

The exact public method names may be refined during planning, but the contract
should preserve the conceptual split: result exports are not checkpoints, event
exports are not checkpoints, and checkpoint files are not generic analytics
payloads.

## Goals

- Define a stable checkpoint envelope with explicit schema versioning.
- Make checkpoints authoritative optimizer state snapshots for continuation.
- Support GA first without blocking on CMA-ES Rust state serialization.
- Mark the existing pickle checkpoint format as legacy GA generation-loop support.
- Make resume validation fail fast on incompatible optimizer type, gene space,
  optimizer configuration, checkpoint schema, or seed derivation contract.
- Preserve deterministic seed behavior through master seed plus counters rather
  than serializing RNG internals by default.
- Keep event history available for audit continuity without making it replayable.
- Document what is checkpointed, what is intentionally not checkpointed, and what
  compatibility EvoCore promises across versions.

## Non-Goals

- Do not implement resume-from-`OptimizationResult`.
- Do not implement `from_dict()` or `from_json()` for result exports as part of
  this contract.
- Do not reconstruct optimizer state by replaying `EventHistory`.
- Do not promise cross-version migration for arbitrary old pickle payloads.
- Do not serialize objective functions, evaluator callables, callbacks, process
  initializers, thread pools, progress bars, metrics file handles, or wall-clock
  runtime state.
- Do not add CMA-ES checkpoint/resume until the Rust-backed state has a stable
  export/import contract.
- Do not change deterministic seed derivation unless a specific compatibility bug
  is found and handled through an explicit seed-derivation version.

## Recommended Architecture

Use a two-layer checkpoint contract:

1. A shared checkpoint envelope for identity, compatibility, and metadata.
2. An optimizer-specific state payload owned by the optimizer implementation.

The shared envelope should live in a focused module such as
`evocore.results.checkpointing` or `evocore.lifecycle.checkpointing` if the
implementation needs to avoid import cycles. Optimizer-specific serializers should
live beside the optimizer code:

- `evocore/optimizers/ga/checkpointing.py`
- `evocore/optimizers/cmaes/checkpointing.py` later

The shared envelope validates identity. The optimizer-specific payload validates
state shape and owns continuation semantics.

## Checkpoint Envelope

The first stable checkpoint format should be JSON-safe by default. It may be saved
with a dedicated extension such as `.evocore-checkpoint.json`. A compressed variant
can be added later without changing the logical schema.

Suggested top-level shape:

```python
{
    "checkpoint_schema_version": 1,
    "checkpoint_kind": "optimizer_state",
    "created_by": {
        "evocore_version": "0.7.0",
        "python_version": "...",
        "platform": "...",
    },
    "optimizer": {
        "optimizer_type": "GeneticAlgorithmOptimizer",
        "optimizer_config": {...},
        "optimizer_config_hash": "...",
        "gene_space_signature": {...},
        "gene_space_hash": "...",
        "direction": "maximize",
        "seed": 42,
        "seed_derivation": {
            "algorithm": "py_derive_seed",
            "version": 1,
        },
    },
    "position": {
        "generation": 12,
        "event_index": 4,
        "n_evaluations": 480,
    },
    "state": {
        "optimizer_type": "GeneticAlgorithmOptimizer",
        "schema_version": 1,
        "payload": {...},
    },
    "audit": {
        "events": [...],
        "telemetry": {...},
        "best": {...},
    },
    "metadata": {...},
}
```

Required fields:

- `checkpoint_schema_version`
- `checkpoint_kind`
- `created_by.evocore_version`
- `optimizer.optimizer_type`
- `optimizer.optimizer_config_hash`
- `optimizer.gene_space_hash`
- `optimizer.direction`
- `optimizer.seed`
- `optimizer.seed_derivation`
- `position`
- `state.optimizer_type`
- `state.schema_version`
- `state.payload`

Optional fields:

- `created_by.python_version`
- `created_by.platform`
- `optimizer.optimizer_config`
- `optimizer.gene_space_signature`
- `audit.events`
- `audit.telemetry`
- `audit.best`
- `metadata`

The signature fields are useful for diagnostics even when hashes are the primary
compatibility check.

## What Is Checkpointable

Checkpointable state is the minimum state required to continue the same optimizer
trajectory under the same deterministic contract.

Shared checkpointable state:

- checkpoint schema version
- EvoCore version that wrote the checkpoint
- optimizer type
- direction
- master seed
- seed derivation algorithm and version
- gene-space signature and hash
- optimizer config signature and hash
- current generation or event index
- evaluation counters used for termination
- best known state-eligible candidate or solution
- telemetry counters needed to continue budget accounting
- pending batch IDs and candidate ledgers for ask/tell optimizers

GA generation-loop checkpointable state:

- current generation
- evaluated population as `Solution`-equivalent JSON-safe values
- per-solution scores and score-valid flags
- per-solution metadata required for selection or final result materialization
- elite history only if needed to preserve final result continuity
- diversity history only if needed to preserve final result continuity
- current budget counters and stop-condition state

GA ask/tell checkpointable state:

- `_event_index`
- candidate ledger keyed by candidate ID
- batch ledger keyed by batch ID
- trusted population used for reproduction
- best candidate
- pending batch IDs
- batch consumption state
- evaluation records already accepted for each candidate/stage
- telemetry counters
- current policy position when running through a managed policy loop

CMA-ES checkpointable state later:

- generation
- mean vector
- sigma
- covariance matrix or decomposition fields required by the Rust state
- evolution paths
- population size and strategy parameters that affect updates
- continuous samples for unconsumed pending batches
- mixed integer or categorical distribution state when those features exist
- candidate and batch ledgers for ask/tell lifecycle state

## What Is Intentionally Not Checkpointed

The checkpoint should not store executable or environment-bound objects.

Intentionally excluded:

- objective functions
- evaluator instances
- callback objects
- progress bar state
- metrics file handles
- thread or process pool state
- process initializers and process initializer arguments
- wall-clock timers and elapsed runtime accumulation internals
- loggers and logging handlers
- pandas DataFrames or rendered tables
- user-created closures or lambdas
- raw Python or Rust RNG object internals by default
- complete `OptimizationResult` objects
- generated build outputs or temporary cache paths

For runtime hooks, checkpoints may include signatures already used by
`ReproducibilityMetadata`, but those signatures are validation and diagnostic
metadata. Resume should require the caller to provide the evaluator and any desired
callbacks again.

## Event History Semantics

`EventHistory` should remain audit data.

The checkpoint may include event rows under `audit.events` so a resumed run can
preserve a continuous event log. Those rows should not be used as the source of
truth for reconstructing optimizer state.

The contract should state:

- Events are append-only observations.
- Events are JSON-safe audit rows.
- Events may be incomplete for state reconstruction.
- Events are not guaranteed to contain every private optimizer field.
- Events are not replayed during resume.
- Resume state comes from `state.payload`.

On resume, EvoCore can either:

- preserve existing audit events and append new events after them, or
- start a new event history with metadata linking to the checkpoint.

The recommended long-term behavior is to preserve audit continuity when event rows
are present, but treat that as a reporting feature. The resumed optimizer should
use the saved `position.event_index` and state payload to avoid candidate ID
collisions regardless of whether events are included.

## RNG And Determinism

EvoCore should continue preferring deterministic seed derivation over serialized
RNG internals.

The checkpoint should store:

- master seed
- seed derivation algorithm name
- seed derivation version
- event index or generation counter
- candidate or offspring indexes where needed in state payloads
- optimizer config hash

This aligns with existing Rust-backed deterministic helpers that derive seeds from
master seed, generation/event index, candidate index, and operation code. It also
avoids binding checkpoint compatibility to Python `random`, NumPy RNG, or Rust
`StdRng` internal representations.

Serialized RNG state should be allowed only as an explicit future extension for a
component that cannot be expressed through deterministic derivation. If that ever
happens, the checkpoint must include the RNG implementation identity and a separate
compatibility rule for that field.

## Compatibility Guarantees

EvoCore should make narrow, testable guarantees:

- Checkpoints with the same `checkpoint_schema_version` should remain readable
  within compatible patch and minor releases unless a release note explicitly
  declares a breaking checkpoint change.
- Resume must fail fast when `optimizer_type` differs.
- Resume must fail fast when `gene_space_hash` differs.
- Resume must fail fast when `optimizer_config_hash` differs, unless the caller
  explicitly opts into a documented compatibility override.
- Resume must fail fast when `seed`, `direction`, or seed derivation version
  differs.
- Unknown required fields are errors.
- Unknown optional metadata fields are preserved or ignored without changing
  optimizer behavior.
- Schema migrations must be explicit functions with tests.

The compatibility error should explain both values when practical:

```text
checkpoint gene_space_hash 'abc...' does not match optimizer gene_space_hash 'def...'.
```

Existing pickle checkpoints should be documented as legacy GA generation-loop
checkpoints. EvoCore can continue reading them best-effort, but they should not
define the forward-compatible checkpoint schema.

## Resume Validation

Resume should validate in this order:

1. File exists and can be parsed.
2. `checkpoint_kind == "optimizer_state"`.
3. Supported checkpoint schema version.
4. `optimizer.optimizer_type` matches the receiving optimizer.
5. `state.optimizer_type` matches the receiving optimizer.
6. Gene-space hash matches.
7. Optimizer config hash matches.
8. Seed, direction, and seed derivation version match.
9. Optimizer-specific state schema is supported.
10. Optimizer-specific state payload is internally consistent.

Only after validation should the optimizer mutate in-memory state.

## Public API Direction

The public API should make checkpointing explicit rather than overloading result
exports.

Possible API shape:

```python
checkpoint = optimizer.checkpoint()
optimizer.save_checkpoint(path, checkpoint)

checkpoint = GeneticAlgorithmOptimizer.load_checkpoint(path)
result = optimizer.resume_from_checkpoint(evaluator, checkpoint, policy=policy)
```

Alternative names can be chosen during implementation planning, but the API should
avoid implying that `OptimizationResult.to_dict()` is loadable or resumable.

`CheckpointCallback` should eventually write the stable checkpoint format. During
transition it may support:

```python
CheckpointCallback(path="./checkpoints", every=10, format="stable")
CheckpointCallback(path="./checkpoints", every=10, format="legacy_pickle")
```

If a format option is added, the new stable format should be the documented
default for new code. Legacy pickle support should remain explicit.

## GA First Slice

The first implementation should target `GeneticAlgorithmOptimizer` and split the
work into two levels.

First level: generation-loop parity with current behavior.

- Write stable JSON-safe checkpoint files.
- Preserve current pickle resume as legacy support.
- Validate seed, direction, gene-space hash, and optimizer config hash.
- Continue from saved population and generation.
- Add tests proving stable checkpoint resume matches uninterrupted generation-loop
  execution for deterministic objectives.

Second level: ask/tell lifecycle support.

- Snapshot candidate ledger.
- Snapshot batch ledger.
- Snapshot trusted population.
- Snapshot best candidate.
- Snapshot telemetry.
- Snapshot event index and pending batch IDs.
- Resume managed policy runs without reusing candidate IDs or losing accepted
  records.
- Add tests for partial batches, cached records, and repeated ask/tell calls.

This two-step GA path avoids mixing legacy generation-loop support with the newer
ask/tell state ledger in one large implementation.

## CMA-ES Deferral

CMA-ES should use the same shared checkpoint envelope, but the optimizer-specific
payload should wait until Rust exposes a stable state contract.

Before CMA-ES checkpoint/resume is implemented, the Rust/PyO3 state needs public
or internal export/import support for all fields required to continue a trajectory.
The contract should not approximate CMA-ES resume by saving only generation,
seed, and final samples. That would produce a new trajectory, not a continuation.

Until then, docs should state:

- GA checkpoint/resume is supported according to the stable checkpoint contract.
- CMA-ES result export and audit history are supported.
- CMA-ES checkpoint/resume is intentionally unsupported until serializable CMA
  state is available.

## Error Handling

Checkpoint failures should raise `CheckpointError`.

Recommended error categories:

- missing file
- unsupported checkpoint schema
- corrupt or non-JSON payload
- wrong checkpoint kind
- wrong optimizer type
- seed mismatch
- direction mismatch
- gene-space hash mismatch
- optimizer config hash mismatch
- unsupported optimizer-specific state schema
- internally inconsistent optimizer-specific state
- unsupported legacy pickle payload

Error messages should be actionable and include available nearby checkpoint files
for missing-path errors, matching the current behavior.

## Documentation And Changelog

Required docs updates during implementation:

- `docs/site/callbacks-checkpointing.md`: explain stable checkpoints, legacy pickle
  checkpoints, and resume examples.
- `docs/site/optimizer-telemetry.md`: clarify that telemetry may be included in
  checkpoints but remains an aggregate accounting surface.
- `docs/site/ga.md`: show GA checkpoint/resume usage.
- `docs/site/cmaes.md`: state that CMA-ES checkpoint/resume is deferred until
  serializable Rust state exists.
- `CHANGELOG.md`: note the new checkpoint/resume contract and any legacy format
  changes.

Docs must explicitly say that `OptimizationResult.to_dict()` and
`EventHistory.to_rows()` are not checkpoint files.

## Testing Strategy

Add tests before implementation.

Contract tests:

- Checkpoint envelope requires `checkpoint_schema_version`.
- Checkpoint envelope requires `checkpoint_kind == "optimizer_state"`.
- Checkpoint validation rejects optimizer type mismatch.
- Checkpoint validation rejects gene-space hash mismatch.
- Checkpoint validation rejects optimizer config hash mismatch.
- Checkpoint validation rejects seed or direction mismatch.
- Unknown optional metadata does not break loading.
- Unknown required schema version raises `CheckpointError`.

GA generation-loop tests:

- Stable checkpoint writes JSON-safe population and position.
- Resume from stable checkpoint matches uninterrupted deterministic run.
- Legacy pickle checkpoints still load on the existing path.
- Legacy pickle checkpoints are documented and tested separately from stable
  checkpoints.

GA ask/tell tests:

- Checkpoint preserves event index and candidate ID derivation.
- Checkpoint preserves trusted population and best candidate.
- Checkpoint preserves pending batches.
- Resume rejects duplicate state-update records already accepted before checkpoint.
- Resume handles cached records without incrementing fresh full-evaluation budget.
- Resume appends new events after restored event history when audit events are
  included.

CMA-ES tests later:

- Export/import of Rust state preserves ask/tell trajectory.
- Resume after a consumed batch matches uninterrupted CMA-ES trajectory.
- Resume rejects partial state payloads missing covariance or evolution path data.

Property tests:

- JSON round-trip for checkpoint envelope with generated JSON-safe metadata.
- Stable hash comparisons are deterministic for repeated exports.

## Verification

Implementation should run targeted tests first, then broader checks according to
the touched surface.

Expected targeted checks for GA checkpoint work:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_checkpointing.py tests/unit/test_ga_engine.py tests/unit/test_ga_ask_tell_vnext.py -v
.\.venv\Scripts\python.exe -m ruff format --check
.\.venv\Scripts\python.exe -m ruff check
git diff --check
```

If Rust CMA-ES state serialization is added later, also run:

```powershell
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
.\.venv\Scripts\python.exe -m maturin develop --release
.\.venv\Scripts\python.exe -m pytest tests/unit/ tests/integration/ -v
```

## Rollout

Recommended implementation slices:

1. Add the shared checkpoint envelope dataclass and validation helpers.
2. Add stable save/load helpers for JSON-safe checkpoint files.
3. Add GA generation-loop stable checkpoint write/resume.
4. Preserve and document legacy pickle checkpoint loading.
5. Update `CheckpointCallback` to write stable checkpoints for new usage.
6. Add GA ask/tell checkpoint state snapshot and restore.
7. Refresh docs and changelog.
8. Defer CMA-ES until Rust state export/import has its own design and tests.

## Acceptance Criteria

- EvoCore has a documented checkpoint contract distinct from result exports and
  event audit exports.
- Stable checkpoints are schema-versioned and JSON-safe.
- GA can resume from stable checkpoints with deterministic trajectory continuity.
- Resume validates optimizer type, seed, direction, gene-space hash, optimizer
  config hash, and seed derivation version before mutating state.
- Event history is documented as audit data, not replayable state.
- Existing pickle checkpoints remain legacy GA support where practical.
- CMA-ES checkpoint/resume is explicitly deferred until Rust state serialization
  is designed.
- Docs and changelog explain the supported format, unsupported formats, and
  compatibility guarantees.

## Deferred Follow-Ups

- CMA-ES Rust state export/import design.
- Compressed checkpoint files.
- Checkpoint migration registry for future schema versions.
- Optional user-provided compatibility overrides for safe config changes.
- External checkpoint stores or artifact registry integration.
- Resume across distributed workers or interrupted process pools.
