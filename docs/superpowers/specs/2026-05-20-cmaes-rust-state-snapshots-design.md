# EvoCore CMA-ES Rust State Snapshot Design

**Date:** 2026-05-20
**Status:** Draft approved for specification
**Scope:** Stable Rust-backed CMA-ES state snapshots for deterministic continuation primitives, before full optimizer checkpoint/resume integration

## Summary

EvoCore should add a stable snapshot contract for the Rust-backed
`PyCMAESState` before wiring CMA-ES into the broader optimizer checkpoint
framework.

The immediate goal is to make the CMA-ES adaptation state exportable,
JSON-safe, validated on restore, and deterministic after round trip:

```python
snapshot = state.to_dict()
restored = PyCMAESState.from_dict(snapshot)

assert restored.ask(seed, restored.generation) == state.ask(seed, state.generation)
```

This design intentionally treats the Rust state snapshot as a foundation layer.
It does not yet make `CMAESOptimizer` ask/tell checkpoints resumable, and it
does not make `CMAESOptimizer.run()` resumable. Those integrations should depend
on this primitive after it is tested and stable.

## Current Context

`CMAESOptimizer` owns Python-side optimizer lifecycle state, including candidate
ledgers, pending batches, telemetry, events, and best-candidate tracking. Its
ask/tell path uses `self._state: _core.PyCMAESState | None`, while the
generation-loop `run()` path currently creates a local Rust `PyCMAESState`.

The Rust `CMAESState` owns the adaptation state that determines future samples:

- dimensionality
- population size
- mean vector
- sigma
- covariance matrix
- evolution paths
- generation counter
- bounds
- eigendecomposition refresh cadence
- pending eigen update count

The Rust implementation also maintains derived strategy constants and an eigen
cache. Those are not all equally checkpointable. Strategy constants can be
recomputed from stable inputs such as dimension and population size. The eigen
cache is derived from covariance and can be invalidated after restore.

Previous checkpoint designs deliberately kept event history as audit data rather
than replay input. CMA-ES should follow that contract. Resume should restore
state, not rebuild state by replaying events.

## Goals

The snapshot primitive should support deterministic continuation of a
`PyCMAESState` instance after export and restore.

The snapshot should preserve:

- current mean vector
- current sigma
- covariance matrix
- evolution paths `pc` and `ps`
- generation counter
- population size, stored as `lambda`
- bounds
- eigendecomposition interval
- pending eigen update count

The restored state should produce the same next sample batch as the original
state when called with the same `master_seed` and generation.

## Non-Goals

This implementation slice should not implement full `CMAESOptimizer`
checkpoint/resume.

It should not make `CMAESOptimizer.run()` resumable. That path owns a local Rust
state today, so run-level resume requires a separate design and refactor.

It should not checkpoint Python optimizer ledgers, candidate records, telemetry,
event history, callbacks, best candidate state, or pending batches. Those belong
to optimizer-level checkpointing.

It should not checkpoint the eigen cache. The cache is derived from covariance
and should be rebuilt lazily after restore.

## Snapshot Schema V1

`PyCMAESState.to_dict()` should return a JSON-safe Python dictionary:

```python
{
    "schema_version": 1,
    "optimizer_type": "cmaes",
    "state": {
        "n": 4,
        "lambda": 8,
        "generation": 12,
        "mean": [0.1, 0.2, 0.3, 0.4],
        "sigma": 0.42,
        "cov": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "pc": [0.0, 0.0, 0.0, 0.0],
        "ps": [0.0, 0.0, 0.0, 0.0],
        "bounds": [[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]],
        "eigendecomp_interval": 4,
        "pending_eigen_updates": 2,
    },
}
```

`cov` should use nested row-major lists for readability and JSON
compatibility. `bounds` should use nested two-item lists.

The snapshot should store `lambda`, not `mu`, weights, or strategy constants.
Restore should recompute:

- `mu`
- weights
- `mueff`
- `cc`
- `cs`
- `c1`
- `cmu`
- `damps`
- `chi_n`

The restored eigen cache should start invalid so the next `ask(...)` rebuilds it
from the restored covariance.

## Validation

`PyCMAESState.from_dict(...)` should validate snapshots strictly and raise a
clear `ValueError` for invalid input.

Validation rules:

- `schema_version` must be `1`.
- `optimizer_type` must be `"cmaes"`.
- `state` must be a mapping.
- `n` must be positive.
- `lambda` must be at least `2`.
- `mean`, `pc`, and `ps` must each have length `n`.
- `cov` must be an `n x n` matrix.
- `bounds` must have length `n`.
- each bound pair must satisfy `low < high`.
- all floats must be finite.
- `sigma` must be finite and positive.
- `generation` must be a non-negative integer.
- `pending_eigen_updates` must be a non-negative integer.
- `eigendecomp_interval` must be positive.
- covariance must be symmetric within a small tolerance.
- covariance should reject clearly invalid negative eigenvalues while tolerating
  tiny numerical drift.

Restore should fail before mutating or returning state when validation fails.

## Compatibility Contract

Schema version `1` should be a public compatibility promise for snapshots
produced by `PyCMAESState.to_dict()`.

Within the same EvoCore major version, patch and minor releases should make a
best-effort attempt to load V1 snapshots. If a future CMA-ES implementation
needs a different shape or semantics, it should introduce `schema_version: 2`
rather than silently changing V1.

Compatibility applies to public snapshot payloads, not to internal Rust struct
layout, private helper names, or transient caches.

## Determinism Contract

A restored `PyCMAESState` must produce the same continuation as the original
state when called with the same `master_seed` and generation:

```python
original_next = state.ask(master_seed, state.generation)
restored_next = restored.ask(master_seed, restored.generation)

assert restored_next == original_next
```

After applying the same `tell(...)` data to both states, their snapshots should
match:

```python
state.tell(samples, fitnesses)
restored.tell(samples, fitnesses)

assert restored.to_dict() == state.to_dict()
```

The snapshot contract preserves algorithmic state, not event history.

Implications:

- RNG state is not checkpointed.
- Sampling remains seed-derived from `master_seed` plus generation.
- Event history remains audit data.
- Resume should not depend on replaying old events.
- Replay can remain useful for debugging and inspection, but it is not the
  source of truth for continuation.

## Public API

The PyO3 API should be:

```python
snapshot = state.to_dict()
restored = PyCMAESState.from_dict(snapshot)
```

The public stub in `evocore/_core.pyi` should expose:

```python
def to_dict(self) -> dict[str, object]: ...

@classmethod
def from_dict(cls, snapshot: dict[str, object]) -> PyCMAESState: ...
```

No optimizer-level public checkpoint API should be introduced in this slice.

## Test Plan

Rust-level tests should cover:

- snapshot round trip preserves internal state
- restore invalidates the eigen cache and still samples correctly
- malformed dimensions fail
- invalid floats fail
- nonsymmetric covariance fails
- invalid schema and optimizer type fail through the PyO3-facing API

Python tests in `tests/unit/test_cmaes_rust.py` should cover:

- `to_dict()` returns JSON-safe data
- `from_dict()` restores generation, sigma, mean, and next `ask(...)`
- restored state after the same `tell(...)` matches uninterrupted state
- malformed snapshots raise `ValueError`
- no snapshot field exposes the eigen cache

Because this is a Rust-backed public API change, verification for the
implementation should include Rust formatting, Rust tests, maturin develop, the
targeted Python CMA-ES tests, and the broader unit/integration suite.

## Documentation and Changelog

Implementation should update user-visible documentation to say that CMA-ES has a
Rust state snapshot primitive.

Docs should continue to say that full `CMAESOptimizer` checkpoint/resume remains
future work until Python optimizer ledgers and pending batches are wired to this
primitive.

`CHANGELOG.md` should be updated because this changes public behavior and
checkpoint capability.

## Follow-Up: CMAESOptimizer Ask/Tell Resume

After the Rust primitive is stable, the next dependent slice should wire it into
`CMAESOptimizer` ask/tell checkpointing.

That future checkpoint should combine:

- the Rust `PyCMAESState` snapshot
- Python candidate ledgers
- pending batch records
- continuous samples by candidate id
- event index
- telemetry
- trusted-record counters
- best candidate state

Generation-loop `run()` resume should remain a separate later decision because
it currently uses a local Rust state and would require a broader execution-path
refactor.
