# Differential Evolution Stabilization Design

**Date:** 2026-06-02
**Status:** Draft approved for specification
**Scope:** Promote `DifferentialEvolutionOptimizer` from newly merged optimizer
to first-class, release-stable optimizer alongside GA and CMA-ES.

## Summary

`DifferentialEvolutionOptimizer` now exists as a public EvoCore optimizer with
mixed search-space support, ask/tell execution, synchronous evaluator-driven
`run()`, checkpointing, docs, and tests. The next goal is not to expand the DE
feature surface. It is to make DE trustworthy on the same footing as GA and
CMA-ES by hardening reproducibility, checkpoint compatibility, edge-case
behavior, and documentation.

This stabilization slice should prove that DE's current semantics are stable and
observable. A checkpointed DE run should resume like an uninterrupted run. A
seeded DE run should produce deterministic candidates, replacement decisions,
and summaries. Mixed `float`, `int`, and `bool` search spaces should remain
valid through initialization, trial generation, replacement, and restore.
User-facing docs should describe those guarantees and the intentional
limitations that remain for later parity work.

The recommended roadmap order is:

1. Release-maturity parity.
2. Feature parity.
3. Performance parity.

This document covers only the first step.

## Current Context

The first DE implementation followed the design in:

```text
docs/superpowers/specs/2026-05-21-differential-evolution-optimizer-design.md
```

It added the DE package:

```text
evocore/optimizers/de/
  __init__.py
  ask_tell.py
  checkpointing.py
  config.py
  engine.py
```

It also introduced shared optimizer-state acceptance decisions through
`UpdateResult.acceptance_decisions` and `UpdateResult.state_accepted_count`.
For DE, those decisions describe whether a trial candidate replaced its target
population slot.

GA and CMA-ES already have deeper maturity in several areas: committed golden
checkpoint fixtures, broad resume tests, long-lived docs, and established
release expectations. DE should now receive the same stabilization treatment
before new convenience features such as `run_multiple(...)`, policy-aware
execution, or performance kernels are added.

## Goals

- Treat DE checkpoint schema v1 as stable from this stabilization pass forward.
- Add committed DE golden checkpoint fixtures and manifest coverage.
- Prove checkpoint restore and continuation behavior for initialization and
  trial/replacement phases.
- Prove deterministic seeded behavior for candidate generation, replacement
  decisions, and result summaries.
- Broaden edge-case tests for mixed search spaces and ask/tell lifecycle errors.
- Lock down minimize and maximize replacement semantics.
- Make DE docs clear enough that users can understand when to choose DE, how to
  checkpoint/resume it, and what limitations remain.
- Update `CHANGELOG.md` because checkpoint compatibility and DE stabilization
  are user-visible.

## Non-Goals

- Do not add `DifferentialEvolutionOptimizer.run_multiple(...)` in this slice.
- Do not add policy or budget-aware `run(...)` support in this slice.
- Do not add custom DE strategy plugins.
- Do not add a Rust/PyO3 DE kernel.
- Do not add a benchmark suite except for notes that inform later performance
  work.
- Do not change public names or move DE modules unless a defect requires it.
- Do not break existing DE checkpoint schema v1 unless implementation uncovers a
  real correctness bug that must be fixed and documented.

## Architecture

The implementation should keep DE in its current package and harden existing
boundaries instead of introducing a new layer.

`engine.py` remains responsible for the public optimizer class, synchronous
`run()` orchestration, callback flow, result construction, evaluator integration,
and direction handling.

`ask_tell.py` remains responsible for lifecycle state: initialization
candidates, trial candidates, target-slot metadata, stale candidate rejection,
duplicate tell rejection, trusted/cached score handling, replacement decisions,
event emission, and `UpdateResult` summaries.

`checkpointing.py` remains responsible for stable serialization and restore of
DE ask/tell state. It should own compatibility validation and errors for
malformed or incompatible checkpoint payloads.

`config.py` remains responsible for constructor validation, hash-stable
configuration export, and unsupported option behavior.

Tests and docs are part of the architecture for this slice. The goal is not only
to make the implementation correct today, but to make accidental semantic drift
visible in future changes.

## Lifecycle Data Flow

The stabilized DE lifecycle should be specified around three repeatable flows.

### Initial Population

`ask()` emits initialization candidates until the population is full. `tell()`
records evaluated scores into fixed population slots for state-eligible records.
The order of slots, candidate IDs, candidate values, and completed records must
be deterministic for a given seed and search space.

Checkpointing during initialization must preserve pending candidates, filled
population slots, evaluated scores, and counters. Restoring from that checkpoint
and completing the population should produce the same state as an uninterrupted
run.

### Trial Replacement

Once initialized, `ask()` emits trial candidates tied to target population
members. `tell()` compares each state-eligible trial against its target using
the optimizer direction:

- maximize accepts a trial when its score is greater than or equal to the target
  score;
- minimize accepts a trial when its score is less than or equal to the target
  score.

Accepted trials replace their target slots. Rejected trials leave population
state unchanged. `UpdateResult.acceptance_decisions` should expose each
state-eligible decision with the candidate ID, target candidate ID, target slot,
boolean acceptance, and reason.

### Checkpoint Resume

DE checkpoint state should contain enough information for:

```text
original uninterrupted run == restored run plus same tells
```

This includes RNG state, population values, population scores, best solution,
generation counters, evaluation counters, pending candidates, pending target
mapping, completed records, optimizer direction, seed, gene-space identity, and
config identity.

The restore path should reject incompatible optimizer type, checkpoint kind,
schema version, gene-space hash, config hash, seed, or direction before the next
optimization action mutates state.

## Checkpoint Fixture Contract

Add DE fixtures under the existing checkpoint fixture structure:

```text
tests/fixtures/checkpoints/v0.9.0/
```

Use flat `de-*` fixture names in that directory, matching the existing v0.8.0
fixture convention. EvoCore is currently versioned as 0.9.0, and this is the
first version line expected to contain DE. If release preparation changes that
version before implementation, update this spec and the implementation plan
together instead of silently creating fixtures under a different baseline.

The fixture set should include at least:

- a DE checkpoint after `ask()` during initialization;
- a DE checkpoint after partial initialization `tell()`;
- a DE checkpoint after a full population is initialized;
- a DE checkpoint after trial `ask()`;
- a DE checkpoint after mixed accepted and rejected trial tells.

Each fixture should have a manifest entry with:

- file name;
- fixture format version;
- source EvoCore version;
- checkpoint schema version;
- optimizer type;
- state kind;
- seed;
- direction;
- expected gene-space hash;
- expected optimizer config hash;
- stable file hash;
- continuation assertion.

Fixture tests should load committed files rather than regenerating them at test
time. A separate explicit helper may regenerate fixtures when the compatibility
baseline intentionally changes, but normal tests must not rewrite fixtures.

## Error Handling

Invalid DE configuration should raise `ConfigurationError` before any run state
is created. This includes invalid population size, mutation factor, crossover
rate, generation budget, evaluation budget, direction, parallel mode, and any
unsupported policy behavior.

Invalid evaluations should follow existing EvoCore error conventions. Unknown
candidates, stale candidates, wrong batch IDs, duplicated stage records, invalid
scores, and malformed records should fail through explicit public errors instead
of silently mutating state.

Checkpoint errors should be clear and early. Malformed payloads, incompatible
optimizer identity, missing required fields, bad schema values, and mismatched
gene-space/config identity should fail during restore. A restored optimizer
should not accept a checkpoint that will only fail during the next `ask()` or
`tell()` due to missing state.

Mixed search-space errors should be deterministic. If a gene value is invalid
after trial generation or restore, the error should identify the checkpoint or
search-space compatibility issue rather than producing an invalid `Solution`.

## Documentation

The docs should position DE as a first-class optimizer while keeping its
limitations explicit.

The DE guide should cover:

- when DE is a good fit compared with GA and CMA-ES;
- mixed search-space behavior;
- seed determinism;
- ask/tell usage;
- synchronous `run()` usage;
- checkpoint save/restore;
- interpretation of `acceptance_decisions` and `state_accepted_count`;
- current non-support for `run_multiple(...)` and policy-aware `run(...)`.

Checkpoint documentation should mention DE alongside GA and CMA-ES wherever the
supported stable checkpoint surfaces are listed.

The changelog should describe DE stabilization, committed checkpoint fixtures,
and the compatibility promise in user-facing language.

## Testing Strategy

The implementation should be test-led. Focused tests should be added or expanded
before implementation changes where practical.

Golden fixture tests should verify:

- every DE manifest entry has a matching file;
- every fixture hash matches the manifest;
- every fixture is valid stable JSON;
- every fixture loads through the public checkpoint path;
- every fixture has expected optimizer type, state kind, schema version, seed,
  direction, gene-space hash, and config hash;
- restored continuation matches the fixture's expected assertion.

Resume equivalence tests should compare uninterrupted and checkpointed/resumed
DE runs across initialization and trial phases.

Determinism tests should verify that same seed plus same search space produces
identical candidate values, target mappings, replacement decisions, best
solution, and result summaries.

Mixed-space tests should cover float, integer, boolean, and fixed genes through
initialization, trial generation, replacement, and restore.

Replacement tests should cover accepted and rejected trials for both maximize
and minimize. They should assert population updates, best-solution updates,
`state_accepted_count`, and `acceptance_decisions`.

Lifecycle error tests should cover duplicate tells, stale candidates, unknown
candidates, wrong batch IDs, bad scores, incomplete batches, rejected records,
surrogate/partial records, and malformed checkpoints.

Docs/API smoke tests should ensure the public DE imports and representative
examples do not drift.

## Verification

The implementation plan should start with targeted DE tests and then broaden.
Expected verification commands are:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
.\.venv\Scripts\python.exe -m ruff check
.\.venv\Scripts\python.exe -m maturin develop --release
.\.venv\Scripts\python.exe -m pytest tests/unit/ tests/integration/ -v
```

Rust verification is not expected unless implementation touches Rust, PyO3
stubs, or cross-language contracts. If Rust is touched, run the relevant Rust
formatting, linting, and test commands from `AGENTS.md`.

## Acceptance Criteria

- DE golden checkpoint fixtures are committed and covered by manifest tests.
- Restored DE checkpoints continue deterministically from initialization and
  trial phases.
- Seeded DE runs are reproducible across the lifecycle surfaces covered by the
  tests.
- Mixed search-space values remain valid after init, trial generation,
  replacement, and restore.
- Invalid lifecycle and checkpoint inputs raise explicit errors.
- DE docs describe checkpointing, reproducibility, acceptance decisions, and
  known limitations.
- `CHANGELOG.md` captures the user-visible stabilization work.
- Verification passes before the implementation branch is committed.

## Future Work

After this stabilization pass, the next DE goal should be feature parity:

- add `run_multiple(...)`;
- decide how DE should integrate with `BudgetPolicy`;
- deepen callback/checkpoint examples;
- consider more DE strategies.

After feature parity, performance parity can be planned with benchmarks,
profiling, and possible Rust/PyO3 acceleration where measurements justify it.
