# EvoCore Checkpoint Compatibility And Golden Fixtures Design

**Date:** 2026-05-21
**Status:** Draft approved for specification
**Scope:** Stable checkpoint compatibility contract and committed golden fixtures for EvoCore 0.8.0 checkpoints forward

## Summary

EvoCore should make 0.8.0 the forward checkpoint compatibility baseline for
stable JSON checkpoint files.

The next implementation plan should focus on checkpoint compatibility only, not
whole-framework release readiness. The plan should add a committed golden
fixture suite for v0.8.0 checkpoints and tests proving that future code can load,
validate, and resume those fixtures. Release-readiness should appear only as an
exit gate for the checkpoint surface: documentation, changelog language, and the
relevant verification commands.

Legacy GA pickle checkpoints may remain supported where they already are, but
they are not part of this forward compatibility guarantee.

## Current Context

EvoCore 0.8.0 introduced a stable JSON checkpoint envelope and checkpoint
support across the major continuation surfaces:

- GA generation-loop checkpoints.
- GA ask/tell checkpoints.
- CMA-ES ask/tell checkpoints.
- Rust-backed `PyCMAESState` snapshots nested inside CMA-ES checkpoints.

Current tests cover round trips, malformed payload rejection, identity mismatch
validation, and resume behavior for generated checkpoints. They do not yet prove
that a committed checkpoint file from a released version remains readable after
future implementation changes.

That missing coverage matters because checkpoint files are user artifacts. A
format that only round-trips within the same working tree can still drift between
releases without any test failure.

## Product Direction

Users should be able to preserve optimizer state in a stable checkpoint file and
expect future compatible EvoCore releases to either:

- resume that checkpoint according to the documented contract, or
- reject it with an explicit `CheckpointError` that identifies the
  incompatibility.

The compatibility promise should be narrow and testable. It should cover stable
checkpoint files created by EvoCore 0.8.0 and later, not every historical or
hand-authored payload.

## Goals

- Declare EvoCore 0.8.0 as the stable checkpoint compatibility baseline.
- Commit golden v0.8.0 checkpoint fixtures for the supported stable checkpoint
  surfaces.
- Add tests that load committed fixtures instead of regenerating all checkpoint
  payloads at test time.
- Prove deterministic continuation from fixture checkpoints where resume is part
  of the public contract.
- Guard against accidental fixture churn with manifest hashes or byte-level
  stability checks.
- Document the checkpoint compatibility promise and its exclusions.
- Update `CHANGELOG.md` because the compatibility guarantee is user-visible.
- Keep verification focused on checkpoint compatibility, not a full framework
  release audit.

## Non-Goals

- Do not make this a whole-framework release-readiness plan.
- Do not promise compatibility for legacy GA pickle checkpoints.
- Do not promise compatibility for `OptimizationResult.to_dict()` exports.
- Do not make `EventHistory` replayable checkpoint state.
- Do not support arbitrary hand-written checkpoint JSON beyond documented schema
  validation.
- Do not introduce checkpoint schema v2 unless implementation discovers a real
  incompatibility.
- Do not implement `CMAESOptimizer.run()` resume in this slice.
- Do not serialize evaluators, callbacks, objective functions, thread/process
  pools, progress bars, open files, or external job handles.

## Compatibility Contract

The compatibility guarantee starts with stable JSON checkpoints produced by
EvoCore 0.8.0.

For checkpoint schema v1, compatible patch and minor releases should continue to
load v0.8.0 fixtures and resume supported optimizer state. If a future release
must break compatibility, it should either add an explicit migration path or
raise an actionable `CheckpointError` and document the breaking change in release
notes.

The guarantee covers:

- shared checkpoint envelope v1,
- GA generation-loop state payloads,
- GA ask/tell state payloads,
- CMA-ES ask/tell state payloads,
- nested `PyCMAESState` snapshot payloads used by CMA-ES ask/tell checkpoints.

The guarantee excludes:

- legacy GA pickle checkpoint files,
- result exports,
- event-history replay,
- evaluator and callback objects,
- runtime process state,
- arbitrary mutated checkpoint JSON.

Validation remains part of the contract. Incompatible checkpoints should raise
`CheckpointError` for mismatched schema, checkpoint kind, optimizer type,
gene-space hash, optimizer config hash, seed, direction, state kind, or nested
CMA-ES state.

## Fixture Layout

Add committed fixtures under:

```text
tests/fixtures/checkpoints/v0.8.0/
```

The fixture set should include:

```text
manifest.json
ga-generation-loop.evocore-checkpoint.json
ga-ask-tell-after-ask.evocore-checkpoint.json
ga-ask-tell-after-partial-tell.evocore-checkpoint.json
cmaes-ask-tell-after-ask.evocore-checkpoint.json
cmaes-ask-tell-after-consumed-batch.evocore-checkpoint.json
```

`manifest.json` should describe each fixture:

- file name,
- fixture format version,
- source EvoCore version,
- checkpoint schema version,
- optimizer type,
- state kind,
- seed,
- direction,
- expected gene-space hash,
- expected optimizer config hash,
- stable file hash,
- expected continuation assertion.

The fixture files should be deterministic UTF-8 JSON saved with the same stable
serialization used by `save_checkpoint(...)`. They should not contain test-only
comments, runtime-only objects, local paths, timestamps beyond existing writer
metadata, or generated build output.

## Fixture Generation

The implementation plan should include a small internal generation script or
test helper that can recreate the fixture payloads when the compatibility
baseline intentionally changes. The fixture tests should not silently rewrite
files.

Fixture generation should use small deterministic optimizer configurations so
the files remain readable:

- fixed seeds,
- tiny gene spaces,
- small population sizes,
- deterministic scoring helpers,
- stable metadata with no machine-local values except existing `created_by`
  diagnostics.

If current `created_by.platform` or Python version metadata makes fixture bytes
too environment-sensitive, the implementation should normalize those fields in
fixture construction while preserving the production behavior of
`CheckpointSnapshot.to_dict()`.

## Test Strategy

Add a focused fixture test module, for example:

```text
tests/unit/test_checkpoint_golden_fixtures.py
```

The tests should have two layers.

Shape tests:

- every manifest entry has a matching fixture file,
- every fixture file hash matches the manifest,
- every fixture is valid stable JSON,
- `load_checkpoint(...)` accepts every valid fixture,
- every valid fixture uses checkpoint schema v1,
- every valid fixture has the expected optimizer type and state kind,
- fixture payloads contain no unsupported runtime-only data.

Behavior tests:

- GA generation-loop fixture resumes and reaches the expected final continuation
  assertion.
- GA ask/tell after-ask fixture resumes, accepts expected trusted records, and
  consumes the pending batch.
- GA ask/tell after-partial-tell fixture resumes, accepts missing records, and
  preserves the accepted records from before the checkpoint.
- CMA-ES ask/tell after-ask fixture resumes, accepts expected trusted records,
  and consumes the pending batch.
- CMA-ES ask/tell after-consumed-batch fixture resumes and the next `ask()`
  matches the expected candidate IDs, batch IDs, and genes.

Tests should compare observable continuation behavior rather than private
implementation details where possible. Private state checks are appropriate only
when no public surface exposes the contract being protected.

## Error Behavior

The fixture suite should also include or derive negative cases that prove stable
errors for common incompatibilities:

- unsupported checkpoint schema version,
- wrong checkpoint kind,
- optimizer type mismatch,
- gene-space hash mismatch,
- optimizer config hash mismatch,
- seed mismatch,
- direction mismatch,
- wrong state kind,
- malformed nested CMA-ES state.

These negative cases can be generated in-memory from valid fixtures so the
repository does not need to carry many near-duplicate JSON files.

The tests should assert that failures raise `CheckpointError` and that the
message names the incompatible field.

## Documentation And Changelog

Update public docs to state:

- EvoCore 0.8.0 is the stable checkpoint compatibility baseline.
- Stable JSON checkpoints are the forward-compatible format.
- Legacy GA pickle checkpoints are legacy support, not part of the forward
  compatibility guarantee.
- Result exports and event history exports are not checkpoint files.
- CMA-ES ask/tell checkpoints are supported, while CMA-ES generation-loop and
  policy-run resume remain outside this guarantee unless implemented later.

Likely docs surface:

- `docs/site/callbacks-checkpointing.md`
- `docs/site/ga.md` if the GA checkpoint wording needs a baseline note
- `docs/site/cmaes.md` if the CMA-ES checkpoint wording needs a baseline note

Update `CHANGELOG.md` under `[Unreleased]` to mention the 0.8.0 checkpoint
compatibility fixture suite and documented compatibility baseline.

## Verification

Run the focused checkpoint checks first:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_checkpoint_golden_fixtures.py -v
.\.venv\Scripts\python.exe -m pytest tests/unit/test_checkpointing.py tests/unit/test_ask_tell_checkpointing.py tests/unit/test_cmaes_ask_tell_checkpointing.py -v
```

Run formatting and lint checks:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
.\.venv\Scripts\python.exe -m ruff check
```

Broaden verification if touched files require it:

- Run `.\.venv\Scripts\python.exe -m pytest tests/unit/ tests/integration/ -v`
  if shared checkpoint runtime code, optimizer restoration logic, or public
  serialization helpers change.
- Run `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`,
  `cargo test`, and `.\.venv\Scripts\python.exe -m maturin develop --release`
  if Rust code, PyO3 signatures, or `evocore/_core.pyi` change.
- Run `.\.venv\Scripts\python.exe -m mkdocs build` if user docs change.

## Rollout

Recommended implementation slices:

1. Add fixture directory, manifest format, and committed v0.8.0 fixture files.
2. Add shape tests for fixture validity, manifest consistency, and stable hashes.
3. Add behavior tests for GA and CMA-ES fixture resume.
4. Add in-memory negative compatibility tests derived from fixture payloads.
5. Update docs and changelog with the 0.8.0 checkpoint baseline.
6. Run focused checkpoint verification, then broaden only if implementation
   touches shared runtime surfaces.

## Acceptance Criteria

- `tests/fixtures/checkpoints/v0.8.0/` contains the manifest and golden
  checkpoint fixtures for GA generation-loop, GA ask/tell, and CMA-ES ask/tell.
- Fixture tests fail if fixture files are accidentally rewritten or reformatted.
- Current EvoCore can load all v0.8.0 fixture checkpoints.
- Current EvoCore can resume all fixture checkpoints that represent supported
  continuation states.
- Compatibility failures raise `CheckpointError` with field-specific messages.
- Public docs identify 0.8.0 as the stable checkpoint compatibility baseline.
- `CHANGELOG.md` records the fixture-backed compatibility guarantee.
- Legacy pickle support remains explicitly outside the forward compatibility
  guarantee.

## Deferred Follow-Ups

- A migration registry for future checkpoint schema versions.
- Golden fixtures for future schema v2 payloads if a new schema is introduced.
- Compatibility test matrix that installs published EvoCore releases and
  regenerates fixtures from real wheels.
- External artifact-store examples for long-running optimization checkpoints.
- CMA-ES generation-loop or policy-run resume if those execution paths later get
  checkpoint support.
