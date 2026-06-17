# EvoCore Phase 2 Expensive Optimization Toolkit Design

**Date:** 2026-06-17
**Status:** Approved design direction, awaiting user spec review
**Scope:** Reusable archive, selection, stopping, and hybrid composition utilities built on the Phase 1 external-state API

## Summary

Phase 2 turns the Phase 1 external-state API into a reusable toolkit for
expensive black-box optimization systems. Phase 1 gave GA, DE, and CMA-ES
stable public methods for warm starts, candidate injection, candidate
snapshots, top-k snapshots, capabilities, and cached evaluation records. Phase
2 should not add more optimizer-specific state access first. It should add
small public utility modules that consume `CandidateSnapshot` and
`PopulationSnapshot`, then produce decisions, archives, warm-start records,
stop decisions, and composition metadata.

Trading-Algo-Scalper-Gold remains the reference integration pressure test:
search memory, cached backtests, seed pools, staged refinement, family
diversity, specialist caps, and outer GA plus inner CMA behavior must be easy
to express. The EvoCore API must stay generic and useful to any expensive
external evaluator.

## Confirmed Phase 1 Baseline

The latest merged Phase 1 state provides these public building blocks:

- `WarmStartRecord`
- `CandidateSnapshot`
- `PopulationSnapshot`
- `ExternalStateCapabilities`
- `InjectionResult`
- `cached_records(...)`
- `ExternalStateOptimizer`
- optimizer methods:
  - `external_state_capabilities()`
  - `warm_start(...)`
  - `inject_candidates(...)`
  - `candidate_snapshot(...)`
  - `top_candidates(...)`

Phase 1 also confirms:

- cached records are state-eligible when passed through approved update paths;
- candidate and record metadata round-trip through events and checkpoints;
- snapshots are detached from optimizer internals;
- GA, DE, and CMA-ES have honest optimizer-specific capability reporting;
- CMA-ES supports conservative tracked-only external injection after state
  creation.

Phase 2 should build on those contracts rather than reopen them.

## Goals

- Add durable archive/search-memory utilities for external expensive systems.
- Add survivor and promotion selection utilities over public snapshots.
- Add reusable diversity, duplicate suppression, family quota, and specialist
  cap behavior.
- Add stop and stall policies for ask/tell loops and generation helpers.
- Add lightweight composition helpers for outer optimizer and inner optimizer
  workflows.
- Preserve deterministic seed behavior and checkpoint compatibility.
- Keep optimizers as state owners. Phase 2 utilities should make decisions,
  not mutate optimizer private state.
- Keep Trading-Algo-specific concepts in examples and metadata, not public
  EvoCore class names.

## Non-Goals

- Do not redesign `GeneSpace` for conditional or hierarchical parameters in
  Phase 2.
- Do not add a heavyweight formal hybrid optimizer class yet.
- Do not change GA, DE, or CMA-ES method signatures unless required by a
  narrow compatibility fix.
- Do not embed archive state into optimizer checkpoints by default.
- Do not make stop policies part of `BudgetPolicy` yet.
- Do not add CMA-ES restart or post-start covariance warm-start semantics in
  Phase 2.
- Do not add Trading-Algo dependencies or trading-specific public names.

## Approaches Considered

### Approach A: Utility-First Toolkit

Add focused lifecycle modules:

- `evocore.lifecycle.archives`
- `evocore.lifecycle.selection`
- `evocore.lifecycle.stopping`
- `evocore.lifecycle.composition`

These modules operate on Phase 1 public data objects and return explicit
results. Optimizers remain unchanged except for optional docs and examples.

Advantages:

- lowest compatibility risk;
- immediately useful to external expensive workflows;
- works across GA, DE, and CMA-ES;
- keeps API boundaries easy to test.

Trade-off:

- users must compose the pieces in their own ask/tell loops.

### Approach B: Policy-Integrated Toolkit

Add the same utilities but wire them directly into optimizer run helpers,
`BudgetPolicy`, and callbacks in the same phase.

Advantages:

- higher-level user experience for common workflows;
- less user orchestration code.

Trade-off:

- higher risk of coupling stop, selection, and budget semantics too early;
- more optimizer-specific behavior to freeze.

### Approach C: Hybrid Workflow Framework

Start with a formal outer/inner optimizer composition layer that owns archive,
selection, stop, and lineage behavior.

Advantages:

- directly addresses Trading-Algo-style hybrid GA plus CMA optimization.

Trade-off:

- too likely to encode one domain workflow as the general abstraction;
- harder to keep small and stable before multiple integrations exercise it.

## Recommendation

Use Approach A for Phase 2. Ship a utility-first toolkit, then add deeper
integration only after the primitives prove stable in recipes and external
workflows. This provides the missing reusable layer without disturbing the
merged Phase 1 optimizer API.

## Architecture Boundary

Optimizers own ask/tell state. Phase 2 utilities own external decisions.

The intended data flow is:

```text
optimizer.ask(...) or optimizer.inject_candidates(...)
external evaluator, cache, or worker queue
optimizer.tell(...)
optimizer.candidate_snapshot(...) or optimizer.top_candidates(...)
archive, selection, stopping, and composition helpers
optimizer.warm_start(...) or optimizer.inject_candidates(...) for later stages
```

Utilities should consume immutable snapshots and ordinary records. They should
not reach into `_trusted_population_vnext`, DE target slots, CMA internal
state, pending batch internals, or other private collections.

## Public Module: Archives

Add `evocore.lifecycle.archives`.

Primary public objects:

- `CandidateArchive`
- `ArchiveEntry`
- `ArchivePolicy`
- `DuplicatePolicy`
- `ArchiveExport`

Suggested API shape:

```python
archive = CandidateArchive(
    duplicate_policy="keep_best",
    score_direction="maximize",
)

archive.add_population(optimizer.candidate_snapshot(scope="trusted"), source="stage1")
archive.add_candidates(optimizer.top_candidates(k=8), source="promotion")

records = archive.to_warm_start_records(
    k=20,
    stage="archive",
    confidence="cached",
)
optimizer.warm_start(records, mode="state")
```

Behavior:

- archive keys default to `candidate_hash`;
- archive entries preserve values, params, score, confidence, stage, cost,
  metrics, metadata, source, and archive insertion metadata;
- duplicate handling is explicit:
  - `keep_first`
  - `keep_latest`
  - `keep_best`
- score direction is explicit and required unless it can be safely inherited
  from a `PopulationSnapshot`;
- export returns existing `WarmStartRecord` objects;
- optional JSON export/import must include a schema version.

The archive should be a user-owned object. Optimizer checkpoints should not
automatically include archive contents.

## Public Module: Selection

Add `evocore.lifecycle.selection`.

Primary public objects and helpers:

- `select_candidates(...)`
- `SelectionResult`
- `SelectionDecision`
- `SelectionReason`
- `FamilyQuota`
- `SpecialistCap`
- `DuplicateSuppression`
- `DiversityMetric`

Suggested API shape:

```python
selection = select_candidates(
    optimizer.candidate_snapshot(scope="trusted").candidates,
    k=32,
    score_direction="maximize",
    duplicate_policy="suppress",
    quotas=[
        FamilyQuota(metadata_key="family", max_count=6),
        SpecialistCap(metadata_key="specialist", max_count=3),
    ],
)
```

`SelectionResult` should include:

- selected candidate snapshots;
- rejected or skipped candidate snapshots;
- stable reasons for every non-selected candidate;
- summary counts by confidence, source, family, specialist, and duplicate
  status;
- deterministic ordering under score ties.

Behavior:

- selection operates over `CandidateSnapshot`, not live candidates;
- duplicate suppression defaults to `candidate_hash`;
- quotas and caps read metadata keys;
- missing metadata behavior is explicit:
  - default: bucket as `"unknown"`;
  - strict mode: raise `ConfigurationError`;
- selection never rewrites scores, confidence, or metadata;
- selection can export selected candidates as `WarmStartRecord`s through a
  helper, but archive export remains the preferred durable path.

## Public Module: Stopping

Add `evocore.lifecycle.stopping`.

Primary public objects:

- `StopPolicy`
- `StopDecision`
- `EvaluationLimitPolicy`
- `NoImprovementPolicy`
- `ConvergencePolicy`
- `CompositeStopPolicy`

Suggested API shape:

```python
stop_policy = CompositeStopPolicy([
    EvaluationLimitPolicy(max_evaluations=5000),
    NoImprovementPolicy(window=12, min_delta=0.001),
])

decision = stop_policy.observe(
    update_result,
    snapshot=optimizer.candidate_snapshot(scope="trusted"),
)
if decision.stop:
    break
```

Behavior:

- stop policies are usable in manual ask/tell loops;
- stop policies can later be adapted by callbacks or generation helpers;
- stop policies consume `UpdateResult`, `PopulationSnapshot`, telemetry, or
  event summaries;
- `StopDecision` includes `stop`, `reason`, `message`, and JSON-safe metadata;
- stop policies do not spend budget or mutate optimizer state.

`BudgetPolicy` should remain responsible for budget accounting. Stop policies
decide termination. Phase 2 examples can show them working together without
merging the concepts.

## Public Module: Composition

Add `evocore.lifecycle.composition`.

Primary public helpers:

- deterministic child seed derivation for nested optimizers;
- lineage metadata construction;
- inner optimizer result to outer `EvaluationRecord` conversion;
- optional composition result dataclasses for recipe clarity.

Suggested API shape:

```python
inner_seed = derive_child_seed(
    parent_seed=outer_seed,
    candidate_hash=template.candidate_hash(space),
    stage="inner_cma",
)

metadata = lineage_metadata(
    outer_candidate=template,
    inner_optimizer_type="cmaes",
    inner_seed=inner_seed,
    stage="template_refinement",
)

record = inner_result_record(
    outer_candidate=template,
    score=tuned_score,
    confidence="trusted_full",
    stage="inner_cma",
    metadata=metadata,
)
```

Behavior:

- helpers preserve outer candidate ID/hash, inner optimizer type, stage, seed,
  archive identifiers, and optional checkpoint pointers;
- helpers should not assume GA or CMA specifically where a generic optimizer
  shape is enough;
- docs can show outer GA plus inner CMA as the main recipe;
- a formal hybrid optimizer class is deferred until real usage validates the
  helper API.

## Phase 3 Boundary

Phase 3 should handle deeper optimizer and search-space semantics:

- constraint and repair hooks;
- conditional and dependent parameters;
- hierarchical search spaces;
- active continuous subspaces for template-based optimization;
- CMA-ES warm-start and restart improvements;
- richer checkpoint inspection for partially completed external batches.

Those changes touch optimizer and search-space behavior more directly, so they
belong after Phase 2 utilities stabilize.

## Backward Compatibility Risks

Archive serialization is the largest sticky surface. If public JSON import and
export ships, include a schema version immediately.

Duplicate identity should default to `candidate_hash`, which ties behavior to
`GeneSpace.value_hash(...)`. Any future hash contract change must be treated as
a reproducibility-affecting change.

Score direction must be explicit in archive and selection utilities unless a
snapshot provides unambiguous direction. Silent maximize/minimize mistakes would
be dangerous for expensive systems.

Metadata-driven quotas need deterministic missing-key behavior. The default
should be an `"unknown"` bucket; strict mode should raise `ConfigurationError`.

Stop reasons should align with existing lifecycle event vocabulary and avoid a
parallel incompatible taxonomy.

Optimizer checkpoints should remain compatible. Archive persistence should be a
separate user-owned artifact unless an explicit future design embeds it.

Top-level `evocore` exports should stay small. Prefer lifecycle-level exports
first and add convenience imports only for the most common names.

## Tests

### Archive Tests

- Add snapshots from GA, DE, and CMA-ES Phase 1 APIs.
- Preserve candidate values, params, score, confidence, stage, cost, metrics,
  metadata, source, and insertion metadata.
- Deduplicate by hash with `keep_first`, `keep_latest`, and `keep_best`.
- Export deterministic `WarmStartRecord` objects.
- Round-trip archive JSON if serialization is public.
- Reject or report non-JSON-safe metadata consistently with existing lifecycle
  validation.

### Selection Tests

- Select top-k deterministically for maximize and minimize direction.
- Preserve stable ordering for score ties.
- Suppress duplicates by candidate hash.
- Enforce family quotas from metadata.
- Enforce specialist caps from metadata.
- Handle missing metadata through default and strict policies.
- Return reasons for skipped and rejected candidates.
- Confirm selection does not mutate snapshots or optimizer state.

### Stopping Tests

- `EvaluationLimitPolicy` stops exactly at the configured cap.
- `NoImprovementPolicy` respects window and `min_delta`.
- `ConvergencePolicy` handles threshold boundaries.
- `CompositeStopPolicy` returns deterministic decisions.
- Policies behave correctly with cached and trusted records.
- Custom reason metadata remains JSON-safe.

### Composition Tests

- Child seed derivation is deterministic.
- Lineage metadata preserves outer candidate identity and inner optimizer
  details.
- Inner results convert to valid outer `EvaluationRecord` objects.
- Hybrid recipe can warm-start an inner optimizer from archive records.
- Helpers do not require Trading-Algo-specific metadata keys.

### Documentation Tests

- Archive-backed warm start recipe.
- Stage survivor promotion recipe.
- Family quota and specialist cap recipe.
- Ask/tell loop with stop policies.
- Cached expensive evaluator loop.
- Outer GA plus inner CMA recipe.
- Checkpoint and resume notes for partially completed external work.

## Documentation

Add or expand docs for:

- expensive search memory archives;
- survivor selection from snapshots;
- family-aware promotion;
- specialist caps;
- stop policies in ask/tell loops;
- archive-backed warm starts;
- hybrid outer optimizer plus inner optimizer composition;
- deterministic reproducibility for nested runs;
- async or queued expensive evaluation workflows.

Trading-Algo can be described as a motivating shape, but examples should use
generic strategy/template/evaluator language and toy code that does not depend
on Trading-Algo.

## Rollout

### Phase 2A: Archive And Selection Core

Implement:

- `evocore.lifecycle.archives`;
- `evocore.lifecycle.selection`;
- lifecycle package exports;
- focused unit tests;
- docs for search memory and survivor selection.

This is the highest-value first slice because it supports search memory, seed
pools, staged refinement, family diversity, specialist caps, and promotion
without changing optimizer internals.

### Phase 2B: Stop Policies

Implement:

- `evocore.lifecycle.stopping`;
- stop decisions for evaluation caps, no-improvement windows, convergence
  thresholds, and composite policies;
- ask/tell recipe docs;
- tests for cached and trusted evaluation accounting interactions.

Stop policies should remain separate from `BudgetPolicy` in this phase.

### Phase 2C: Composition Recipes

Implement:

- `evocore.lifecycle.composition`;
- deterministic child seed and lineage helpers;
- inner-result-to-outer-record helper;
- hybrid outer optimizer plus inner optimizer docs.

Keep this recipe-first. Defer a formal hybrid optimizer abstraction.

## Acceptance Criteria

- External systems can build durable search memory from public snapshots.
- Archive contents can be exported back into existing warm-start APIs.
- External systems can select survivors with diversity, family, specialist, and
  duplicate policies without private state access.
- Ask/tell loops can use reusable stop policies with explicit stop decisions.
- Hybrid workflows can preserve lineage and deterministic nested seeds.
- Existing GA, DE, CMA-ES ask/tell behavior and checkpoints remain compatible.
- Documentation demonstrates expensive external evaluation workflows without
  Trading-Algo-specific dependencies.
- Phase 3 remains clearly scoped to search-space, repair, and deeper CMA
  behavior.
