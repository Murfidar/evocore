# EvoCore Phase 3 Projection and Mixed-Numeric CMA Design

**Date:** 2026-06-22
**Status:** Approved design, awaiting user spec review
**Scope:** Named domain projections, deterministic constraints and transforms, constraint-penalty records, active-subspace CMA, native integer-margin handling, and lifecycle-managed CMA restarts

## Summary

Phase 3 adds the missing domain-to-optimizer translation layer for expensive black-box systems. EvoCore already has stable ask/tell lifecycle contracts, external warm starts and snapshots, cached records, archives, selection and stopping policies, and outer/inner composition helpers. What remains difficult is translating named, conditional, repaired, and partially active domain parameters into the ordered numeric coordinates required by optimizer engines.

The design introduces a protocol-first projection layer. A projection binds structural values, selects active named genes, transforms domain and optimizer representations, reconstructs complete parameter mappings, validates or repairs dependent values, and produces stable identity and snapshot data. The flat `GeneSpace` remains the optimizer-native representation and keeps schema version 1.

Phase 3 also completes EvoCore's mixed-numeric CMA direction. Native integer genes gain explicit `round` and `margin` strategies. `round` remains the compatibility default. `margin` becomes real ask/tell behavior with deterministic sampling and checkpoint support, not merely a standalone utility. Fixed, boolean, categorical, and inactive values remain outside active CMA coordinates and are reconstructed by the projection.

Trading-Algo-Scalper-Gold is the proving workload, but no trading-specific API belongs in EvoCore. Its hybrid flow uses a discrete outer GA template and a template-specific inner CMA space made of bounded float coordinates; some coordinates decode into integers or toggles. That validates conditional projection, transforms, caching, warm starts, staged selection, and lineage. Native integer-margin CMA remains in Phase 3 as a general EvoCore capability even though Trading-Algo's inner optimizer does not require it for migration.

## Long-Term Direction

EvoCore should become the optimizer-agnostic control layer for external expensive black-box optimization systems:

```text
named domain parameters
        |
projection, transforms, repair, validation, active selection
        |
ordered optimizer-native GeneSpace coordinates
        |
GA, DE, CMA-ES, and future optimizer engines
        |
external evaluators, caches, workers, and domain infrastructure
        |
records, archives, telemetry, checkpoints, and reproducible results
```

External applications own evaluation, workers, databases, and domain policy. EvoCore owns optimizer state, translation into optimizer coordinates, reusable search policies, lifecycle semantics, and reproducibility contracts. A future package may offer both composable primitives and an optional `OptimizationSession`; Phase 3 builds the required primitives but not that runtime.

## Confirmed Baseline

- Phase 1 provides warm starts, candidate injection, detached snapshots, top-k access, cached-record helpers, metadata persistence, and conservative pre-start CMA mean construction.
- Phase 2 provides archives, duplicate suppression, family quotas, specialist caps, stopping policies, and deterministic child-seed and lineage helpers.
- `GeneSpace` is flat and ordered; `signature()` and `value_hash()` use schema version 1.
- CMA accepts float and integer genes, rejects booleans and fixed numeric genes, requires full-length initial means, and checkpoints Rust-backed state plus pending continuous samples.
- `IntegerMarginDistribution` and `CategoricalDistributionState` exist as isolated foundations but are not integrated into CMA ask/tell.
- Checkpoints validate exact search-space and optimizer-config hashes.

## Trading-Algo Reference Findings

Trading-Algo's outer discrete GA chooses signal family, session, TP mode, EMA behavior, and regime behavior. `build_continuous_space(outer)` then creates only the bounded inner coordinates active for that template.

The inner coordinates are floats. Reconstruction uses identity decoding, logarithmic integer decoding, binary threshold decoding, conditional family parameters, conditional TP parameters, and a final compilation step that derives execution-facing aliases and flags.

The broader workflow needs historical seed pools with family caps, persistent search memory, cached results, random immigrants, unique stage-1 archives, family-diverse survivors, specialist caps, stage-2 refinement, and outer/inner accounting. Phase 1 and Phase 2 cover most lifecycle behavior. Phase 3 must make the representation boundary reusable and reproducible.

## Goals

- Add a stable public protocol for translating named domain parameters to and from optimizer-native coordinates.
- Require stable names only for projection-aware workflows; preserve unnamed flat workflows.
- Support structural bindings, active-gene selection, coordinate transforms, dependent repair, and validation.
- Exclude inactive values from canonical identity while including structural bindings and projection semantics.
- Support portable versioned hooks and honest runtime-only callables.
- Convert irreparable generated candidates into deterministic zero-cost penalties that complete optimizer batches without becoming trusted evidence.
- Build CMA means from projected domain values and trusted historical records.
- Integrate native integer-margin behavior into real CMA ask/tell execution.
- Preserve `round` as the default integer strategy.
- Add deterministic lifecycle-managed fresh-run CMA restarts.
- Preserve existing seeds, flat spaces, and checkpoints unless a user selects a new feature.

## Non-Goals

- Do not mutate `GeneSpace` into a hierarchical or conditional DSL.
- Do not add native categorical genes or categorical-distribution CMA.
- Do not add an in-place `CMAESOptimizer.restart()`.
- Do not add a formal hybrid engine, `OptimizationSession`, worker queue, persistence backend, or telemetry sink.
- Do not serialize arbitrary Python code.
- Do not add Trading-Algo names or dependencies.
- Do not reinterpret historical trusted or cached records as penalties.

## Approaches Considered

### Approach A: Projection Protocol and Compiled Active Space

Add an optimizer-neutral protocol and a practical implementation for named `GeneSpace` values. Future declarative spaces compile into the same protocol. This preserves `GeneSpace` v1, works across optimizers and external stores, and creates a stable future compilation target. Users with complex domains must provide hooks until a DSL exists.

### Approach B: Declarative Structured-Space DSL Now

Add categorical, conditional, hierarchical, and dependent parameters immediately. This is ergonomic but freezes broad hashing, serialization, checkpoint, and adapter semantics before enough integrations validate them.

### Approach C: Optimizer-Specific Hooks

Add activation and repair callbacks separately to GA, DE, and CMA. This is initially shorter but duplicates semantics, prevents one canonical cache identity, and obstructs future session orchestration.

## Recommendation

Use Approach A. Ship `ActiveGeneProjection`, portable transforms, constraint semantics, and CMA integration now. Add a declarative DSL later after multiple systems exercise this compilation target.

## Package Architecture

```text
evocore/
  search_space/
    constraints.py        # violations, repair and validation protocols
    projection.py         # protocol, result, snapshot, built-in projection
    transforms.py         # portable coordinate transforms
  lifecycle/
    records.py            # additive confidence semantics
    telemetry.py          # constraint-penalty accounting
    checkpointing.py      # additive record literal support
  optimizers/
    cmaes/
      mixed.py            # integer strategies and margin behavior
      projection.py       # projected means and record conversion
      restarts.py          # restart policies and fresh-run construction
```

Package `__init__.py` files re-export public names only. Exact implementation splits may change during planning to keep modules focused.

## Named Parameter Identity

Projection-aware workflows require unique source gene names. Existing optimizer APIs continue to accept unnamed spaces.

- Names are canonical identifiers, not labels.
- Active order derives from source `GeneSpace` order, not caller collection order.
- Hierarchical-looking names such as `entry.fast_period` remain opaque strings.
- Structural bindings may contain JSON-safe scalar values including strings, booleans, numbers, and `None`.
- Optimizer values remain `GeneValue` instances supported by `GeneSpace`.
- Duplicate or unknown names, missing reconstruction values, and incompatible outputs raise `ConfigurationError` before evaluation.

The long-term model deliberately separates named domain identity from ordered optimizer vectors.

## Projection Contract

The public structural protocol has this conceptual shape:

```python
@runtime_checkable
class ParameterProjection(Protocol):
    optimizer_space: GeneSpace
    checkpointable: bool

    def project(self, parameters: Mapping[str, object]) -> ProjectionResult: ...
    def reconstruct(self, values: Sequence[GeneValue]) -> ProjectionResult: ...
    def signature(self) -> Mapping[str, object]: ...
    def snapshot(self) -> ProjectionSnapshot: ...
    def value_hash(self, parameters: Mapping[str, object]) -> str: ...
```

`ActiveGeneProjection` is configured with a named source `GeneSpace`, active names, structural or inactive base values, transforms, repair and validation hooks, schema ID and version, and optional identity keys. It canonicalizes active names into source order and builds an optimizer `GeneSpace` containing only active coordinates.

### `ProjectionResult`

The immutable result exposes canonical reconstructed parameters, ordered optimizer values, active names, structural bindings, repair records, violations, projection and parameter hashes, JSON-safe metadata, validity, and portability. Returned collections are detached from mutable projection state.

### `ProjectionSnapshot`

The JSON-safe snapshot includes projection schema version, user schema ID/version, source and optimizer `GeneSpace` signatures and hashes, active names, structural bindings, transform and hook signatures, identity keys, portability, and final signature hash. It describes behavior but never serializes executable code.

On resume, applications recreate hooks and projections; EvoCore compares the recreated signature with the persisted snapshot before accepting paired optimizer state.

## Transform Contract

`ParameterTransform` maps optimizer coordinates and domain values:

```python
class ParameterTransform(Protocol):
    checkpointable: bool

    def decode(self, value: GeneValue) -> object: ...
    def encode(self, value: object) -> GeneValue: ...
    def signature(self) -> Mapping[str, object]: ...
```

`encode()` raises a clear projection error for intentionally one-way transforms. Such transforms can reconstruct evaluator parameters but cannot project that field into a historical CMA warm start.

Phase 3 includes portable identity, threshold/binary, exponential/logarithmic integer, and output-renaming transforms. Cross-parameter derived values belong in a full-mapping normalization hook after reconstruction.

## Repair, Validation, and Portability

The deterministic reconstruction pipeline is:

1. Validate vector length and primitive values.
2. Apply existing codec repair semantics.
3. Decode coordinate transforms.
4. Merge structural and inactive base values.
5. Run declared full-mapping repair or normalization hooks in order.
6. Validate constraints.
7. Compute canonical identity and metadata.

`RepairRecord` stores parameter name, previous and repaired values, reason code, hook identity, and metadata. `ConstraintViolation` stores code, message, affected names, hook identity, and metadata.

Portable hooks provide stable IDs and versions, participate in signatures, and support snapshots, checkpoints, and cross-run caches. Runtime-only callables remain usable locally but set `checkpointable=False`, cannot claim portable cache identity, and fail early during snapshot or checkpoint preparation. EvoCore never hashes `repr(callable)` as reproducible identity.

## Canonical Hashing

A projected hash includes projection signature, behavior-affecting structural bindings, canonical transformed active values, and configured identity fields produced by normalization. It excludes inactive values, display metadata, job IDs, fold or dataset identity unless composed externally, and repair narration that does not change canonical values.

Therefore inactive-only differences hash equally, structural-template changes hash differently, and transform or hook version changes invalidate the projection hash. `GeneSpace.value_hash()` remains unchanged.

## Constraint-Penalty Records

Add `"constraint_penalty"` to `EvaluationConfidence` and separate trusted evidence from optimizer update eligibility:

```python
TRUSTED_CONFIDENCES = ("trusted_full", "cached")
STATE_UPDATE_CONFIDENCES = (
    "trusted_full",
    "cached",
    "constraint_penalty",
)
```

A penalty record has a finite direction-aware score and zero cost, completes GA/DE/CMA batches, is state-update eligible, and marks the candidate eliminated rather than trusted. Default trusted snapshots, top-k results, archives, promotions, search-memory exports, and warm starts exclude it. Metadata includes violations, repairs, projection identity, canonical hash, and penalty-policy signature. Telemetry counts it separately from rejection and evaluation.

The default flow repairs, validates, rejects invalid external seeds before optimizer state, and penalizes irreparable optimizer-generated candidates. Resampling remains an explicit external policy because it can bias search, change batch semantics, and loop indefinitely.

`WarmStartRecord` continues to accept only `TRUSTED_CONFIDENCES`; expanding state eligibility must not admit penalties into historical trusted state.

## Active-Subspace CMA

CMA continues to operate on a flat numeric `GeneSpace` produced by a projection:

- active floats are direct CMA coordinates;
- active native integers use the configured integer strategy;
- fixed numeric genes are removed and reconstructed;
- booleans, categoricals, and inactive parameters are structural or externally selected;
- float coordinates decoded into toggles or integers remain float CMA coordinates governed by transforms.

The projection, not `CMAESOptimizer`, owns domain reconstruction.

### Projected Warm Starts

Focused helpers project full historical parameters into active values, reject structural-template mismatches, report non-invertible transforms, construct pre-start means through `best` or `top_k_centroid`, and preserve provenance. They never mutate covariance state after CMA begins.

## Native Integer CMA Strategies

Add configuration conceptually equivalent to:

```python
CMAESOptimizer(
    gene_space,
    ...,
    integer_strategy="round",
    integer_min_probability=0.02,
)
```

### `round`

This preserves current behavior: continuous latent samples are rounded and clamped for user-facing integer values. It remains the default and preserves existing config and checkpoint behavior when new options are omitted.

### `margin`

Margin handling applies protected discrete sampling to each active native integer coordinate. It uses `IntegerMarginDistribution` semantics, stays inside bounds, and prevents alternatives from collapsing to zero sampling probability. Randomness derives deterministically from optimizer seed, batch/event identity, candidate position, and coordinate identity. Original continuous latent samples remain indexed by candidate ID for Rust CMA state updates.

Strategy and margin configuration participate in config hashes, checkpoints, state summaries, and reproducibility metadata. Every additional state element required for exact resume is persisted or deterministically derived.

The implementation may minimally extend Rust/PyO3 CMA state to expose coordinate statistics required for correct sampling. Public Rust-backed changes update `evocore/_core.pyi`; lifecycle and reconstruction remain in Python. Invalid probability configurations fail at construction, and the implementation must document or reject unsupported large integer ranges rather than hide a small-range assumption.

Changing the default to `margin` is outside Phase 3 and requires benchmark evidence plus an explicit compatibility decision.

## Lifecycle-Managed CMA Restarts

Restarts create fresh optimizer instances rather than resetting one in place. Public concepts include `CMAESRestartPolicy`, `CMAESRestartDecision`, a planner/factory helper, and a snapshot if policy state persists.

A decision contains restart index/reason, deterministic child seed, projected mean strategy and sources, sigma, population size, projection signature, and parent/archive lineage. Initial policies are fixed-size and IPOP-style growth; BIPOP and portfolios are deferred.

Restarts are forbidden with pending batches. Each restart receives fresh CMA state and an ordinary checkpoint. Trusted elites enter through existing warm starts. Existing composition helpers derive child seeds using restart index and projection identity. Events and results preserve the chain; archives remain user-owned.

## Checkpoint and Compatibility Model

### Existing Flat Workflows

- `GeneSpace` schema remains version 1.
- Existing flat constructors and unnamed spaces remain unchanged when Phase 3 options are omitted.
- `integer_strategy="round"` preserves default behavior and old seed semantics.
- Existing checkpoint fixtures remain readable.

### Projection Workflows

- Projection snapshots use a separate schema.
- Durable workflows persist optimizer checkpoint plus projection snapshot.
- Resume recreates hooks and validates the full signature.
- Runtime-only projections fail before claiming checkpoint support.
- Active-order, binding, transform, hook, or identity mismatches produce clear errors.

### Integer-Margin Workflows

- Strategy and minimum probability participate in optimizer config hashes.
- New readers accept existing round-only checkpoints.
- Round-only payload shape remains unchanged where practical.
- Margin-specific state uses an explicit version or state kind.
- Old EvoCore versions need not read new margin checkpoints.
- Resumed margin runs reproduce uninterrupted future candidates and updates.

### Additive Confidence Risk

The new confidence is runtime-additive but can affect external exhaustive matches or assumptions that every state-update record is trusted. Docs and changelog must call this out. Public helpers should encourage semantic checks instead of raw tuple matching.

## Telemetry and Events

Candidate and record metadata may include projection schema/signature, canonical hash, active names, structural source, transform provenance, repair/violation summaries, candidate source, external job ID, and restart lineage. Large snapshots are persisted once; ordinary events carry hashes and concise summaries.

Telemetry distinguishes trusted, cached, penalty, rejected, repaired, invalid-seed, projection-failure, and restart counts, plus integer strategy and margin configuration.

## Public API Changes

Additive public APIs are required for projection/result/snapshot types, transforms, repair and validation contracts, trusted-confidence helpers, `constraint_penalty`, penalty policy/helpers, CMA integer configuration, projected warm-start helpers, restart policies/decisions, and relevant telemetry summaries.

Internal changes are required in GA/DE/CMA batch completion, candidate status, checkpoint literal validation, CMA sample bookkeeping, config hashing, and telemetry/event accounting.

Docs and examples alone cover the Trading-Algo-style orchestration recipe, dataset/fold cache-key composition, runtime-only experimentation, staged archive/refinement composition, and deterministic restart orchestration.

## Delivery Slices

### Phase 3A: Projection and Transform Foundation

- Add projection contracts, `ActiveGeneProjection`, result and snapshot types.
- Add portable transforms and two-tier hook portability.
- Add deterministic hashing and snapshot validation.

### Phase 3B: Constraints and Evaluation Semantics

- Add violations, repairs, penalty policy, and record helper.
- Separate trusted and state-update confidence helpers.
- Update GA, DE, CMA, telemetry, events, and checkpoints.
- Prove penalties complete batches without entering trusted archives.

### Phase 3C: Active and Mixed-Numeric CMA

- Add projected historical-value and mean helpers.
- Integrate real `round` and `margin` strategies.
- Add deterministic margin sampling and resume.
- Add fixed and IPOP restart planning.
- Preserve round-only behavior and fixtures.

### Phase 3D: External Integration Recipe

- Demonstrate a template-controlled outer GA/inner CMA flow.
- Include binary and log-integer transforms, cached records, projected warm starts, family quotas, specialist caps, random immigrants, stage-1 archive, stage-2 refinement, and restart lineage.
- Document reproducibility and checkpoint ownership.

Each slice receives its own implementation plan and reviewable change set. CMA consumes projection and constraint contracts rather than creating optimizer-local equivalents.

## Focused Test Plan

### Projection and Identity

- Reject unnamed, duplicate, or unknown active names; canonicalize source order.
- Round-trip named values without mutation leaks.
- Prove inactive hash invariance and structural hash separation.
- Prove transform/hook versions alter signatures.
- Round-trip snapshots and reject recreated-signature mismatches.

### Transforms and Hooks

- Test identity, threshold boundaries, exponential integer behavior, bounds, and renaming.
- Report one-way inverse failures clearly.
- Record ordered dependent repairs and provenance.
- Verify portable resume, hook mismatch rejection, runtime-only local use, and runtime-only checkpoint rejection.

### Constraints and Penalties

- Verify repair-before-validation and invalid external-seed rejection.
- Test finite maximize/minimize penalties and zero cost.
- Complete GA, DE, and CMA batches with penalties.
- Keep candidate status eliminated and exclude penalties from top-k, archive, promotion, and warm starts.
- Round-trip penalty metadata and telemetry.

### Active CMA and Warm Starts

- Remove fixed/inactive coordinates and reconstruct complete parameters.
- Project historical parameters into active means.
- Reject template mismatches and report non-invertible transforms.
- Preserve latent samples after user-facing decoding.

### Integer Margin

- Test normalization, probability floors, invalid configuration, and bounded sampling.
- Preserve nonzero alternatives under concentrated state.
- Test mixed float/integer ask/tell and equal-seed determinism.
- Prove uninterrupted and resumed runs generate identical future candidates and summaries.
- Regress round behavior and config-hash differences.
- Add representative mixed-numeric convergence tests.

### Restarts

- Test fixed size, IPOP growth, stable unique child seeds, projected best/centroid means, pending-batch rejection, snapshot round trip, and parent-state isolation.

### External Integration and Regression

- Exercise a synthetic template-controlled outer/inner workflow with transforms, cache hits, matching-template warm starts, family/specialist policies, staged refinement, penalties, and deterministic restarts.
- Run existing unit/integration suites and checkpoint fixtures.
- Add property tests for invertible projection round trips, inactive hash invariance, repair idempotence, and margin bounds/normalization.
- Run strict MkDocs and public import checks.

## Backward Compatibility Risks

### Additive Confidence

External exhaustive literal matches may break or misclassify penalties. Add semantic helpers, changelog notes, migration examples, and candidate-status tests.

### Projection Hash Stability

Hook or transform changes intentionally invalidate caches. Require explicit schema/version identity and expose snapshots for diagnosis.

### Runtime-Only Hooks

Users may assume callables are durable. Expose `checkpointable`, fail early, and never use unstable callable representations.

### Integer-Margin State

New sampling changes sequences and payloads. Keep `round` default, include strategy in config identity, version margin state, and test exact resume.

### CMA Rust Boundary

Correct coordinate statistics may require PyO3 changes. Keep the extension minimal, update stubs/fixtures, and retain Python ownership of lifecycle and reconstruction.

### Trading-Algo Overfitting

Keep templates, folds, and strategy fields in recipes. Public APIs use generic bindings, transforms, hooks, and lineage.

## Acceptance Criteria

Phase 3 is complete when:

- Named conditional parameters compile into a stable active `GeneSpace` without private-state access.
- Projections round-trip, hash, snapshot, repair, and validate deterministically.
- Portable and runtime-only hooks have honest durability semantics.
- Penalties complete GA/DE/CMA batches without becoming trusted evidence.
- CMA warm-starts from projected historical parameters.
- Native integers support real opt-in margin sampling with exact checkpoint resume.
- Fixed and inactive values reconstruct outside CMA.
- Fixed and IPOP planners create fresh deterministic runs.
- A Trading-Algo-style recipe expresses templates, active coordinates, decoders, caches, diversity, archives, refinement, and lineage through public APIs.
- Old flat workflows, round defaults, seeds, and checkpoint fixtures remain valid.
- Focused unit, integration, property, docs, and regression checks pass.

## Deferred Roadmap

- Declarative structured/hierarchical DSL compiled into `ParameterProjection`.
- Native categorical genes and categorical CMA.
- BIPOP and restart portfolios.
- Optional `OptimizationSession`.
- Async partial-batch orchestration and durable job adapters.
- Persistence and telemetry sink protocols.
- Optimizer plugin and adapter registration.

These features should build on Phase 3 rather than reopen domain identity, projection, or mixed-numeric checkpoint semantics.
