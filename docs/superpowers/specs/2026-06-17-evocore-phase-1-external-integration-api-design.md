# EvoCore Phase 1 External Integration API Design

**Date:** 2026-06-17
**Status:** Approved for implementation planning
**Scope:** Shared public external-state API for GA, DE, and CMA-ES ask/tell workflows

## Summary

Phase 1 makes EvoCore easier to use as the primary optimizer package for real
external expensive black-box systems. The work adds a shared public API shape
for warm starts, candidate injection, read-only candidate snapshots, top-k
candidate access, and cached evaluation helpers across
`GeneticAlgorithmOptimizer`, `DifferentialEvolutionOptimizer`, and
`CMAESOptimizer`.

The design builds on the current vNext foundation: `Candidate`,
`EvaluationRecord`, explicit confidence semantics, append-only events,
`OptimizationTelemetry`, `BudgetPolicy`, and ask/tell checkpoints. The goal is
not to introduce a new optimizer architecture. The goal is to turn existing
mechanics into stable integration surfaces so external systems do not need to
touch private optimizer state.

Trading-Algo-Scalper-Gold is the reference workload, but no trading-specific
code belongs in EvoCore. Concepts such as search memory, cached backtests,
seed pools, staged refinement, family diversity, and hybrid GA/CMA workflows
must remain expressible through generic candidate metadata, records, and
policies.

## Context

The current codebase already supports most low-level pieces:

- `Candidate` and `EvaluationRecord` carry lifecycle identity, scores,
  confidence, cost, metrics, and metadata.
- `cached` records are state-eligible alongside `trusted_full` records.
- GA keeps a trusted population internally.
- DE keeps target slots and trial state internally.
- CMA-ES stores Rust-backed covariance state and pending batches internally.
- Stable ask/tell checkpoints persist candidates, batches, telemetry, events,
  and best candidate IDs.
- Documentation already explains asynchronous-friendly ask/tell, cached
  confidence semantics, and external queue checkpoint examples.

The missing layer is a public, optimizer-neutral integration surface for
external memory and reporting workflows:

- initialize from known good historical candidates,
- inspect current trusted and scored candidates,
- inject domain-generated or archive candidates,
- convert cache hits into valid evaluation records,
- preserve external metadata across events, checkpoints, snapshots, and exports.

## Goals

- Define one shared public API shape for GA, DE, and CMA-ES.
- Implement the shared shape for all three optimizers with honest
  optimizer-specific semantics.
- Preserve deterministic seed behavior and checkpoint compatibility.
- Keep snapshots read-only so external users cannot mutate optimizer internals.
- Preserve domain metadata across ask/tell, events, checkpoints, snapshots,
  and result exports.
- Make cached evaluations ergonomic without changing their budget semantics.
- Keep Phase 2 archive/diversity policies and Phase 3 hierarchical/CMA restart
  work out of this implementation slice.

## Non-Goals

- Do not add family quota, novelty, archive, or specialist-cap policy behavior
  in Phase 1.
- Do not redesign `GeneSpace` for conditional or hierarchical search spaces.
- Do not add arbitrary post-start CMA-ES covariance injection.
- Do not add CMA-ES restart strategies.
- Do not expose live private optimizer collections.
- Do not add Trading-Algo-specific classes, names, or dependencies.
- Do not introduce old public vocabulary such as `Engine`, `RunResult`, `Rung`,
  `TellResult`, `Individual`, `Population`, `history`, or `engine_type`.

## Public API Shape

Add a shared lifecycle-level external state surface in
`evocore.lifecycle.external`, re-exported from `evocore.lifecycle` and from the
top-level `evocore` package.

The public surface should include:

```python
@dataclass(frozen=True)
class WarmStartRecord:
    values: tuple[GeneValue, ...] | None
    params: Mapping[str, GeneValue] | None
    score: float
    confidence: EvaluationConfidence = "cached"
    stage: str = "warm_start"
    cost: float = 0.0
    metrics: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateSnapshot:
    candidate_id: str
    candidate_hash: str
    values: tuple[GeneValue, ...]
    params: Mapping[str, GeneValue] | None
    origin: CandidateOrigin
    batch_id: str
    event_index: int
    generation: int | None
    status: CandidateStatus
    stage: str | None
    confidence: EvaluationConfidence | None
    score: float | None
    scores: Mapping[str, ScoreObservation]
    cost: float
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class PopulationSnapshot:
    optimizer_type: str
    direction: Direction
    event_index: int
    pending_batch_ids: tuple[str, ...]
    trusted_count: int
    candidates: tuple[CandidateSnapshot, ...]
    telemetry: OptimizationTelemetry
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ExternalStateCapabilities:
    warm_start_before_ask: bool
    warm_start_after_ask: bool
    proposed_candidate_injection: bool
    state_candidate_injection: bool
    tracked_only_injection: bool
    population_snapshots: bool
    top_candidate_snapshots: bool
    cached_record_helpers: bool


@dataclass(frozen=True)
class InjectionResult:
    accepted: tuple[CandidateSnapshot, ...]
    skipped_duplicates: tuple[CandidateSnapshot, ...]
    rejected: tuple[Mapping[str, object], ...]
    acceptance_decisions: tuple[AcceptanceDecision, ...] = ()
```

The exact implementation can split these types into smaller files if needed,
but they should remain part of one coherent lifecycle-level API surface.

Update the optimizer protocol surface with a structural protocol:

```python
class ExternalStateOptimizer(Optimizer, Protocol):
    def external_state_capabilities(self) -> ExternalStateCapabilities: ...
    def warm_start(
        self,
        records: Sequence[WarmStartRecord],
        *,
        deduplicate: bool = True,
        mode: Literal["state", "tracked"] = "state",
        cma_mean_strategy: Literal["best", "top_k_centroid"] = "best",
        top_k: int | None = None,
    ) -> UpdateResult: ...
    def inject_candidates(
        self,
        records: Sequence[WarmStartRecord],
        *,
        origin: CandidateOrigin = "memory_seed",
        mode: Literal["proposed", "tracked"] = "proposed",
        deduplicate: bool = True,
        metadata: Mapping[str, object] | None = None,
    ) -> InjectionResult: ...
    def candidate_snapshot(
        self,
        *,
        scope: Literal["trusted", "known", "pending", "scored"] = "trusted",
    ) -> PopulationSnapshot: ...
    def top_candidates(
        self,
        k: int = 10,
        *,
        scope: Literal["trusted", "known", "pending", "scored"] = "trusted",
        confidence: tuple[EvaluationConfidence, ...] = ("trusted_full", "cached"),
    ) -> tuple[CandidateSnapshot, ...]: ...
```

The protocol is a contract for external integrations. The optimizers do not
need to share identical internal state transitions.

## Shared Method Semantics

### `warm_start(...)`

`warm_start(...)` initializes optimizer state from prior trusted candidates,
search-memory records, archived elites, or known good parameter sets.

Rules:

- Each `WarmStartRecord` must resolve to decoded values through either
  `values` or `params`.
- Values are validated and repaired through existing `GeneSpace` and codec
  semantics.
- Warm-start records must have state-eligible confidence:
  `trusted_full` or `cached`.
- `mode="state"` updates optimizer state according to optimizer-specific rules.
- `mode="tracked"` records candidates for events, snapshots, and checkpoints
  without changing optimizer state.
- Historical candidate IDs are not reused. EvoCore assigns fresh current-run
  candidate IDs and stores historical provenance in metadata.
- Duplicate decoded values are detected by `GeneSpace.value_hash(...)`.
- Default duplicate behavior is to skip duplicates and report them.
- Accepted warm-start candidates emit lifecycle events and update telemetry.
- Metadata must be JSON-safe so checkpointing fails early and clearly.

### `inject_candidates(...)`

`inject_candidates(...)` adds current-run candidates from external sources such
as random immigrants, domain-generated proposals, repaired candidates, archive
seeds, or search-memory seeds.

Rules:

- Injection creates current-run candidates, not historical identity aliases.
- Injected candidates are proposed by default. They are not trusted unless
  later evaluated through `tell(...)` or passed through `warm_start(...)`.
- `origin` should use existing allowed `CandidateOrigin` values where possible,
  especially `memory_seed`, `random`, `surrogate_proposal`, and `restart`.
- Finer provenance such as `archive_elite`, `domain_seed`, `random_immigrant`,
  or `repaired` belongs in metadata in Phase 1.
- Duplicate handling mirrors warm start.
- Unsupported optimizer modes fail with explicit `ConfigurationError` messages.

### `candidate_snapshot(...)`

`candidate_snapshot(...)` returns read-only copies of optimizer candidate state.
It must never return live optimizer-owned `Candidate` objects.

Recommended scopes:

- `"trusted"`: state-eligible candidates that can influence optimizer state.
- `"known"`: all candidates known to the ask/tell ledger.
- `"pending"`: candidates in pending batches.
- `"scored"`: candidates with at least one score observation.

### `top_candidates(...)`

`top_candidates(...)` returns read-only candidate snapshots sorted by
direction-aware comparison score.

Defaults:

- `k=10`
- `scope="trusted"`
- `confidence=("trusted_full", "cached")`

Partial and surrogate scores are excluded by default. Callers may opt into
other confidence filters for reporting, but non-state scores must not become
optimizer best state through this method.

### Cached Evaluation Helpers

Add lifecycle-level helpers for current-run cache hits:

```python
cached_records(
    candidates: Sequence[Candidate],
    cache: Mapping[str, object] | Callable[[Candidate], object | None],
    *,
    gene_space: GeneSpace,
    stage: str = "full",
    cost: float = 0.0,
    metadata: Mapping[str, object] | None = None,
    key: Callable[[Candidate, GeneSpace], str] | None = None,
) -> tuple[EvaluationRecord, ...]
```

The helper maps current candidates to `EvaluationRecord(confidence="cached")`
when cache hits are present.

Rules:

- Cache lookup uses `candidate.candidate_hash(gene_space)` unless a caller
  supplies a key strategy.
- Returned records carry cache metadata such as cache key, cache reason,
  evaluator version, source run, or fingerprint.
- Cached records remain state-eligible.
- Cached records do not spend fresh full-evaluation budget.
- The helper must not call `tell(...)` itself.

## Optimizer-Specific Semantics

### Genetic Algorithm

GA has the most complete Phase 1 behavior.

- `warm_start(...)` adds accepted records to the trusted population.
- The trusted population is sorted by direction-aware state score and trimmed to
  `population_size`.
- `best_candidate`, telemetry, events, and checkpoints reflect accepted
  warm-start candidates.
- The next `ask(...)` uses the warm-start trusted population for reproduction.
- `inject_candidates(...)` creates proposed candidates for explicit evaluation
  or for a documented injection mode that makes them available to the next
  ask/tell cycle.
- Snapshots expose trusted, known, pending, and scored candidate scopes.

### Differential Evolution

DE must preserve target-slot semantics.

- `warm_start(...)` fills target population slots before normal trial
  generation.
- If warm-start records exceed `population_size`, records are ranked by
  direction-aware score and trimmed deterministically.
- `cached` and `trusted_full` records can initialize targets.
- After target initialization, arbitrary state injection is not the default.
- Post-initialization injection may create proposed external candidates only in
  an explicit mode; replacement of target slots must be reported through
  `AcceptanceDecision`.
- Snapshots expose target population, pending trials, known candidates, and
  top-k scored candidates without exposing private slot collections.

### CMA-ES

CMA-ES must keep covariance semantics honest.

- Phase 1 supports pre-ask warm start.
- Before the Rust CMA state exists, `cma_mean_strategy="best"` derives the
  initial mean from the best state-eligible record.
- `cma_mean_strategy="top_k_centroid"` derives the initial mean from the
  direction-aware top `top_k` state-eligible records. `top_k` defaults to all
  accepted warm-start records when omitted.
- Once CMA state exists, `warm_start(...)` must reject state mutation in Phase 1
  unless the call is explicitly tracked-only.
- Arbitrary post-start candidate injection must not silently update covariance.
- Tracked-only injected candidates may appear in snapshots and events, but CMA
  state updates still require valid CMA batches of state-eligible records.
- Snapshots expose known candidates, pending batches, best scored candidates,
  and generation/state summary. They do not expose mutable Rust state internals.

## Metadata Contract

Phase 1 relies on metadata rather than expanding closed enum values.

Metadata should support external fields such as:

- fold ID,
- family name,
- decoded strategy summary,
- cache reason,
- cache key,
- source run ID,
- candidate source,
- external job ID,
- evaluator version,
- repair reason,
- archive ID.

Metadata must be copied into:

- candidates,
- evaluation records,
- score observations,
- events,
- snapshots,
- ask/tell checkpoints,
- result exports where candidates become solutions.

The implementation should validate JSON-safety before storing metadata in
checkpointed state.

## Checkpoint Contract

Phase 1 should preserve existing checkpoint schema compatibility where possible.

- Do not add new `CandidateOrigin` literals in Phase 1 unless checkpoint
  fixtures are intentionally updated.
- Store fine-grained source labels in metadata.
- Warm-started and injected candidates must round-trip through ask/tell
  checkpoints.
- Restored optimizers must produce the same next ask sequence as uninterrupted
  optimizers when state mutation was supported by that optimizer.
- Existing golden fixtures remain valid.

If an implementation requires new checkpoint fields, it should add optional
fields only. Required schema changes are out of scope for Phase 1 unless the
implementation plan explicitly calls them out.

## Error Handling

- Unknown or unsupported injection modes raise `ConfigurationError`.
- Non-state warm-start confidence raises `ConfigurationError`.
- Missing values and params raise `ConfigurationError`.
- Providing both values and params is allowed only if they resolve to the same
  decoded values; otherwise raise `ConfigurationError`.
- Non-finite scores raise `FitnessError` through existing
  `EvaluationRecord` validation.
- Duplicate warm-start or injected values are skipped by default and reported.
- Non-JSON-safe metadata raises a clear configuration error before checkpoint
  save.
- CMA post-start state mutation raises `ConfigurationError` in Phase 1.

## Tests

### Shared Contract Tests

- GA, DE, and CMA-ES satisfy `ExternalStateOptimizer` at runtime.
- `WarmStartRecord` accepts values and params and rejects invalid value shapes.
- State-eligible confidences are required for state warm starts.
- Duplicate detection uses `GeneSpace.value_hash(...)`.
- `CandidateSnapshot` and `PopulationSnapshot` are immutable copies.
- `top_candidates(...)` sorts correctly for maximize and minimize.
- `top_candidates(...)` excludes partial and surrogate scores by default.
- Cached helper records use `confidence="cached"`.
- Cached helper metadata is preserved.
- Cached records do not increment fresh full-evaluation budget by themselves.

### GA Tests

- Warm start fills trusted population, trims to `population_size`, and updates
  best candidate.
- The next GA `ask(...)` uses warm-start state.
- Injected candidates preserve metadata and ask events.
- Snapshots expose trusted population without exposing
  `_trusted_population_vnext`.
- Warm-started candidates round-trip through ask/tell checkpoints.

### DE Tests

- Warm start fills target slots deterministically.
- Excess warm-start records are ranked and trimmed.
- Cached warm-start records initialize targets.
- Post-initialization state injection rejects by default or reports explicit
  acceptance decisions in an approved mode.
- Warm-started targets round-trip through ask/tell checkpoints.

### CMA-ES Tests

- Pre-ask warm start derives initial mean deterministically.
- Best-record mode and top-k centroid mode produce expected initial means.
- Warm start after first `ask(...)` raises `ConfigurationError`.
- Tracked-only injection does not update Rust CMA state.
- Snapshots expose known candidates and best scored candidates without exposing
  mutable Rust state.
- Warm-started initial state round-trips through ask/tell checkpoints.

### Compatibility Tests

- Existing golden checkpoint fixtures still pass.
- Warm-started and injected candidates preserve metadata after checkpoint
  restore.
- Restored telemetry includes cached, trusted, and proposed counts.
- Existing ask/tell behavior without warm starts or injection is unchanged.

## Documentation

Update user docs as part of Phase 1 implementation:

- Expand `docs/site/ask-tell-engines.md` with warm start, injection,
  snapshots, and top-k examples.
- Expand `docs/site/budget-aware-optimization.md` with cached evaluation
  accounting.
- Add an expensive external evaluation recipe page covering cached backtests,
  async workers, warm starts from archives, top-k survivor export, and
  deterministic resume.
- Add API reference entries for new lifecycle types and helper functions.
- Include a small Trading-Algo-style example without importing or depending on
  Trading-Algo.

## Rollout

Implementation should proceed in reviewable slices:

1. Add shared public types, protocol, snapshot conversion helpers, and cached
   record helpers.
2. Implement GA warm start, injection, snapshots, and top-k.
3. Implement DE warm start, conservative injection, snapshots, and top-k.
4. Implement CMA-ES pre-ask warm start, tracked-only injection, snapshots, and
   top-k.
5. Add docs, examples, changelog, and focused compatibility tests.

The implementation plan may split this into multiple PRs if one branch becomes
too large, but the public design should remain one coherent Phase 1 API.

## Acceptance Criteria

- GA, DE, and CMA-ES expose the shared external-state API.
- External users can warm start all three optimizers without private state
  access.
- External users can inspect trusted, known, pending, scored, and top-k
  candidates through read-only snapshots.
- Cache hits can be converted to `cached` records with preserved metadata.
- Metadata survives ask/tell, checkpoints, snapshots, and result exports.
- CMA-ES does not accept unsafe post-start covariance injection.
- Existing checkpoint fixtures remain compatible.
- Documentation includes expensive external evaluation recipes.
- No Trading-Algo-specific API enters EvoCore.

## Future Work

Phase 2 should add reusable archive, diversity, family quota, specialist cap,
stall policy, and hybrid composition utilities.

Phase 3 should address repair hooks, dependent parameters, hierarchical spaces,
CMA restart semantics, fixed-gene reconstruction, integer margin behavior, and
mixed or active subspace support.
