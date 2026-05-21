# EvoCore Optimizer Lifecycle Protocols Design

**Date:** 2026-05-13
**Status:** Draft approved for specification
**Scope:** Clean-break public ask/tell lifecycle contract for single-objective EvoCore optimizers

## Summary

EvoCore should stabilize its next public API around a clean-break, protocol-based
optimizer lifecycle. The first stabilization slice defines the ask/tell contract that
future optimizers, evaluators, schedulers, advisors, and asynchronous execution systems can
rely on.

This design intentionally does not preserve legacy lambda-first `run(...)` usage as an
acceptance criterion. Current engines may keep compatibility temporarily during rollout, but
the public product direction is a structural protocol API with explicit candidate,
evaluation, batch, confidence, and error semantics.

The slice stays single-objective. Multi-objective optimization, constraints, result
serialization, and convenience `optimize(...)` wrappers are deferred until the lifecycle
contract is stable.

## Product Direction

EvoCore becomes a general-purpose evolutionary optimization package with explicit lifecycle
objects instead of ad hoc engine signatures. The core user model is:

```python
candidates = optimizer.ask(16)
records = evaluator.evaluate(candidates, context)
summary = optimizer.tell(records)
```

Concrete engines such as `GAEngine` and `CMAESEngine` implement the same structural
contract. External evaluators and future optimizers can participate without subclassing
EvoCore base classes.

## Goals

- Define public structural protocols for optimizers and evaluators.
- Stabilize candidate and evaluation record semantics for single-objective optimization.
- Make `candidate_id`, `batch_id`, rung, confidence, score, cost, metrics, and metadata
  explicit parts of the lifecycle.
- Preserve raw user scores while making maximize/minimize direction explicit.
- Define strict validation and error behavior for manual and orchestrated ask/tell flows.
- Make `GAEngine` and `CMAESEngine` conform to the lifecycle contract.
- Keep EvoCore free of domain-specific evaluator logic.

## Non-Goals

- Do not implement multi-objective or Pareto behavior in this slice.
- Do not add constraint modeling beyond metadata and rejected records.
- Do not redesign result history, serialization, JSON export, or pandas export.
- Do not add the public `optimize(...)` convenience wrapper yet.
- Do not make old lambda-based `run(fitness_fn)` compatibility an acceptance criterion.
- Do not add trading-specific metrics, validation, or evaluator semantics.

## Public API Shape

Add an `evocore.protocols` module for structural protocols. Protocols should describe
required behavior without forcing inheritance.

```python
from collections.abc import Sequence
from typing import Literal, Protocol

class Optimizer(Protocol):
    direction: Literal["maximize", "minimize"]

    def ask(self, n: int | None = None) -> Sequence[Candidate]:
        ...

    def tell(self, records: Sequence[EvaluationRecord]) -> TellResult:
        ...

    def state_summary(self) -> EngineStateSummary:
        ...

class Evaluator(Protocol):
    def evaluate(
        self,
        candidates: Sequence[Candidate],
        context: EvaluationContext,
    ) -> Sequence[EvaluationRecord]:
        ...
```

The protocol set should stay small. Avoid adding broad abstract base classes or many helper
protocols until a concrete extension need appears.

## Lifecycle Records

Concrete lifecycle records live in `evocore.evaluation` unless a later implementation
finds a cleaner local module split.

### Candidate

`Candidate` represents one proposed point in the search space.

Required stable concepts:

- `candidate_id`: deterministic identifier within an optimizer run.
- `batch_id`: deterministic token shared by candidates from one `ask()` call.
- `genes`: decoded gene values suitable for evaluator use.
- `params`: optional named parameter mapping.
- `origin`: proposal source such as random, crossover, mutation, CMA sample, surrogate
  proposal, memory seed, or restart.
- `parents`: optional parent candidate IDs for lineage.
- `event_index`: monotonic ask/tell event index.
- `generation`: optional generation number for generation-oriented policies.
- `status`: lifecycle state such as proposed, screened, racing, promoted, trusted,
  eliminated, or archived.
- `confidence`: latest evaluation confidence, if any.
- `rung`: latest evaluation rung, if any.
- `scores`: observed scores by rung.
- `cost`: accumulated evaluator-reported cost.
- `metadata`: user and engine diagnostics.

`candidate_hash()` returns a stable hash of decoded gene values. It is used for trial
accounting and duplicate genome telemetry, not for candidate identity.

### EvaluationRecord

`EvaluationRecord` is the only value accepted by `Optimizer.tell(...)`.

Required concepts:

- `candidate_id`
- `score`
- `confidence`
- `rung`

Optional concepts:

- `batch_id`
- `cost`
- `metrics`
- `metadata`

For confidence values other than `rejected`, `score` must be finite. A rejected record may
use `score=None`.

### EvaluationContext

Introduce `EvaluationContext` so evaluator signatures do not churn as EvoCore grows.

Suggested fields:

- `rung: Rung | None`
- `batch_id: str`
- `event_index: int`
- `direction: Literal["maximize", "minimize"]`
- `budget: float | None`
- `metadata: dict[str, Any]`

`run(evaluator)` and policy-driven orchestration should pass context to evaluators. Manual
ask/tell users may also construct contexts when integrating remote or asynchronous
evaluation systems.

### TellResult And EngineStateSummary

`tell(...)` should return a lightweight summary of what changed.

Suggested `TellResult` fields:

- accepted record count
- trusted, partial, surrogate, cached, and rejected counts
- changed best candidate ID and score, if any
- consumed batch IDs
- pending batch IDs
- telemetry snapshot or telemetry reference

`EngineStateSummary` should provide a stable read-only view of engine status without
exposing private ledgers. It can include best candidate, best score, event index, pending
batches, trusted count, and telemetry.

## Confidence Semantics

Confidence values remain explicit and general-purpose:

- `surrogate`: model estimate or advisor score. It may influence scheduling, but it does
  not update trusted optimizer state by default.
- `partial`: reduced-fidelity or partial evidence. It may influence promotion, but it does
  not update trusted optimizer state by default.
- `cached`: compatible reused result. A policy decides whether cached records count as
  trusted.
- `trusted_full`: full objective evidence suitable for default optimizer-state updates.
- `rejected`: structural or runtime rejection. No finite score is required.

Surrogate, partial, cached, and rejected records must never silently masquerade as trusted
full records.

## Direction And Score Handling

This slice remains single-objective. Public records keep the raw user score.

Each optimizer exposes:

```python
direction: Literal["maximize", "minimize"]
```

Engines should normalize comparisons internally through a small helper rather than mutating
stored scores. This keeps telemetry, records, and evaluator outputs honest while allowing
both minimization and maximization APIs.

Future multi-objective support should add an explicit objective record type or vector
field in a later slice. This design does not reserve behavior for Pareto ranking beyond
choosing names that will not block that addition.

## Validation And Error Handling

Invalid tell data should raise `FitnessError` unless a future policy explicitly introduces
non-raising rejection modes.

Required validation:

- Unknown `candidate_id` raises.
- Unknown explicit `batch_id` raises.
- Explicit `batch_id` that does not match the candidate raises.
- Duplicate record for the same candidate and rung raises.
- Non-finite score for non-`rejected` confidence raises.
- Invalid confidence value raises.
- Invalid or empty rung name raises.
- A consumed CMA batch receiving another trusted update raises.

`tell([])` should be a valid no-op returning an empty `TellResult`. This supports queue and
polling integrations without special casing empty result batches.

`ConfigurationError` remains appropriate for invalid optimizer, policy, or rung
configuration before evaluation starts.

## Engine Responsibilities

`GAEngine` and `CMAESEngine` should both conform to `Optimizer`.

Shared responsibilities:

- Generate deterministic candidate IDs and batch IDs.
- Maintain a private candidate and batch ledger.
- Accept partial `tell(...)` calls in any order.
- Reject duplicate candidate/rung records.
- Track telemetry for proposed, unique, screened, partial, full, promoted, eliminated, and
  cost accounting.
- Keep raw records and scores distinguishable by confidence.

GA-specific responsibilities:

- Update trusted population and best candidate from `trusted_full` records by default.
- Allow partial or surrogate records to affect scheduling only through explicit policy.
- Keep candidate status transitions consistent with accepted records.

CMA-specific responsibilities:

- Store continuous samples associated with each ask batch.
- Advance the distribution exactly once when a full trusted batch arrives.
- Preserve deterministic sample and score alignment regardless of record arrival order.
- Keep partial, surrogate, cached, and rejected records from updating the distribution by
  default.

## Evaluator Responsibilities

Evaluators consume candidates and return `EvaluationRecord` values. They should not mutate
candidate objects to report results.

For synchronous `run(evaluator)` orchestration, EvoCore should validate that evaluator
responses match assigned candidates for the requested rung. Missing, duplicate, unknown, or
batch-mismatched records raise `FitnessError`.

Manual ask/tell flows may report subsets of a batch across multiple calls. The optimizer
ledger, not the evaluator, decides when a batch is complete.

## Documentation And Migration

Docs should present the protocol lifecycle as the primary API.

Required docs updates for implementation:

- Refresh ask/tell docs around `Optimizer`, `Evaluator`, and `EvaluationContext`.
- Move examples away from lambda-first usage and toward evaluator objects or explicit
  ask/tell.
- Document direction handling and raw score preservation.
- Document validation failures and confidence semantics.
- Add a changelog entry for the clean-break API stabilization.

Legacy lambda-style examples may be removed or clearly marked as compatibility examples if
temporarily retained during rollout.

## Testing Strategy

Contract tests should lead implementation.

Required Python tests:

- `GAEngine` and `CMAESEngine` satisfy the `Optimizer` protocol shape under type checking
  or runtime protocol checks where practical.
- Example evaluators satisfy the `Evaluator` protocol shape.
- `ask()` assigns deterministic candidate IDs.
- `ask()` assigns one deterministic batch ID per ask call.
- `tell()` accepts partial batch records in any order.
- `tell()` returns a useful `TellResult`.
- `tell([])` returns a no-op `TellResult`.
- `tell()` rejects unknown candidates.
- `tell()` rejects unknown explicit batch IDs.
- `tell()` rejects explicit batch mismatches.
- `tell()` rejects duplicate candidate/rung records.
- `tell()` rejects non-finite non-rejected scores.
- `tell()` accepts rejected records with `score=None`.
- CMA consumes a trusted batch exactly once.
- GA updates trusted state only from `trusted_full` by default.
- Surrogate and partial records do not silently update trusted state.
- Direction handling chooses the correct best candidate for maximize and minimize engines.
- `EvaluationContext` passed through `run(evaluator)` contains correct rung, batch,
  direction, and event data.

Rust tests should be added only where public lifecycle work touches Rust-backed state or
batch handling. Otherwise this slice can remain mostly Python-level contract work.

## Verification

Implementation should run the smallest reliable targeted tests first, then broaden before
commit.

Expected verification after implementation:

```powershell
python -m pytest tests/unit/test_vnext_evaluation.py tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_cmaes_ask_tell_vnext.py tests/unit/test_vnext_policy_scheduler.py -v
python -m ruff format --check
python -m ruff check
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
python -m maturin develop --release
python -m pytest tests/unit/ tests/integration/ -v
python -m pytest tests/property/ -v
```

Docs-only spec work should at minimum pass a whitespace and placeholder review before
commit.

## Rollout

This design should become one implementation plan. It should not absorb result/history
export, problem/gene declaration redesign, constraints, multi-objective support, or the
`optimize(...)` wrapper.

Recommended rollout slices:

1. Add protocols and lifecycle record refinements.
2. Make `GAEngine` conform and update tests.
3. Make `CMAESEngine` conform and update tests.
4. Refresh docs, examples, changelog, and API docs.
5. Run full verification and prepare a breaking-change PR.

## Acceptance Criteria

- EvoCore exposes stable `Optimizer` and `Evaluator` structural protocols.
- Public lifecycle records cover candidate identity, batch identity, evaluation records,
  context, tell results, and state summaries.
- `GAEngine` and `CMAESEngine` implement the protocol contract.
- Manual ask/tell accepts asynchronous partial records safely.
- Invalid tell data is rejected with clear errors.
- Trusted optimizer state updates only from explicitly trusted records by default.
- Minimize and maximize directions behave correctly without rewriting raw user scores.
- Documentation describes the protocol lifecycle as the primary public API.
- No trading-specific evaluator logic or metrics are added.

## Deferred Follow-Ups

- Result, history, telemetry, JSON, and pandas export stabilization.
- Problem and gene specification stabilization, including categorical genes,
  permutations, constraints, and noisy objectives.
- Multi-objective optimization and Pareto result objects.
- Convenience `optimize(...)` orchestration.
- Advisor protocol stabilization and richer type-aware encoders.
- Island models, operator libraries, repair hooks, and diversity-preserving selection.
