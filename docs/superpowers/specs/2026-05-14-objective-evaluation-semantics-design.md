# Objective And Evaluation Semantics Design

**Date:** 2026-05-14
**Status:** Draft approved for specification
**Scope:** Stable single-objective scoring, evaluation record, failure, repeated observation, cache, state, and budget semantics for EvoCore optimizers

## Summary

EvoCore should use one strict objective-evaluation contract across manual `ask/tell`
flows and evaluator-driven runs. Evaluators return raw objective scores in the domain's
natural direction. EvoCore preserves those raw scores in public result and history
surfaces, and derives explicit direction-aware comparison scores internally and in
dedicated export fields.

The stable core remains single-objective. Every non-rejected objective observation must
have a finite raw score. Recoverable per-candidate failures are represented by
`EvaluationRecord(confidence="rejected", score=None, ...)`; uncaught evaluator exceptions
abort the evaluation or run as `FitnessError`. Legacy behavior that sanitizes non-finite
fitness values into `-inf` is not part of the target contract because the legacy
`fitness_fn` entrypoint will be removed.

This slice intentionally keeps constraints metadata-only and noisy-objective aggregation
outside the optimizer core. It defines the semantics that future budget, termination,
constraint, and noisy-objective features can build on without changing what a score means.

## Product Direction

The public user model is explicit evaluation:

```python
candidates = optimizer.ask(16)
records = evaluator.evaluate(candidates, context)
result = optimizer.tell(records)
```

An `EvaluationRecord` is the only public observation accepted by `Optimizer.tell(...)`.
Its `score` is always the raw user score when a valid observation exists. The optimizer's
`direction` determines comparison only:

```text
comparison_score = raw_score   # direction="maximize"
comparison_score = -raw_score  # direction="minimize"
```

This keeps result exports readable for both maximization and minimization. Users who
minimize a loss return the loss itself, not an inverted value.

## Goals

- Define one strict scoring contract for evaluator-driven runs and manual `ask/tell`.
- Preserve raw objective scores on public result, state, candidate, and history surfaces.
- Expose direction-aware values only through explicit `comparison_score` fields or helper
  methods.
- Reject non-finite non-rejected scores instead of sanitizing them.
- Represent recoverable candidate-level failures as rejected evaluation records.
- Keep uncaught evaluator exceptions as run-aborting `FitnessError` failures.
- Define which confidence levels are state-eligible, budget-consuming, and best-eligible.
- Define repeated-observation and duplicate-genome semantics without adding aggregation.
- Keep constraints metadata-only in this slice.
- Preserve deterministic history and telemetry export behavior.

## Non-Goals

- Do not add multi-objective, Pareto, or vector-valued objective semantics.
- Do not add first-class constraint objects, feasibility ranking, or penalty helpers.
- Do not add core noisy-objective aggregation, uncertainty estimates, or confidence
  interval semantics.
- Do not add stale-cache invalidation, cache storage, or cache key APIs.
- Do not design global budget and termination controls beyond the evaluation-counting
  rules required by this objective contract.
- Do not preserve legacy lambda-first `run(fitness_fn)` sanitization behavior.
- Do not add domain-specific objective, metric, or evaluator logic.

## Public Scoring Model

Evaluators return raw objective values. `EvaluationRecord.score`, `EventRecord.raw_score`,
`RunResult.best_score`, compatibility `RunResult.best_fitness` while it exists,
`EngineStateSummary.best_score`, and candidate best-score helpers all use raw values.

Direction-aware comparison is separate. EvoCore may store or export `comparison_score`
where larger is always better, but comparison scores do not replace raw score fields.

Rules:

- `direction="maximize"` compares larger raw scores as better.
- `direction="minimize"` compares smaller raw scores as better.
- `comparison_score` is `raw_score` for maximize and `-raw_score` for minimize.
- Evaluators must not invert minimization objectives themselves.
- Public best-score fields report the winning raw score, not the comparison score.

## EvaluationRecord Semantics

`EvaluationRecord` confidence determines how an observation may be used:

| Confidence | Score | State-eligible | Budget-consuming | Best-eligible | Meaning |
| --- | --- | --- | --- | --- | --- |
| `trusted_full` | finite raw score | yes | yes | yes | Fresh full objective observation. |
| `cached` | finite raw score | yes | no | yes | Trusted previous full observation reused for this candidate. |
| `partial` | finite raw score | no | no | no | Lower-fidelity or incomplete objective observation. |
| `surrogate` | finite raw score | no | no | no | Model, heuristic, or screening score. |
| `rejected` | `None` | no | no | no | No valid objective observation. |

All accepted records remain history-visible. `TellResult`, `OptimizationTelemetry`, and
`EventHistory` should preserve enough detail for users to audit partial, surrogate,
cached, trusted, and rejected records separately.

`rejected` records must use `score=None`. A finite score that should not update optimizer
state should use `partial` or `surrogate`, not `rejected`.

## Validation And Errors

The validation boundary should be strict:

- Non-rejected records require a finite numeric `score`.
- `rejected` records require `score=None`.
- Record `cost` must remain finite and non-negative.
- Record confidence values must be known.
- Candidate IDs must be known to the optimizer receiving `tell(...)`.
- Explicit batch IDs must be known and match the candidate's batch.
- Each candidate/rung pair accepts at most one record.
- Each candidate accepts at most one state-eligible record in a batch.

Recoverable candidate-level failures are records, not exceptions:

```python
EvaluationRecord(
    candidate_id=candidate.candidate_id,
    score=None,
    confidence="rejected",
    rung="full",
    metadata={
        "reason": "evaluation_timeout",
        "exception_type": "TimeoutError",
    },
)
```

Uncaught evaluator exceptions abort the evaluator call or run and are surfaced as
`FitnessError`. EvoCore should not silently convert evaluator exceptions into scores or
records.

## State, Best, And Budget Semantics

State update and budget accounting are related but distinct:

```text
state-eligible = trusted_full or cached
best-eligible = trusted_full or cached
full-budget-consuming = trusted_full only
history-visible = every accepted record
```

Only fresh `trusted_full` records consume `max_evaluations` or policy-level full
evaluation budget. `cached` records are trusted evidence and may update optimizer state,
but they do not represent new objective work. `partial`, `surrogate`, and `rejected`
records do not consume full-evaluation budget.

Cheap observations may influence scheduling, promotion, or telemetry, but they cannot
become `RunResult.best_score`, `EngineStateSummary.best_score`, or the optimizer's
state best unless a later `trusted_full` or `cached` record exists for that candidate.

## Cached Evaluation Semantics

`cached` means the evaluator is reporting a trusted score from previous full objective
work. EvoCore does not own cache lookup, cache storage, invalidation, or cache freshness
in this slice.

Rules:

- Cached records require finite raw scores.
- Cached records are state-eligible and best-eligible.
- Cached records do not consume fresh full-evaluation budget.
- Cached records should be counted separately in `TellResult.cached_count`, telemetry, and
  event history.
- Cache provenance, key, age, or source may be carried in `metadata`.

## Repeated Observations And Noisy Objectives

The core does not aggregate repeated observations. A candidate/rung pair accepts one
record. This keeps optimizer state deterministic and avoids choosing a premature
aggregation policy.

For noisy objectives:

- Evaluators may run repeated samples internally and return one aggregated
  `EvaluationRecord`.
- Aggregation diagnostics such as `n_repeats`, `mean`, `std`, raw replicate summaries, or
  confidence information belong in `metrics` or `metadata`.
- If EvoCore later proposes the same genes again, the new proposal has a distinct
  `candidate_id`.
- Duplicate genomes are observed through shared `candidate_hash` telemetry, not candidate
  identity merging.
- Different fidelity or protocol levels should use distinct rungs.

Future noisy-objective support can add explicit aggregation semantics on top of this
contract without changing the one-record-per-candidate/rung rule for the base optimizer
core.

## Constraint Semantics

Constraints remain metadata-only in this slice.

Hard constraints that prevent valid evaluation should produce rejected records with
diagnostics:

```python
EvaluationRecord(
    candidate_id=candidate.candidate_id,
    score=None,
    confidence="rejected",
    rung="full",
    metadata={
        "reason": "constraint_violation",
        "constraint": "max_drawdown",
    },
)
```

Soft constraints are evaluator-owned. If a user wants a penalty, the evaluator applies it
and returns a finite raw score. EvoCore preserves constraint diagnostics in history and
telemetry metadata but does not interpret them, rank by feasibility, or combine penalties.

## Result And History Export

Result and history exports should continue to make raw and comparison values explicit:

- `RunResult.best_score` is the raw winning score.
- `RunResult.best_fitness`, while retained for compatibility, is the same raw score.
- `EngineStateSummary.best_score` is raw.
- `EventRecord.raw_score` is raw.
- `EventRecord.comparison_score` is derived from direction when a finite score exists.
- Rejected records export `raw_score=None` and `comparison_score=None`.
- `ReproducibilityMetadata.direction` records the comparison direction used for the run.

Deterministic JSON export should keep rejecting non-finite score payloads. Runtime fields
remain opt-in according to the result/history telemetry contract.

## Compatibility And Migration

The strict evaluator contract is the target behavior. Legacy `fitness_fn` paths that
sanitize `NaN` or `Inf` into `-inf` should not be carried forward as public semantics.
During cleanup:

- Evaluator-driven `run(...)` and manual `ask/tell` should share the same
  `EvaluationRecord` validation rules.
- Legacy non-finite warning/sanitization behavior should be removed with the legacy
  `fitness_fn` entrypoint.
- Result compatibility names may remain temporarily, but they should preserve raw scores.
- Documentation should direct users to rejected records for recoverable failures and
  finite raw scores for valid observations.

## Testing Guidance

Implementation should cover:

- `EvaluationRecord` rejects non-finite non-rejected scores.
- `EvaluationRecord` rejects scored `rejected` records.
- `score_for_direction(...)` preserves maximize and minimize comparison semantics.
- `tell(...)` rejects unknown candidates, mismatched batches, duplicate candidate/rung
  records, and duplicate state-eligible records.
- `trusted_full` updates state, best, telemetry, and full-evaluation budget.
- `cached` updates state and best but not full-evaluation budget.
- `partial` and `surrogate` are recorded but cannot become state best.
- `rejected` records are accepted as failures, recorded in history, and excluded from
  state and budget.
- Recoverable failure diagnostics survive in event history export.
- Uncaught evaluator exceptions surface as `FitnessError`.
- Minimize result fields remain raw while exported comparison scores are negated.
