# Ask/Tell Engines

EvoCore optimizers expose a structural ask/tell lifecycle. An optimizer does not need
to inherit from a base class; it satisfies `Optimizer` when it exposes `direction`,
`ask(...)`, `tell(...)`, and `state_summary()`.

`ask()` returns `Candidate` objects with stable `candidate_id` values, decoded genes,
optional params, and one shared `batch_id` for the ask event. `tell()` accepts only
`EvaluationRecord` values and returns `UpdateResult`.

## Candidate And Solution Boundary

`Candidate` is the lifecycle-facing record returned by `ask()`. It carries lifecycle
identity and scheduling state such as `candidate_id`, `batch_id`, `origin`, `parents`,
`stage`, `status`, decoded `genes`, optional `params`, and evaluation observations.
Evaluators receive candidates because they need proposal identity as well as decoded
values.

`Solution` is the population/result-facing record exposed by completed optimizer runs. It
stores decoded `values`, `score`, `score_valid`, and result metadata. Result metadata may
include provenance such as `candidate_id`, `candidate_hash`, `batch_id`, `origin`, and
`generation`, but scheduler state and observation history stay on lifecycle records.

Use `candidate_id` to refer to one lifecycle proposal. Use
`candidate.candidate_hash(gene_space)` or `gene_space.value_hash(candidate.genes)` to
compare search-space values. The hash is schema-aware and includes the `GeneSpace` hash,
so identical raw values in incompatible spaces do not collapse into the same search
point.

Evaluators satisfy the structural `Evaluator` protocol by implementing:

```python
def evaluate(candidates, context):
    return []
```

`EvaluationContext` carries the evaluation stage, batch ID, event index, direction, budget, and
metadata for the evaluator call.

`tell()` is asynchronous-friendly: callers may report any subset of a batch, in any
order, as long as each candidate/stage pair is reported at most once. `tell([])` is a
valid no-op for queue polling integrations.

`UpdateResult.accepted_count` counts records accepted by the ask/tell ledger.
Optimizers that make a separate state decision also return
`acceptance_decisions`. `AcceptanceDecision.accepted_for_state` is the per-record
boolean for whether optimizer state changed. For Differential Evolution, this
means a trial replaced its target slot.

Confidence values are explicit:

- `trusted_full` records carry finite raw scores from fresh full objective work. They
  update optimizer state and consume full-evaluation budget.
- `cached` records carry finite raw scores from trusted previous full evaluations. They
  update optimizer state but do not consume fresh full-evaluation budget.
- `partial` and `surrogate` records carry finite scores for scheduling, telemetry, and
  history, but they cannot become optimizer best state.
- `rejected` records represent recoverable candidate-level failures. They must use
  `score=None` and carry diagnostics in `metrics` or `metadata`.

Raw user scores are preserved. Optimizers use `direction="maximize"` or
`direction="minimize"` to compare candidates without rewriting the score stored in
`EvaluationRecord`. State best-candidate tracking compares only state-eligible
`trusted_full` and `cached` scores, so partial and surrogate scores cannot become the
reported optimizer best. `OptimizerStateSummary.trusted_count` counts candidates with
state-eligible records.

Invalid records raise `FitnessError`: unknown candidates, unknown explicit batch IDs,
batch mismatches, duplicate candidate/stage records, non-finite non-rejected scores, and
scored `rejected` records are rejected.

## External State

GA, DE, and CMA-ES expose shared external-state methods for expensive systems
that reuse historical optimization knowledge. Use `warm_start(...)` to import
trusted archives or search-memory records, `candidate_snapshot(...)` and
`top_candidates(...)` for survivor selection and reporting, and
`inject_candidates(...)` when the optimizer supports externally proposed
candidates during an ask/tell run.

```python
from evocore import GeneSpace, GeneticAlgorithmOptimizer, WarmStartRecord


optimizer = GeneticAlgorithmOptimizer(GeneSpace.uniform(-5.0, 5.0, 3), seed=42)
optimizer.warm_start(
    [
        WarmStartRecord(
            values=(0.1, -0.2, 0.3),
            score=0.88,
            metadata={"source": "search_memory"},
        )
    ]
)

survivors = optimizer.top_candidates(4)
```

Use `candidate_snapshot(scope="trusted")` for state-eligible candidates,
`scope="scored"` for any scored candidate, `scope="pending"` for in-flight
ask/tell work, and `scope="known"` for everything the optimizer currently knows.
See [Expensive External Evaluations](expensive-external-evaluations.md) for
warm starts, cached records, injected candidates, and hybrid GA/CMA-ES recipes.

## Event History

Ask/tell engines record append-only lifecycle events. Every proposed candidate receives
an `ask` event with its batch ID, candidate ID, schema-aware candidate hash, origin,
genes, params, and metadata. Every accepted evaluation record receives a `tell` event with
the raw score, direction-aware comparison score, confidence, stage, cost, resulting
status, metrics, and record metadata.

```python
candidates = engine.ask(4)
records = [
    EvaluationRecord(
        candidate_id=candidate.candidate_id,
        batch_id=candidate.batch_id,
        score=score_candidate(candidate),
        confidence="trusted_full",
        stage="full",
        cost=1.0,
    )
    for candidate in candidates
]
engine.tell(records)

rows = engine.events.to_rows()
```

Raw user scores are stored under `raw_score`. EvoCore stores the value used for
direction-aware comparisons separately under `comparison_score`.
