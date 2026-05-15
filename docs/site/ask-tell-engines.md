# Ask/Tell Engines

EvoCore optimizers expose a structural ask/tell lifecycle. An optimizer does not need
to inherit from a base class; it satisfies `Optimizer` when it exposes `direction`,
`ask(...)`, `tell(...)`, and `state_summary()`.

`ask()` returns `Candidate` objects with stable `candidate_id` values, decoded genes,
optional params, and one shared `batch_id` for the ask event. `tell()` accepts only
`EvaluationRecord` values and returns `TellResult`.

Evaluators satisfy the structural `Evaluator` protocol by implementing:

```python
def evaluate(candidates, context):
    return []
```

`EvaluationContext` carries the rung, batch ID, event index, direction, budget, and
metadata for the evaluator call.

`tell()` is asynchronous-friendly: callers may report any subset of a batch, in any
order, as long as each candidate/rung pair is reported at most once. `tell([])` is a
valid no-op for queue polling integrations.

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
reported optimizer best. `EngineStateSummary.trusted_count` counts candidates with
state-eligible records.

Invalid records raise `FitnessError`: unknown candidates, unknown explicit batch IDs,
batch mismatches, duplicate candidate/rung records, non-finite non-rejected scores, and
scored `rejected` records are rejected.

## Event History

Ask/tell engines record append-only lifecycle events. Every proposed candidate receives
an `ask` event with its batch ID, candidate ID, genome hash, origin, genes, params, and
metadata. Every accepted evaluation record receives a `tell` event with the raw score,
direction-aware comparison score, confidence, rung, cost, resulting status, metrics, and
record metadata.

```python
candidates = engine.ask(4)
records = [
    EvaluationRecord(
        candidate_id=candidate.candidate_id,
        batch_id=candidate.batch_id,
        score=score_candidate(candidate),
        confidence="trusted_full",
        rung="full",
        cost=1.0,
    )
    for candidate in candidates
]
engine.tell(records)

rows = engine.history.to_rows()
```

Raw user scores are stored under `raw_score`. EvoCore stores the value used for
direction-aware comparisons separately under `comparison_score`.
