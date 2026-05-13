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

- `trusted_full` updates optimizer state by default.
- `partial` and `surrogate` can inform scheduling and telemetry.
- `cached` records are tracked separately so policies can decide whether to trust them.
- `rejected` records may omit score.

Raw user scores are preserved. Optimizers use `direction="maximize"` or
`direction="minimize"` to compare candidates without rewriting the score stored in
`EvaluationRecord`.

Invalid records raise `FitnessError`: unknown candidates, unknown explicit batch IDs,
batch mismatches, duplicate candidate/rung records, and non-finite non-rejected scores
are rejected.
