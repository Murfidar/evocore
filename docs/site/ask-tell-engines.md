# Ask/Tell Engines

`GAEngine` and `CMAESEngine` expose ask/tell state.

`ask()` returns candidates with stable IDs and decoded params. `tell()` accepts
`EvaluationRecord` values and updates engine state only according to each record's
confidence level.

Each `ask()` call also assigns one shared `batch_id` to the returned candidates. That
batch token is public and deterministic for a given engine seed and ask event, so remote
or asynchronous evaluators can safely regroup partial results before calling `tell()`.

`tell()` is first-class asynchronous: you may report any subset of records from an ask
batch, in any order, as long as each candidate/rung pair is reported at most once.
Duplicate trusted results for the same candidate are rejected.

Trusted full records update optimizer state by default. Surrogate and partial records are
used for scheduling, screening, and telemetry unless a policy explicitly allows
aggressive state updates.

For `GAEngine.run(...)`, EvoCore stays stricter than the manual ask/tell API. A
synchronous evaluator must return exactly one record for each assigned candidate at each
rung. Missing, duplicate, or batch-mismatched records raise `FitnessError` instead of
silently stalling the policy loop.

For `CMAESEngine`, trusted full records are accumulated per `batch_id` and only advance
the covariance state once a full trusted batch is complete. After a batch has been
consumed, later trusted records for that batch are rejected.
