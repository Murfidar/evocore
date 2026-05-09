# Ask/Tell Engines

`GAEngine` and `CMAESEngine` expose ask/tell state.

`ask()` returns candidates with stable IDs and decoded params. `tell()` accepts
`EvaluationRecord` values and updates engine state only according to each record's
confidence level.

Trusted full records update optimizer state by default. Surrogate and partial records are
used for scheduling, screening, and telemetry unless a policy explicitly allows aggressive
state updates.
