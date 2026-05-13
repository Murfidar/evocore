# Optimizer Telemetry

`OptimizationTelemetry` tracks the true breadth and cost of optimizer search.

Telemetry includes proposed candidates, unique candidate genome hashes, screened candidates,
partial evaluations, full evaluations, promoted and eliminated counts by rung, and cost by
rung. External evaluators can use this evidence to audit search breadth, budget use, and
selection pressure without embedding domain-specific metrics in EvoCore.
