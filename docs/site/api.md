# API Reference

::: evocore.gene_space

::: evocore.individual

::: evocore.operators

::: evocore.stats

::: evocore.parallel

::: evocore.exceptions

## vNext Expensive Optimization

`Candidate.batch_id` groups candidates that came from the same `ask()` event.
`EvaluationRecord.batch_id` may be supplied to make asynchronous evaluators explicit;
when present, it must match the candidate batch. Batch IDs are especially useful when
trusted records arrive in multiple `tell()` calls.

::: evocore.evaluation.Candidate

::: evocore.evaluation.EvaluationRecord

::: evocore.evaluation.Rung

::: evocore.evaluation.OptimizationTelemetry

::: evocore.policies.MultiFidelityPolicy

::: evocore.scheduler.EvaluationScheduler

::: evocore.advisors.InverseDistanceSurrogateAdvisor

::: evocore.mixed_cma.IntegerMargin

::: evocore.mixed_cma.CategoricalState

