# API Reference

::: evocore.search_space.GeneSpace

::: evocore.Solution

::: evocore.search_space.OperatorCodec

::: evocore.search_space.repair_gene_value

::: evocore.search_space.repair_gene_values

::: evocore.search_space.encode_gene_values

::: evocore.search_space.decode_gene_values

::: evocore.search_space.ActiveGeneProjection

::: evocore.search_space.ParameterProjection

::: evocore.search_space.ProjectionResult

::: evocore.search_space.ProjectionSnapshot

::: evocore.search_space.ParameterTransform

::: evocore.search_space.IdentityTransform

::: evocore.search_space.BinaryThresholdTransform

::: evocore.search_space.ExponentialIntegerTransform

::: evocore.search_space.OutputNameTransform

::: evocore.search_space.ConstraintViolation

::: evocore.search_space.RepairRecord

::: evocore.optimizers.operators.BoundsPolicy

::: evocore.optimizers.operators.CrossoverOperator

::: evocore.optimizers.operators.MutationOperator

::: evocore.optimizers.operators.SelectionOperator

::: evocore.optimizers.de.DifferentialEvolutionOptimizer

::: evocore.results.GenerationHistory

::: evocore.core.parallel

::: evocore.core.errors

## Optimizer Lifecycle

`Optimizer` and `Evaluator` are structural protocols. Engines and evaluators conform by
shape, without subclassing.

::: evocore.lifecycle.Optimizer

::: evocore.lifecycle.Evaluator

::: evocore.lifecycle.Candidate

::: evocore.lifecycle.EvaluationRecord

::: evocore.lifecycle.EvaluationContext

::: evocore.lifecycle.UpdateResult

::: evocore.lifecycle.AcceptanceDecision

::: evocore.lifecycle.OptimizerStateSummary

::: evocore.lifecycle.WarmStartRecord

::: evocore.lifecycle.CandidateSnapshot

::: evocore.lifecycle.PopulationSnapshot

::: evocore.lifecycle.ExternalStateCapabilities

::: evocore.lifecycle.InjectionResult

::: evocore.lifecycle.cached_records

::: evocore.lifecycle.derive_child_seed

::: evocore.lifecycle.lineage_metadata

::: evocore.lifecycle.inner_result_record

::: evocore.lifecycle.CandidateArchive

::: evocore.lifecycle.ArchiveEntry

::: evocore.lifecycle.ArchiveExport

::: evocore.lifecycle.select_candidates

::: evocore.lifecycle.SelectionResult

::: evocore.lifecycle.SelectionDecision

::: evocore.lifecycle.FamilyQuota

::: evocore.lifecycle.SpecialistCap

::: evocore.lifecycle.StopDecision

::: evocore.lifecycle.StopPolicy

::: evocore.lifecycle.EvaluationLimitPolicy

::: evocore.lifecycle.NoImprovementPolicy

::: evocore.lifecycle.ConvergencePolicy

::: evocore.lifecycle.CompositeStopPolicy

::: evocore.lifecycle.EvaluationStage

::: evocore.lifecycle.OptimizationTelemetry

::: evocore.results.EventRecord

::: evocore.results.EventHistory

::: evocore.results.ReproducibilityMetadata

::: evocore.lifecycle.BudgetPolicy

::: evocore.lifecycle.BudgetScheduler

::: evocore.surrogates.InverseDistanceAdvisor

::: evocore.optimizers.cmaes.IntegerMarginDistribution

::: evocore.optimizers.cmaes.CategoricalDistributionState

::: evocore.optimizers.OptimizerConfig

::: evocore.optimizers.RuntimeHookSignature

::: evocore.optimizers.ConfigurableComponent

::: evocore.optimizers.config_hash
