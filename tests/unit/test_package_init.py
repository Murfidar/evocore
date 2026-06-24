def test_evocore_imports_without_error():
    import evocore  # noqa: F401


def test_core_extension_accessible():
    from evocore import _core

    assert hasattr(_core, "FloatIndividual")
    assert hasattr(_core, "IntegerIndividual")
    assert hasattr(_core, "BinaryIndividual")
    assert hasattr(_core, "py_derive_seed")


def test_errors_accessible_from_top_level():
    from evocore import (
        CheckpointError,
        ConfigurationError,
        ConfigurationWarning,
        ConvergenceError,
        EvocoreError,
        FitnessError,
        FitnessWarning,
        ParallelError,
    )

    assert issubclass(ConfigurationError, EvocoreError)
    assert issubclass(FitnessWarning, Warning)
    assert issubclass(ConfigurationWarning, Warning)
    assert issubclass(CheckpointError, EvocoreError)
    assert issubclass(ConvergenceError, EvocoreError)
    assert issubclass(FitnessError, EvocoreError)
    assert issubclass(ParallelError, EvocoreError)


def test_search_space_exports_accessible_from_top_level():
    from evocore import Gene, GeneSpace, Solution, SolutionSet

    space = GeneSpace.uniform(-1.0, 1.0, 2)
    solution = Solution([1.0, 0.0])
    solutions = SolutionSet([solution])

    assert Gene("x", "float", -1.0, 1.0).name == "x"
    assert space.length == 2
    assert solution.values == [1.0, 0.0]
    assert len(solutions) == 1


def test_optimizer_exports_accessible_from_top_level():
    from evocore import CMAESOptimizer, DifferentialEvolutionOptimizer, GeneticAlgorithmOptimizer

    assert CMAESOptimizer is not None
    assert DifferentialEvolutionOptimizer is not None
    assert GeneticAlgorithmOptimizer is not None


def test_result_exports_accessible_from_top_level():
    from evocore import (
        CheckpointSnapshot,
        GenerationHistory,
        GenerationRecord,
        OptimizationBatchResult,
        OptimizationResult,
    )

    assert CheckpointSnapshot is not None
    assert GenerationHistory is not None
    assert GenerationRecord is not None
    assert OptimizationBatchResult is not None
    assert OptimizationResult is not None


def test_lifecycle_exports_accessible_from_top_level():
    import evocore

    assert evocore.Candidate.__name__ == "Candidate"
    assert evocore.EvaluationRecord.__name__ == "EvaluationRecord"
    assert evocore.EvaluationStage.__name__ == "EvaluationStage"
    assert evocore.BudgetPolicy.__name__ == "BudgetPolicy"
    assert evocore.BudgetScheduler.__name__ == "BudgetScheduler"
    assert evocore.OptimizationTelemetry.__name__ == "OptimizationTelemetry"
    assert evocore.AcceptanceDecision.__name__ == "AcceptanceDecision"
    assert evocore.UpdateResult.__name__ == "UpdateResult"
    assert evocore.OptimizerStateSummary.__name__ == "OptimizerStateSummary"
    assert evocore.EventRecord.__name__ == "EventRecord"
    assert evocore.EventHistory.__name__ == "EventHistory"
    assert evocore.ReproducibilityMetadata.__name__ == "ReproducibilityMetadata"


def test_external_state_public_exports():
    from evocore import (
        CandidateSnapshot,
        ExternalStateCapabilities,
        ExternalStateOptimizer,
        InjectionResult,
        PopulationSnapshot,
        WarmStartRecord,
        cached_records,
    )

    assert WarmStartRecord is not None
    assert CandidateSnapshot is not None
    assert PopulationSnapshot is not None
    assert ExternalStateCapabilities is not None
    assert ExternalStateOptimizer is not None
    assert InjectionResult is not None
    assert cached_records is not None


def test_phase2a_public_exports():
    from evocore import (
        CandidateArchive,
        FamilyQuota,
        SelectionResult,
        SpecialistCap,
        select_candidates,
    )

    assert CandidateArchive is not None
    assert FamilyQuota is not None
    assert SelectionResult is not None
    assert SpecialistCap is not None
    assert select_candidates is not None


def test_phase2b_stopping_public_exports():
    from evocore import (
        CompositeStopPolicy,
        ConvergencePolicy,
        EvaluationLimitPolicy,
        NoImprovementPolicy,
        StopDecision,
        StopPolicy,
    )

    assert CompositeStopPolicy is not None
    assert ConvergencePolicy is not None
    assert EvaluationLimitPolicy is not None
    assert NoImprovementPolicy is not None
    assert StopDecision is not None
    assert StopPolicy is not None


def test_phase2c_composition_public_exports():
    from evocore import derive_child_seed, inner_result_record, lineage_metadata

    assert derive_child_seed is not None
    assert inner_result_record is not None
    assert lineage_metadata is not None


def test_phase3a_projection_public_exports():
    from evocore import ActiveGeneProjection, BinaryThresholdTransform, ConstraintViolation

    assert ActiveGeneProjection is not None
    assert BinaryThresholdTransform is not None
    assert ConstraintViolation is not None
