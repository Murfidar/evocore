import importlib

import pytest


def test_new_domain_imports_are_available():
    modules = [
        "evocore.core.errors",
        "evocore.core.serialization",
        "evocore.core.parallel",
        "evocore.search_space",
        "evocore.search_space.genes",
        "evocore.search_space.solutions",
        "evocore.search_space.codec",
        "evocore.lifecycle",
        "evocore.lifecycle.records",
        "evocore.lifecycle.batches",
        "evocore.lifecycle.policies",
        "evocore.lifecycle.scheduler",
        "evocore.lifecycle.protocols",
        "evocore.lifecycle.telemetry",
        "evocore.lifecycle.events",
        "evocore.results",
        "evocore.results.checkpointing",
        "evocore.results.generation",
        "evocore.results.reproducibility",
        "evocore.results.run",
        "evocore.optimizers",
        "evocore.optimizers.ga",
        "evocore.optimizers.cmaes",
        "evocore.optimizers.config",
        "evocore.optimizers.ga.config",
        "evocore.optimizers.cmaes.config",
        "evocore.callbacks",
        "evocore.surrogates",
    ]

    for module_name in modules:
        assert importlib.import_module(module_name).__name__ == module_name


def test_new_domain_symbols_are_importable():
    from evocore import CheckpointSnapshot as TopLevelCheckpointSnapshot
    from evocore import OptimizerConfig as TopLevelOptimizerConfig
    from evocore import RuntimeHookSignature as TopLevelRuntimeHookSignature
    from evocore.lifecycle import BudgetPolicy, BudgetScheduler, EvaluationStage
    from evocore.optimizers import ConfigurableComponent, OptimizerConfig, RuntimeHookSignature
    from evocore.optimizers.cmaes import CMAESOptimizer
    from evocore.optimizers.ga import GeneticAlgorithmOptimizer
    from evocore.results import CheckpointSnapshot, OptimizationBatchResult, OptimizationResult
    from evocore.search_space import Gene, GeneSpace, Solution, SolutionSet
    from evocore.surrogates import InverseDistanceAdvisor, SurrogateScore

    assert TopLevelCheckpointSnapshot is CheckpointSnapshot
    assert TopLevelOptimizerConfig is OptimizerConfig
    assert TopLevelRuntimeHookSignature is RuntimeHookSignature
    assert CheckpointSnapshot is not None
    assert ConfigurableComponent is not None
    assert OptimizerConfig is not None
    assert RuntimeHookSignature is not None
    assert BudgetPolicy is not None
    assert BudgetScheduler is not None
    assert EvaluationStage is not None
    assert CMAESOptimizer is not None
    assert GeneticAlgorithmOptimizer is not None
    assert OptimizationBatchResult is not None
    assert OptimizationResult is not None
    assert Gene is not None
    assert GeneSpace is not None
    assert Solution is not None
    assert SolutionSet is not None
    assert InverseDistanceAdvisor is not None
    assert SurrogateScore is not None


def test_domain_packages_export_symbols_owned_by_focused_modules():
    from evocore.callbacks import (
        Callback,
        CheckpointCallback,
        EarlyStopping,
        GenerationInfo,
        MetricsLogger,
        ProgressBar,
    )
    from evocore.lifecycle import (
        OptimizationTelemetry,
        OptimizerStateSummary,
        UpdateResult,
        candidate_to_solution,
        solution_to_candidate,
    )
    from evocore.lifecycle.events import EventHistory, EventRecord, append_run_stop_event
    from evocore.optimizers import OptimizerConfig, RuntimeHookSignature, config_hash
    from evocore.optimizers.cmaes import CMAESAskTellMixin, CMAESOptimizer
    from evocore.optimizers.config import ConfigurableComponent
    from evocore.optimizers.ga import (
        GeneticAlgorithmAskTellMixin,
        GeneticAlgorithmCheckpointingMixin,
        GeneticAlgorithmGenerationLoopMixin,
        GeneticAlgorithmMultiRunMixin,
        GeneticAlgorithmOptimizer,
    )
    from evocore.results.checkpointing import (
        CheckpointSnapshot,
        load_checkpoint,
        save_checkpoint,
        validate_checkpoint_identity,
    )
    from evocore.results.reproducibility import (
        ReproducibilityMetadata,
        gene_space_hash,
        gene_space_signature,
    )
    from evocore.surrogates import InverseDistanceAdvisor, SurrogateScore

    assert GenerationInfo.__module__ == "evocore.callbacks.base"
    assert Callback.__module__ == "evocore.callbacks.base"
    assert EarlyStopping.__module__ == "evocore.callbacks.stopping"
    assert ProgressBar.__module__ == "evocore.callbacks.progress"
    assert CheckpointCallback.__module__ == "evocore.callbacks.checkpointing"
    assert MetricsLogger.__module__ == "evocore.callbacks.metrics"

    assert SurrogateScore.__module__ == "evocore.surrogates.scoring"
    assert InverseDistanceAdvisor.__module__ == "evocore.surrogates.inverse_distance"

    assert EventRecord.__module__ == "evocore.lifecycle.events"
    assert EventHistory.__module__ == "evocore.lifecycle.events"
    assert append_run_stop_event.__module__ == "evocore.lifecycle.events"
    assert candidate_to_solution.__module__ == "evocore.lifecycle.conversion"
    assert solution_to_candidate.__module__ == "evocore.lifecycle.conversion"
    assert OptimizationTelemetry.__module__ == "evocore.lifecycle.telemetry"
    assert UpdateResult.__module__ == "evocore.lifecycle.telemetry"
    assert OptimizerStateSummary.__module__ == "evocore.lifecycle.telemetry"
    assert CheckpointSnapshot.__module__ == "evocore.results.checkpointing"
    assert load_checkpoint.__module__ == "evocore.results.checkpointing"
    assert save_checkpoint.__module__ == "evocore.results.checkpointing"
    assert validate_checkpoint_identity.__module__ == "evocore.results.checkpointing"
    assert ReproducibilityMetadata.__module__ == "evocore.results.reproducibility"
    assert gene_space_signature.__module__ == "evocore.results.reproducibility"
    assert gene_space_hash.__module__ == "evocore.results.reproducibility"

    assert OptimizerConfig.__module__ == "evocore.optimizers.config"
    assert RuntimeHookSignature.__module__ == "evocore.optimizers.config"
    assert ConfigurableComponent.__module__ == "evocore.optimizers.config"
    assert config_hash.__module__ == "evocore.optimizers.config"

    assert GeneticAlgorithmAskTellMixin in GeneticAlgorithmOptimizer.__mro__
    assert GeneticAlgorithmGenerationLoopMixin in GeneticAlgorithmOptimizer.__mro__
    assert GeneticAlgorithmCheckpointingMixin in GeneticAlgorithmOptimizer.__mro__
    assert GeneticAlgorithmMultiRunMixin in GeneticAlgorithmOptimizer.__mro__
    assert CMAESAskTellMixin in CMAESOptimizer.__mro__


def test_solution_and_callback_compatibility_aliases_are_removed():
    from evocore.callbacks import GenerationInfo
    from evocore.search_space import Solution, SolutionSet

    with pytest.raises(TypeError):
        Solution([1.0], fitness=1.0)
    with pytest.raises(TypeError):
        Solution([1.0], fitness_valid=True)

    solution = Solution([1.0], score=2.0, score_valid=True)
    assert not hasattr(solution, "genes")
    assert not hasattr(solution, "fitness")
    assert not hasattr(solution, "fitness_valid")

    solutions = SolutionSet([solution])
    assert not hasattr(solutions, "mean_fitness")
    assert not hasattr(solutions, "std_fitness")

    info = GenerationInfo(generation=0, nan_score_count=1, cached_count=2)
    assert not hasattr(info, "nan_fitness_count")
