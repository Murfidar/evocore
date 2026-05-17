import importlib


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
        "evocore.results.generation",
        "evocore.results.reproducibility",
        "evocore.results.run",
        "evocore.optimizers",
        "evocore.optimizers.ga",
        "evocore.optimizers.cmaes",
        "evocore.callbacks",
        "evocore.surrogates",
    ]

    for module_name in modules:
        assert importlib.import_module(module_name).__name__ == module_name


def test_new_domain_symbols_are_importable():
    from evocore.lifecycle import BudgetPolicy, BudgetScheduler, EvaluationStage
    from evocore.optimizers.cmaes import CMAESOptimizer
    from evocore.optimizers.ga import GeneticAlgorithmOptimizer
    from evocore.results import OptimizationBatchResult, OptimizationResult
    from evocore.search_space import Gene, GeneSpace, Solution, SolutionSet
    from evocore.surrogates import InverseDistanceAdvisor, SurrogateScore

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
