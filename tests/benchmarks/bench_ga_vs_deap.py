import time

import pytest

from evocore import GeneSpace, GeneticAlgorithmOptimizer
from tests.vnext_helpers import IndividualEvaluator


def sphere(ind):
    return -sum(x * x for x in ind.genes)


def test_evocore_ga_wall_time_smoke():
    engine = GeneticAlgorithmOptimizer(
        GeneSpace.uniform(-5.0, 5.0, 20), population_size=300, max_generations=40, seed=42
    )

    started = time.perf_counter()
    result = engine.run(IndividualEvaluator(sphere))
    elapsed = time.perf_counter() - started

    assert result.best_score <= 0.0
    print(f"evocore elapsed={elapsed:.3f}s")


def test_deap_comparison_optional():
    pytest.importorskip("deap")
