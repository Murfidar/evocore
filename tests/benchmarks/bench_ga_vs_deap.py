import time

import pytest

from evocore import GAEngine, GeneSpace


def sphere(ind):
    return -sum(x * x for x in ind.genes)


def test_evocore_ga_wall_time_smoke():
    engine = GAEngine(GeneSpace.uniform(-5.0, 5.0, 20), population_size=300, generations=40, seed=42)

    started = time.perf_counter()
    result = engine.run(sphere)
    elapsed = time.perf_counter() - started

    assert result.best_fitness <= 0.0
    print(f"evocore elapsed={elapsed:.3f}s")


def test_deap_comparison_optional():
    pytest.importorskip("deap")
