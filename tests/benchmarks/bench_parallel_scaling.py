import time

from evocore import GeneSpace, GeneticAlgorithmOptimizer
from tests.vnext_helpers import IndividualEvaluator


def sphere(ind):
    return -sum(x * x for x in ind.values)


def test_run_multiple_parallel_scaling_smoke():
    engine = GeneticAlgorithmOptimizer(
        GeneSpace.uniform(-5.0, 5.0, 10), population_size=60, max_generations=20, seed=42
    )

    started = time.perf_counter()
    sequential = engine.run_multiple(IndividualEvaluator(sphere), n_runs=2, run_parallel=False)
    elapsed = time.perf_counter() - started

    assert sequential.n_runs == 2
    print(f"sequential multi-run elapsed={elapsed:.3f}s")
