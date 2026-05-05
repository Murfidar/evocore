import time

from evocore import GAEngine, GeneSpace


def sphere(ind):
    return -sum(x * x for x in ind.genes)


def test_run_multiple_parallel_scaling_smoke():
    engine = GAEngine(
        GeneSpace.uniform(-5.0, 5.0, 10), population_size=60, generations=20, seed=42
    )

    started = time.perf_counter()
    sequential = engine.run_multiple(sphere, n_runs=2, run_parallel=False)
    elapsed = time.perf_counter() - started

    assert sequential.n_runs == 2
    print(f"sequential multi-run elapsed={elapsed:.3f}s")
