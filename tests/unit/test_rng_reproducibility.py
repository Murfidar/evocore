import numpy as np

from evocore import GAEngine, GeneSpace


def sphere(ind):
    return -sum(x * x for x in ind.genes)


def numpy_sphere(ind):
    arr = np.array(ind.genes)
    return float(-np.sum(arr * arr))


def test_run_twice_same_engine_identical_results():
    engine = GAEngine(GeneSpace.uniform(-5.0, 5.0, 5), population_size=30, generations=8, seed=42)

    r1 = engine.run(sphere)
    r2 = engine.run(sphere)

    assert r1.best_fitness == r2.best_fitness
    assert r1.best_individual.genes == r2.best_individual.genes


def test_sequential_and_thread_parallel_identical():
    seq = GAEngine(
        GeneSpace.uniform(-5.0, 5.0, 5),
        population_size=30,
        generations=8,
        parallel="none",
        seed=99,
    )
    thr = GAEngine(
        GeneSpace.uniform(-5.0, 5.0, 5),
        population_size=30,
        generations=8,
        parallel="thread",
        n_workers=4,
        seed=99,
    )

    r_seq = seq.run(numpy_sphere)
    r_thr = thr.run(numpy_sphere)

    assert r_seq.best_fitness == r_thr.best_fitness
    assert r_seq.best_individual.genes == r_thr.best_individual.genes


def test_n_workers_does_not_affect_results():
    results = []
    for n_workers in [1, 2, 4]:
        engine = GAEngine(
            GeneSpace.uniform(-5.0, 5.0, 5),
            population_size=30,
            generations=8,
            parallel="thread",
            n_workers=n_workers,
            seed=123,
        )
        results.append(engine.run(numpy_sphere).best_individual.genes)

    assert results[0] == results[1] == results[2]


def test_different_seeds_diverge():
    e1 = GAEngine(GeneSpace.uniform(-5.0, 5.0, 5), population_size=20, generations=4, seed=1)
    e2 = GAEngine(GeneSpace.uniform(-5.0, 5.0, 5), population_size=20, generations=4, seed=2)

    assert e1.run(sphere).best_individual.genes != e2.run(sphere).best_individual.genes


def test_multi_run_child_seeds_are_independent():
    engine = GAEngine(GeneSpace.uniform(-5.0, 5.0, 5), population_size=20, generations=4, seed=42)

    multi = engine.run_multiple(sphere, n_runs=5)

    assert len({tuple(run.best_individual.genes) for run in multi.all_runs}) > 1
