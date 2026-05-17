import numpy as np

from evocore import GAEngine, GeneSpace
from tests.vnext_helpers import IndividualEvaluator


def sphere(ind):
    return -sum(x * x for x in ind.genes)


def numpy_sphere(ind):
    arr = np.array(ind.genes)
    return float(-np.sum(arr * arr))


def test_same_seed_engines_return_identical_results():
    evaluator = IndividualEvaluator(sphere)
    left = GAEngine(
        GeneSpace.uniform(-5.0, 5.0, 5), population_size=30, max_generations=8, seed=42
    )
    right = GAEngine(
        GeneSpace.uniform(-5.0, 5.0, 5), population_size=30, max_generations=8, seed=42
    )

    r1 = left.run(evaluator)
    r2 = right.run(evaluator)

    assert r1.best_fitness == r2.best_fitness
    assert r1.best_individual.genes == r2.best_individual.genes


def test_sequential_and_thread_parallel_identical():
    seq = GAEngine(
        GeneSpace.uniform(-5.0, 5.0, 5),
        population_size=30,
        max_generations=8,
        parallel="none",
        seed=99,
    )
    thr = GAEngine(
        GeneSpace.uniform(-5.0, 5.0, 5),
        population_size=30,
        max_generations=8,
        parallel="thread",
        n_workers=4,
        seed=99,
    )

    evaluator = IndividualEvaluator(numpy_sphere)
    r_seq = seq.run(evaluator)
    r_thr = thr.run(evaluator)

    assert r_seq.best_fitness == r_thr.best_fitness
    assert r_seq.best_individual.genes == r_thr.best_individual.genes


def test_n_workers_does_not_affect_results():
    results = []
    for n_workers in [1, 2, 4]:
        engine = GAEngine(
            GeneSpace.uniform(-5.0, 5.0, 5),
            population_size=30,
            max_generations=8,
            parallel="thread",
            n_workers=n_workers,
            seed=123,
        )
        results.append(engine.run(IndividualEvaluator(numpy_sphere)).best_individual.genes)

    assert results[0] == results[1] == results[2]


def test_different_seeds_diverge():
    e1 = GAEngine(GeneSpace.uniform(-5.0, 5.0, 5), population_size=20, max_generations=4, seed=1)
    e2 = GAEngine(GeneSpace.uniform(-5.0, 5.0, 5), population_size=20, max_generations=4, seed=2)

    evaluator = IndividualEvaluator(sphere)

    assert e1.run(evaluator).best_individual.genes != e2.run(evaluator).best_individual.genes


def test_multi_run_child_seeds_are_independent():
    engine = GAEngine(
        GeneSpace.uniform(-5.0, 5.0, 5), population_size=20, max_generations=4, seed=42
    )

    multi = engine.run_multiple(IndividualEvaluator(sphere), n_runs=5)

    assert len({tuple(run.best_individual.genes) for run in multi.all_runs}) > 1
