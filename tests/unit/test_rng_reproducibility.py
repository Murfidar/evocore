import numpy as np

from evocore import GeneSpace, GeneticAlgorithmOptimizer
from tests.vnext_helpers import IndividualEvaluator


def sphere(ind):
    return -sum(x * x for x in ind.values)


def numpy_sphere(ind):
    arr = np.array(ind.values)
    return float(-np.sum(arr * arr))


def test_same_seed_engines_return_identical_results():
    evaluator = IndividualEvaluator(sphere)
    left = GeneticAlgorithmOptimizer(
        GeneSpace.uniform(-5.0, 5.0, 5), population_size=30, max_generations=8, seed=42
    )
    right = GeneticAlgorithmOptimizer(
        GeneSpace.uniform(-5.0, 5.0, 5), population_size=30, max_generations=8, seed=42
    )

    r1 = left.run(evaluator)
    r2 = right.run(evaluator)

    assert r1.best_score == r2.best_score
    assert r1.best_solution.values == r2.best_solution.values


def test_sequential_and_thread_parallel_identical():
    seq = GeneticAlgorithmOptimizer(
        GeneSpace.uniform(-5.0, 5.0, 5),
        population_size=30,
        max_generations=8,
        parallel="none",
        seed=99,
    )
    thr = GeneticAlgorithmOptimizer(
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

    assert r_seq.best_score == r_thr.best_score
    assert r_seq.best_solution.values == r_thr.best_solution.values


def test_n_workers_does_not_affect_results():
    results = []
    for n_workers in [1, 2, 4]:
        engine = GeneticAlgorithmOptimizer(
            GeneSpace.uniform(-5.0, 5.0, 5),
            population_size=30,
            max_generations=8,
            parallel="thread",
            n_workers=n_workers,
            seed=123,
        )
        results.append(engine.run(IndividualEvaluator(numpy_sphere)).best_solution.values)

    assert results[0] == results[1] == results[2]


def test_different_seeds_diverge():
    e1 = GeneticAlgorithmOptimizer(
        GeneSpace.uniform(-5.0, 5.0, 5), population_size=20, max_generations=4, seed=1
    )
    e2 = GeneticAlgorithmOptimizer(
        GeneSpace.uniform(-5.0, 5.0, 5), population_size=20, max_generations=4, seed=2
    )

    evaluator = IndividualEvaluator(sphere)

    assert e1.run(evaluator).best_solution.values != e2.run(evaluator).best_solution.values


def test_multi_run_child_seeds_are_independent():
    engine = GeneticAlgorithmOptimizer(
        GeneSpace.uniform(-5.0, 5.0, 5), population_size=20, max_generations=4, seed=42
    )

    multi = engine.run_multiple(IndividualEvaluator(sphere), n_runs=5)

    assert len({tuple(run.best_solution.values) for run in multi.all_runs}) > 1
