"""
Smoke tests for evaluate_sequential and evaluate_parallel_rayon via PyO3.
"""

import numpy as np

from evocore._core import evaluate_parallel_rayon, evaluate_sequential


def neg_sphere(genes):
    return -sum(x**2 for x in genes)


def numpy_neg_sphere(genes):
    return float(-np.sum(np.array(genes) ** 2))


class TestEvaluateSequential:
    def test_correct_length(self):
        pop = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        result = evaluate_sequential(pop, neg_sphere)
        assert len(result) == 3

    def test_correct_values(self):
        pop = [[1.0, 2.0], [3.0, 4.0]]
        result = evaluate_sequential(pop, neg_sphere)
        assert abs(result[0] - (-5.0)) < 1e-10, f"expected -5.0, got {result[0]}"
        assert abs(result[1] - (-25.0)) < 1e-10, f"expected -25.0, got {result[1]}"

    def test_returns_list_of_floats(self):
        pop = [[0.0, 1.0], [2.0, 3.0]]
        result = evaluate_sequential(pop, neg_sphere)
        assert all(isinstance(value, float) for value in result)

    def test_empty_population(self):
        result = evaluate_sequential([], neg_sphere)
        assert result == []

    def test_single_individual(self):
        result = evaluate_sequential([[1.0, 1.0, 1.0]], neg_sphere)
        assert abs(result[0] - (-3.0)) < 1e-10


class TestEvaluateParallelRayon:
    def test_correct_length(self):
        pop = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        result = evaluate_parallel_rayon(pop, numpy_neg_sphere, 2)
        assert len(result) == 3

    def test_matches_sequential_for_numpy_fitness(self):
        pop = [[float(i), float(i * 2)] for i in range(20)]
        seq = evaluate_sequential(pop, numpy_neg_sphere)
        par = evaluate_parallel_rayon(pop, numpy_neg_sphere, 4)
        for i, (seq_value, par_value) in enumerate(zip(seq, par)):
            assert abs(seq_value - par_value) < 1e-10, (
                f"mismatch at {i}: seq={seq_value}, par={par_value}"
            )

    def test_n_threads_1_matches_sequential(self):
        pop = [[float(i)] for i in range(10)]
        seq = evaluate_sequential(pop, numpy_neg_sphere)
        par = evaluate_parallel_rayon(pop, numpy_neg_sphere, 1)
        for seq_value, par_value in zip(seq, par):
            assert abs(seq_value - par_value) < 1e-10

    def test_empty_population(self):
        result = evaluate_parallel_rayon([], numpy_neg_sphere, 2)
        assert result == []


class TestEvaluationDeterminism:
    def test_same_genes_same_fitness(self):
        pop = [[1.0, 2.0, 3.0]] * 5
        r1 = evaluate_sequential(pop, neg_sphere)
        r2 = evaluate_sequential(pop, neg_sphere)
        assert r1 == r2

    def test_different_genes_different_fitness(self):
        pop1 = [[1.0, 2.0], [3.0, 4.0]]
        pop2 = [[5.0, 6.0], [7.0, 8.0]]
        r1 = evaluate_sequential(pop1, neg_sphere)
        r2 = evaluate_sequential(pop2, neg_sphere)
        assert r1 != r2
