"""
Smoke tests for the Rust operator functions exposed via PyO3.
These tests verify the Python-callable API surface, not exhaustive correctness
(that is covered by the Rust unit tests). Focus: correct return types/shapes,
determinism from Python, and the f64 encoding contract.
"""

from evocore._core import (
    OP_CROSSOVER,
    OP_MUTATION,
    bit_flip_mutation,
    blend_crossover,
    gaussian_mutation,
    int_gaussian_mutation,
    int_simulated_binary_crossover,
    int_uniform_mutation,
    one_point_crossover,
    py_derive_seed,
    simulated_binary_crossover,
    two_point_crossover,
    uniform_crossover,
    uniform_mutation,
)


class TestFloatOperators:
    def test_blend_crossover_returns_two_lists_of_correct_length(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        c1, c2 = blend_crossover(a, b, 0.5, 42, 0, 0)
        assert len(c1) == 3
        assert len(c2) == 3

    def test_blend_crossover_deterministic_from_python(self):
        a = [1.0, 2.0]
        b = [3.0, 4.0]
        c1a, c2a = blend_crossover(a, b, 0.5, 42, 5, 3)
        c1b, c2b = blend_crossover(a, b, 0.5, 42, 5, 3)
        assert c1a == c1b
        assert c2a == c2b

    def test_sbx_returns_two_lists(self):
        a = [0.0, 1.0, 2.0]
        b = [3.0, 4.0, 5.0]
        c1, c2 = simulated_binary_crossover(a, b, 2.0, 42, 0, 0)
        assert len(c1) == 3
        assert len(c2) == 3

    def test_sbx_conservation(self):
        a = [1.0, 3.0, 5.0]
        b = [2.0, 6.0, 8.0]
        c1, c2 = simulated_binary_crossover(a, b, 2.0, 42, 0, 0)
        for i in range(3):
            assert abs(c1[i] + c2[i] - a[i] - b[i]) < 1e-9

    def test_gaussian_mutation_prob_zero_unchanged(self):
        genes = [5.0] * 5
        result = gaussian_mutation(genes, 1.0, 0.0, 42, 0, 0)
        assert result == genes

    def test_gaussian_mutation_deterministic(self):
        genes = [1.0, 2.0, 3.0]
        r1 = gaussian_mutation(genes, 0.5, 1.0, 42, 0, 0)
        r2 = gaussian_mutation(genes, 0.5, 1.0, 42, 0, 0)
        assert r1 == r2

    def test_uniform_mutation_respects_bounds(self):
        genes = [0.0] * 50
        result = uniform_mutation(genes, -1.0, 1.0, 1.0, 42, 0, 0)
        assert all(-1.0 <= v < 1.0 for v in result)

    def test_uniform_mutation_prob_zero_unchanged(self):
        genes = [3.0] * 5
        result = uniform_mutation(genes, 0.0, 10.0, 0.0, 42, 0, 0)
        assert result == genes


class TestIntegerOperators:
    def test_int_sbx_returns_correct_length(self):
        a = [10.0, 50.0, 100.0]
        b = [20.0, 80.0, 200.0]
        c1, c2 = int_simulated_binary_crossover(a, b, 2.0, 42, 0, 0)
        assert len(c1) == 3
        assert len(c2) == 3

    def test_int_sbx_outputs_are_integers(self):
        a = [10.0, 50.0, 100.0]
        b = [20.0, 80.0, 200.0]
        c1, c2 = int_simulated_binary_crossover(a, b, 2.0, 42, 0, 0)
        assert all(v == int(v) for v in c1)
        assert all(v == int(v) for v in c2)

    def test_int_gaussian_mutation_outputs_are_integers(self):
        genes = [50.0] * 20
        result = int_gaussian_mutation(genes, 5.0, 1.0, 42, 0, 0)
        assert all(v == int(v) for v in result)

    def test_int_gaussian_mutation_prob_zero_unchanged(self):
        genes = [42.0] * 5
        result = int_gaussian_mutation(genes, 10.0, 0.0, 42, 0, 0)
        assert result == genes

    def test_int_uniform_mutation_outputs_are_integers(self):
        genes = [50.0] * 30
        result = int_uniform_mutation(genes, 5.0, 200.0, 1.0, 42, 0, 0)
        assert all(v == int(v) for v in result)

    def test_int_uniform_mutation_respects_bounds(self):
        genes = [50.0] * 100
        result = int_uniform_mutation(genes, 5.0, 200.0, 1.0, 42, 0, 0)
        assert all(5.0 <= v <= 200.0 for v in result)

    def test_int_uniform_mutation_prob_zero_unchanged(self):
        genes = [77.0] * 5
        result = int_uniform_mutation(genes, 1.0, 100.0, 0.0, 42, 0, 0)
        assert result == genes


class TestBinaryOperators:
    def _is_binary(self, vals):
        return all(v == 0.0 or v == 1.0 for v in vals)

    def test_one_point_crossover_lengths(self):
        a = [1.0] * 8
        b = [0.0] * 8
        c1, c2 = one_point_crossover(a, b, 42, 0, 0)
        assert len(c1) == 8
        assert len(c2) == 8

    def test_one_point_crossover_only_binary(self):
        a = [1.0] * 10
        b = [0.0] * 10
        c1, c2 = one_point_crossover(a, b, 42, 0, 0)
        assert self._is_binary(c1)
        assert self._is_binary(c2)

    def test_two_point_crossover_lengths(self):
        a = [1.0] * 10
        b = [0.0] * 10
        c1, c2 = two_point_crossover(a, b, 42, 0, 0)
        assert len(c1) == 10
        assert len(c2) == 10

    def test_two_point_crossover_only_binary(self):
        a = [1.0] * 12
        b = [0.0] * 12
        c1, c2 = two_point_crossover(a, b, 42, 0, 0)
        assert self._is_binary(c1)
        assert self._is_binary(c2)

    def test_uniform_crossover_lengths(self):
        a = [1.0] * 8
        b = [0.0] * 8
        c1, c2 = uniform_crossover(a, b, 0.5, 42, 0, 0)
        assert len(c1) == 8
        assert len(c2) == 8

    def test_uniform_crossover_only_binary(self):
        a = [1.0] * 20
        b = [0.0] * 20
        c1, c2 = uniform_crossover(a, b, 0.5, 42, 0, 0)
        assert self._is_binary(c1)
        assert self._is_binary(c2)

    def test_bit_flip_mutation_prob_zero_unchanged(self):
        genes = [1.0, 0.0, 1.0, 0.0]
        result = bit_flip_mutation(genes, 0.0, 42, 0, 0)
        assert result == genes

    def test_bit_flip_mutation_prob_one_all_flipped(self):
        genes = [1.0, 0.0, 1.0, 0.0]
        result = bit_flip_mutation(genes, 1.0, 42, 0, 0)
        assert result == [0.0, 1.0, 0.0, 1.0]

    def test_bit_flip_mutation_only_binary(self):
        genes = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        result = bit_flip_mutation(genes, 0.5, 42, 0, 0)
        assert self._is_binary(result)


class TestOperatorDeterminism:
    def test_different_master_seeds_diverge(self):
        genes = [1.0, 2.0, 3.0, 4.0, 5.0]
        r1 = gaussian_mutation(genes, 1.0, 1.0, 1, 0, 0)
        r2 = gaussian_mutation(genes, 1.0, 1.0, 2, 0, 0)
        assert r1 != r2

    def test_different_generations_diverge(self):
        genes = [1.0, 2.0, 3.0, 4.0, 5.0]
        r1 = gaussian_mutation(genes, 1.0, 1.0, 42, 0, 0)
        r2 = gaussian_mutation(genes, 1.0, 1.0, 42, 1, 0)
        assert r1 != r2

    def test_different_individual_indices_diverge(self):
        genes = [1.0, 2.0, 3.0, 4.0, 5.0]
        r1 = gaussian_mutation(genes, 1.0, 1.0, 42, 0, 0)
        r2 = gaussian_mutation(genes, 1.0, 1.0, 42, 0, 1)
        assert r1 != r2

    def test_crossover_and_mutation_use_different_op_constants(self):
        seed_xo = py_derive_seed(42, 0, 0, OP_CROSSOVER)
        seed_mut = py_derive_seed(42, 0, 0, OP_MUTATION)
        assert seed_xo != seed_mut
