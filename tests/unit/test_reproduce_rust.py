"""
Smoke tests for init_population and reproduce_population via PyO3.
Focus: correct sizes, type constraints (int/bool), bounds, and determinism.
"""

import pytest

from evocore._core import init_population, reproduce_population


def make_float_pop(pop_size: int, gene_len: int, seed: int = 42):
    return init_population([(-5.0, 5.0)] * gene_len, ["float"] * gene_len, pop_size, seed)


def run_reproduce(pop, gene_len: int, pop_size: int, seed: int = 42, gen: int = 0):
    fitnesses = [float(i) for i in range(len(pop))]
    return reproduce_population(
        pop,
        fitnesses,
        "sbx",
        0.9,
        2.0,
        0.5,
        "gaussian",
        0.1,
        [0.5] * gene_len,
        [(-5.0, 5.0)] * gene_len,
        ["float"] * gene_len,
        "tournament",
        3,
        pop_size,
        seed,
        gen,
    )


class TestInitPopulation:
    def test_correct_population_size(self):
        pop = make_float_pop(20, 5)
        assert len(pop) == 20

    def test_correct_gene_length(self):
        pop = make_float_pop(10, 7)
        assert all(len(ind) == 7 for ind in pop)

    def test_float_genes_within_bounds(self):
        bounds = [(-2.0, 2.0), (0.0, 10.0), (-1.0, 1.0)]
        pop = init_population(bounds, ["float"] * 3, 50, 42)
        for ind in pop:
            for i, gene in enumerate(ind):
                low, high = bounds[i]
                assert low <= gene < high, f"gene[{i}]={gene} outside [{low}, {high})"

    def test_int_genes_are_integer_valued(self):
        bounds = [(5.0, 200.0), (10.0, 500.0)]
        pop = init_population(bounds, ["int", "int"], 20, 42)
        for ind in pop:
            for gene in ind:
                assert gene == int(gene), f"int gene {gene} is not integer-valued"

    def test_bool_genes_are_binary(self):
        bounds = [(0.0, 1.0)] * 8
        pop = init_population(bounds, ["bool"] * 8, 20, 42)
        for ind in pop:
            for gene in ind:
                assert gene in (0.0, 1.0), f"bool gene {gene} is not 0.0 or 1.0"

    def test_deterministic(self):
        p1 = make_float_pop(10, 4, seed=7)
        p2 = make_float_pop(10, 4, seed=7)
        assert p1 == p2

    def test_different_seeds_diverge(self):
        p1 = make_float_pop(10, 4, seed=1)
        p2 = make_float_pop(10, 4, seed=2)
        assert p1[0] != p2[0]

    def test_invalid_gene_kind_raises(self):
        with pytest.raises(Exception, match="gene kind"):
            init_population([(0.0, 1.0)], ["quantum"], 5, 42)

    def test_fixed_float_and_int_bounds_initialize_to_fixed_values(self):
        bounds = [(1.25, 1.25), (2.0, 2.0), (-5.0, 5.0)]
        pop = init_population(bounds, ["float", "int", "float"], 12, 42)

        assert len(pop) == 12
        assert all(ind[0] == 1.25 for ind in pop)
        assert all(ind[1] == 2.0 for ind in pop)
        assert any(ind[2] != pop[0][2] for ind in pop[1:])


class TestReproducePopulation:
    def test_returns_correct_population_size(self):
        pop = make_float_pop(20, 5)
        new_pop = run_reproduce(pop, 5, 20)
        assert len(new_pop) == 20

    def test_returns_correct_gene_length(self):
        pop = make_float_pop(10, 6)
        new_pop = run_reproduce(pop, 6, 10)
        assert all(len(ind) == 6 for ind in new_pop)

    def test_float_genes_within_bounds(self):
        pop = make_float_pop(30, 4)
        fitnesses = list(range(30))
        new_pop = reproduce_population(
            pop,
            fitnesses,
            "sbx",
            0.9,
            2.0,
            0.5,
            "gaussian",
            0.1,
            [0.5] * 4,
            [(-5.0, 5.0)] * 4,
            ["float"] * 4,
            "tournament",
            3,
            30,
            42,
            0,
        )
        for ind in new_pop:
            for gene in ind:
                assert -5.0 <= gene <= 5.0, f"float gene {gene} outside [-5, 5]"

    def test_int_genes_always_integer_valued(self):
        bounds = [(0.0, 20.0)] * 3
        pop = init_population(bounds, ["int"] * 3, 20, 42)
        fitnesses = list(range(20))
        new_pop = reproduce_population(
            pop,
            fitnesses,
            "sbx",
            0.9,
            2.0,
            0.5,
            "gaussian",
            0.5,
            [2.0] * 3,
            bounds,
            ["int"] * 3,
            "tournament",
            2,
            20,
            42,
            0,
        )
        for ind in new_pop:
            for gene in ind:
                assert gene == int(gene), f"int gene {gene} not integer-valued"

    def test_bool_genes_always_binary(self):
        bounds = [(0.0, 1.0)] * 8
        pop = init_population(bounds, ["bool"] * 8, 20, 42)
        fitnesses = list(range(20))
        new_pop = reproduce_population(
            pop,
            fitnesses,
            "one_point",
            0.8,
            2.0,
            0.5,
            "bit_flip",
            0.1,
            [0.0] * 8,
            bounds,
            ["bool"] * 8,
            "tournament",
            2,
            20,
            42,
            0,
        )
        for ind in new_pop:
            for gene in ind:
                assert gene in (0.0, 1.0), f"bool gene {gene} not 0.0 or 1.0"

    def test_deterministic(self):
        pop = make_float_pop(10, 4)
        r1 = run_reproduce(pop, 4, 10, seed=77, gen=5)
        r2 = run_reproduce(pop, 4, 10, seed=77, gen=5)
        assert r1 == r2

    def test_different_generations_diverge(self):
        pop = make_float_pop(10, 4)
        r1 = run_reproduce(pop, 4, 10, seed=42, gen=0)
        r2 = run_reproduce(pop, 4, 10, seed=42, gen=1)
        assert r1 != r2, "different generations must produce different offspring"

    def test_invalid_crossover_type_raises(self):
        pop = make_float_pop(10, 3)
        with pytest.raises(Exception, match="crossover_type"):
            reproduce_population(
                pop,
                list(range(10)),
                "quadratic_crossover",
                0.9,
                2.0,
                0.5,
                "gaussian",
                0.1,
                [0.5] * 3,
                [(-1.0, 1.0)] * 3,
                ["float"] * 3,
                "tournament",
                3,
                10,
                42,
                0,
            )

    def test_invalid_mutation_type_raises(self):
        pop = make_float_pop(10, 3)
        with pytest.raises(Exception, match="mutation_type"):
            reproduce_population(
                pop,
                list(range(10)),
                "sbx",
                0.9,
                2.0,
                0.5,
                "quantum_mutation",
                0.1,
                [0.5] * 3,
                [(-1.0, 1.0)] * 3,
                ["float"] * 3,
                "tournament",
                3,
                10,
                42,
                0,
            )

    def test_invalid_selection_type_raises(self):
        pop = make_float_pop(10, 3)
        with pytest.raises(Exception, match="selection_type"):
            reproduce_population(
                pop,
                list(range(10)),
                "sbx",
                0.9,
                2.0,
                0.5,
                "gaussian",
                0.1,
                [0.5] * 3,
                [(-1.0, 1.0)] * 3,
                ["float"] * 3,
                "best_of_best",
                3,
                10,
                42,
                0,
            )

    def test_roulette_selection_mode(self):
        pop = make_float_pop(15, 3)
        fitnesses = [float(i) for i in range(15)]
        new_pop = reproduce_population(
            pop,
            fitnesses,
            "sbx",
            0.9,
            2.0,
            0.5,
            "gaussian",
            0.1,
            [0.3] * 3,
            [(-5.0, 5.0)] * 3,
            ["float"] * 3,
            "roulette",
            3,
            15,
            42,
            0,
        )
        assert len(new_pop) == 15

    def test_rank_selection_mode(self):
        pop = make_float_pop(12, 3)
        fitnesses = [float(i) for i in range(12)]
        new_pop = reproduce_population(
            pop,
            fitnesses,
            "sbx",
            0.9,
            2.0,
            0.5,
            "gaussian",
            0.1,
            [0.3] * 3,
            [(-5.0, 5.0)] * 3,
            ["float"] * 3,
            "rank",
            3,
            12,
            42,
            0,
        )
        assert len(new_pop) == 12

    def test_fixed_numeric_bounds_survive_uniform_mutation_and_crossover(self):
        bounds = [(1.25, 1.25), (2.0, 2.0), (-5.0, 5.0)]
        kinds = ["float", "int", "float"]
        pop = init_population(bounds, kinds, 20, 42)
        new_pop = reproduce_population(
            pop,
            [float(i) for i in range(20)],
            "sbx",
            1.0,
            2.0,
            0.5,
            "uniform",
            1.0,
            [0.0, 0.0, 2.0],
            bounds,
            kinds,
            "tournament",
            3,
            20,
            42,
            0,
        )

        assert len(new_pop) == 20
        assert all(ind[0] == 1.25 for ind in new_pop)
        assert all(ind[1] == 2.0 for ind in new_pop)
        assert all(-5.0 <= ind[2] <= 5.0 for ind in new_pop)

    def test_mutation_individual_probability_zero_skips_all_gene_mutation(self):
        pop = [[0.0, 5.0], [1.0, 10.0], [2.0, 15.0], [3.0, 20.0]]
        new_pop = reproduce_population(
            pop,
            [float(i) for i in range(len(pop))],
            "uniform",
            0.0,
            2.0,
            0.5,
            "uniform",
            1.0,
            [1.0, 1.0],
            [(0.0, 100.0), (0.0, 100.0)],
            ["float", "int"],
            "tournament",
            2,
            8,
            42,
            0,
            0.0,
        )

        parent_values = {tuple(ind) for ind in pop}
        assert len(new_pop) == 8
        assert all(tuple(ind) in parent_values for ind in new_pop)

    def test_blx_crossover_mode(self):
        pop = init_population([(-5.0, 5.0)] * 4, ["float"] * 4, 10, 42)
        new_pop = reproduce_population(
            pop,
            list(range(10)),
            "blx",
            0.9,
            2.0,
            0.5,
            "gaussian",
            0.1,
            [0.3] * 4,
            [(-5.0, 5.0)] * 4,
            ["float"] * 4,
            "tournament",
            3,
            10,
            42,
            0,
        )
        assert len(new_pop) == 10
