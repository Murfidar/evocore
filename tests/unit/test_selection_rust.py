"""
Smoke tests for the Rust selection functions exposed via PyO3.
Focus: correct return types/shapes, NaN safety, and determinism invariant.
"""

from evocore._core import rank_selection, roulette_selection, tournament_selection


class TestTournamentSelection:
    def test_returns_correct_length(self):
        fitnesses = [1.0, 5.0, 3.0, 2.0, 4.0]
        idx = tournament_selection(fitnesses, 4, 2, 42, 0)
        assert len(idx) == 4

    def test_all_indices_in_range(self):
        fitnesses = [1.0, 5.0, 3.0, 2.0, 4.0]
        idx = tournament_selection(fitnesses, 10, 3, 42, 0)
        assert all(0 <= i < len(fitnesses) for i in idx)

    def test_deterministic(self):
        fitnesses = [1.0, 5.0, 3.0]
        a = tournament_selection(fitnesses, 5, 2, 42, 0)
        b = tournament_selection(fitnesses, 5, 2, 42, 0)
        assert a == b

    def test_different_generations_diverge(self):
        fitnesses = [1.0, 2.0, 3.0, 4.0, 5.0]
        a = tournament_selection(fitnesses, 5, 2, 42, 0)
        b = tournament_selection(fitnesses, 5, 2, 42, 1)
        assert a != b

    def test_nan_never_wins_full_tournament(self):
        fitnesses = [float("nan")] * 4 + [99.0]
        idx = tournament_selection(fitnesses, 50, 200, 42, 0)
        assert all(i == 4 for i in idx), "NaN individuals should not win when best is sampled"

    def test_large_tournament_samples_with_replacement_like_deap(self):
        fitnesses = [1.0, 2.0, 3.0, 4.0, 5.0]
        idx = tournament_selection(fitnesses, 200, 5, 42, 0)

        assert any(i != 4 for i in idx)

    def test_returns_list_of_ints(self):
        idx = tournament_selection([1.0, 2.0, 3.0], 3, 2, 42, 0)
        assert all(isinstance(i, int) for i in idx)


class TestRouletteSelection:
    def test_returns_correct_length(self):
        idx = roulette_selection([1.0, 2.0, 3.0, 4.0], 6, 42, 0)
        assert len(idx) == 6

    def test_all_indices_in_range(self):
        fitnesses = [1.0, 5.0, 3.0, 2.0, 4.0]
        idx = roulette_selection(fitnesses, 20, 42, 0)
        assert all(0 <= i < len(fitnesses) for i in idx)

    def test_deterministic(self):
        a = roulette_selection([1.0, 2.0, 3.0], 5, 7, 2)
        b = roulette_selection([1.0, 2.0, 3.0], 5, 7, 2)
        assert a == b

    def test_nan_fitness_effectively_excluded(self):
        fitnesses = [float("nan"), float("nan"), float("nan"), 100.0]
        idx = roulette_selection(fitnesses, 100, 42, 0)
        assert all(i == 3 for i in idx), "NaN individuals should not be selected"


class TestRankSelection:
    def test_returns_correct_length(self):
        idx = rank_selection([1.0, 5.0, 3.0, 2.0, 4.0], 7, 42, 0)
        assert len(idx) == 7

    def test_all_indices_in_range(self):
        fitnesses = [1.0, 5.0, 3.0, 2.0, 4.0]
        idx = rank_selection(fitnesses, 10, 42, 0)
        assert all(0 <= i < len(fitnesses) for i in idx)

    def test_deterministic(self):
        a = rank_selection([3.0, 1.0, 2.0], 5, 13, 5)
        b = rank_selection([3.0, 1.0, 2.0], 5, 13, 5)
        assert a == b

    def test_nan_individual_lowest_rank(self):
        fitnesses = [float("nan"), float("nan"), 50.0]
        idx = rank_selection(fitnesses, 50, 42, 0)
        assert all(i == 2 for i in idx), "NaN Solution must have lowest rank"


class TestSelectionDeterminismInvariant:
    def test_different_seeds_diverge(self):
        fitnesses = [1.0, 2.0, 3.0, 4.0, 5.0]
        a = tournament_selection(fitnesses, 5, 2, 1, 0)
        b = tournament_selection(fitnesses, 5, 2, 2, 0)
        assert a != b

    def test_different_generations_diverge(self):
        fitnesses = [1.0, 2.0, 3.0, 4.0, 5.0]
        a = tournament_selection(fitnesses, 5, 2, 42, 0)
        b = tournament_selection(fitnesses, 5, 2, 42, 1)
        assert a != b
