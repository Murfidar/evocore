from __future__ import annotations

import math
from collections.abc import Sequence

from evocore import _core
from evocore.core.errors import ConfigurationError
from evocore.search_space import Solution, SolutionSet


class GeneticAlgorithmReproductionMixin:
    """Rust-backed GA initialization and reproduction helpers."""

    def _compute_sigma_fraction(self, gen: int) -> float:
        if self.max_generations <= 1 or self.mutation_sigma_schedule == "constant":
            return self.mutation_sigma

        t = gen / max(1, self.max_generations - 1)
        if self.mutation_sigma_schedule == "linear_decay":
            return self.mutation_sigma + t * (self.mutation_sigma_end - self.mutation_sigma)
        if self.mutation_sigma_schedule == "cosine_decay":
            cosine = 0.5 * (1.0 + math.cos(math.pi * t))
            return self.mutation_sigma_end + cosine * (
                self.mutation_sigma - self.mutation_sigma_end
            )
        raise ConfigurationError("unknown mutation_sigma_schedule")

    def _initial_population(self) -> list[Solution]:
        population_size = self.population_size
        if self.max_evaluations is not None:
            population_size = min(population_size, self.max_evaluations)
        encoded = _core.init_population(
            self.operators.gene_bounds,
            self.operators.gene_kinds,
            population_size,
            self.seed,
        )
        return self.operators.decode_population(encoded)

    def _clone_elites(self, solutions: Sequence[Solution]) -> list[Solution]:
        if self.elitism == 0:
            return []
        return [solution.clone() for solution in SolutionSet(solutions).best(self.elitism)]

    def _make_offspring(
        self,
        working_population: Sequence[Solution],
        fitnesses: Sequence[float],
        gen: int,
        offspring_count: int,
    ) -> list[Solution]:
        if offspring_count <= 0:
            return []

        sigma_list = self.operators.sigma_abs_list(self._compute_sigma_fraction(gen))
        offspring_encoded = _core.reproduce_population(
            self.operators.encode_population(working_population),
            fitnesses,
            self.crossover,
            self.crossover_prob,
            self.crossover_eta,
            self.crossover_alpha,
            self.mutation,
            self.mutation_prob,
            sigma_list,
            self.operators.gene_bounds,
            self.operators.gene_kinds,
            self.selection,
            self.tournament_size,
            offspring_count,
            self.seed,
            gen,
            self.mutation_individual_prob,
        )
        return self.operators.decode_population(offspring_encoded)


__all__ = ["GeneticAlgorithmReproductionMixin"]
