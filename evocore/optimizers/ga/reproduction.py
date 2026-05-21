from __future__ import annotations

import math
import random
from collections.abc import Sequence

from evocore import _core
from evocore.core.errors import ConfigurationError
from evocore.optimizers.operators import (
    CrossoverContext,
    MutationContext,
    SelectionContext,
    apply_bounds_policy,
    gene_space_profile,
)
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

        if self._uses_python_reproduction():
            return self._make_offspring_python(
                working_population,
                fitnesses,
                gen,
                offspring_count,
            )

        crossover_params = self.crossover_operator.parameters
        mutation_params = self.mutation_operator.parameters
        selection_params = self.selection_operator.parameters
        sigma_list = self.operators.sigma_abs_list(self._compute_sigma_fraction(gen))
        offspring_encoded = _core.reproduce_population(
            self.operators.encode_population(working_population),
            fitnesses,
            self.crossover_operator.name,
            float(crossover_params.get("probability", self.crossover_prob)),
            float(crossover_params.get("eta", self.crossover_eta)),
            float(crossover_params.get("alpha", self.crossover_alpha)),
            self.mutation_operator.name,
            float(mutation_params.get("probability", self.mutation_prob)),
            sigma_list,
            self.operators.gene_bounds,
            self.operators.gene_kinds,
            self.selection_operator.name,
            int(selection_params.get("tournament_size", self.tournament_size)),
            offspring_count,
            self.seed,
            gen,
            float(
                mutation_params.get(
                    "individual_probability",
                    self.mutation_individual_prob,
                )
            ),
        )
        return self.operators.decode_population(offspring_encoded)

    def _select_parent_indices_python(
        self,
        fitnesses: Sequence[float],
        count: int,
        gen: int,
    ) -> list[int]:
        if self.selection_operator.custom:
            context = SelectionContext(
                gene_space=self.gene_space,
                generation=gen,
                seed=self.seed,
                individual_index=None,
                pair_index=None,
                bounds_policy=self.bounds_policy,
                tournament_size=self.tournament_size,
            )
            selected = list(
                self.selection_operator.implementation.select(fitnesses, count, context)
            )
            valid_indices = [
                index for index in selected if type(index) is int and 0 <= index < len(fitnesses)
            ]
            if len(selected) != count or len(valid_indices) != count:
                raise ConfigurationError("custom selection returned invalid parent indices.")
            return valid_indices
        if self.selection_operator.name == "tournament":
            return _core.tournament_selection(
                fitnesses, count, self.tournament_size, self.seed, gen
            )
        if self.selection_operator.name == "roulette":
            return _core.roulette_selection(fitnesses, count, self.seed, gen)
        if self.selection_operator.name == "rank":
            return _core.rank_selection(fitnesses, count, self.seed, gen)
        raise ConfigurationError(f"unknown selection operator: {self.selection_operator.name!r}")

    def _uses_custom_operator(self) -> bool:
        return any(
            getattr(operator, "custom", False)
            for operator in (
                self.crossover_operator,
                self.mutation_operator,
                self.selection_operator,
            )
        )

    def _uses_mixed_gene_space(self) -> bool:
        return gene_space_profile(self.gene_space) == "mixed"

    def _uses_python_reproduction(self) -> bool:
        return self._uses_custom_operator() or self._uses_mixed_gene_space()

    def _crossover_children_python(
        self,
        left: Sequence[float | int | bool],
        right: Sequence[float | int | bool],
        *,
        gen: int,
        pair_index: int,
    ) -> list[list[float | int | bool]]:
        if self.crossover_operator.custom:
            context = CrossoverContext(
                gene_space=self.gene_space,
                generation=gen,
                seed=self.seed,
                individual_index=None,
                pair_index=pair_index,
                bounds_policy=self.bounds_policy,
                probability=self.crossover_prob,
            )
            child_left, child_right = self.crossover_operator.implementation.crossover(
                left,
                right,
                context,
            )
            return [list(child_left), list(child_right)]

        if self.crossover_prob <= 0.0:
            return [list(left), list(right)]
        apply_xo = True
        if self.crossover_prob < 1.0:
            rng = random.Random(  # noqa: S311 - deterministic optimizer RNG, not cryptography.
                int(_core.py_derive_seed(self.seed, gen, pair_index, _core.OP_CROSSOVER_PROB))
            )
            apply_xo = rng.random() < self.crossover_prob
        if not apply_xo:
            return [list(left), list(right)]

        encoded_left = self.operators.encode_values(left)
        encoded_right = self.operators.encode_values(right)
        if self.crossover_operator.name == "sbx":
            child_left, child_right = _core.simulated_binary_crossover(
                encoded_left,
                encoded_right,
                self.crossover_eta,
                self.seed,
                gen,
                pair_index,
            )
        elif self.crossover_operator.name == "blx":
            child_left, child_right = _core.blend_crossover(
                encoded_left,
                encoded_right,
                self.crossover_alpha,
                self.seed,
                gen,
                pair_index,
            )
        elif self.crossover_operator.name == "one_point":
            child_left, child_right = _core.one_point_crossover(
                encoded_left,
                encoded_right,
                self.seed,
                gen,
                pair_index,
            )
        elif self.crossover_operator.name == "two_point":
            child_left, child_right = _core.two_point_crossover(
                encoded_left,
                encoded_right,
                self.seed,
                gen,
                pair_index,
            )
        elif self.crossover_operator.name == "uniform":
            child_left, child_right = _core.uniform_crossover(
                encoded_left,
                encoded_right,
                0.5,
                self.seed,
                gen,
                pair_index,
            )
        else:
            raise ConfigurationError(
                f"unknown crossover operator: {self.crossover_operator.name!r}"
            )
        return [
            self.operators.decode_values(child_left),
            self.operators.decode_values(child_right),
        ]

    def _mutate_child_python(
        self,
        values: Sequence[float | int | bool],
        *,
        gen: int,
        individual_index: int,
        mutation_sigmas: Sequence[float],
    ) -> list[float | int | bool]:
        if self.mutation_operator.custom:
            context = MutationContext(
                gene_space=self.gene_space,
                generation=gen,
                seed=self.seed,
                individual_index=individual_index,
                pair_index=None,
                bounds_policy=self.bounds_policy,
                probability=self.mutation_prob,
                mutation_sigma=self.mutation_sigma,
                mutation_sigmas=tuple(float(value) for value in mutation_sigmas),
            )
            mutated = list(self.mutation_operator.implementation.mutate(list(values), context))
            return apply_bounds_policy(mutated, self.gene_space, self.bounds_policy)

        rng = random.Random(  # noqa: S311 - deterministic optimizer RNG, not cryptography.
            int(_core.py_derive_seed(self.seed, gen, individual_index, _core.OP_MUTATION))
        )
        if self.mutation_individual_prob <= 0.0:
            return list(values)
        if self.mutation_individual_prob < 1.0 and rng.random() >= self.mutation_individual_prob:
            return list(values)

        mutated = list(values)
        for index, gene in enumerate(self.gene_space.genes):
            if rng.random() >= self.mutation_prob:
                continue
            if self.mutation_operator.name == "gaussian" and gene.kind in ("float", "int"):
                mutated[index] = float(mutated[index]) + rng.gauss(
                    0.0, max(float(mutation_sigmas[index]), 1e-20)
                )
            elif self.mutation_operator.name == "uniform" and gene.kind == "float":
                mutated[index] = rng.uniform(float(gene.low), float(gene.high))
            elif self.mutation_operator.name == "uniform" and gene.kind == "int":
                mutated[index] = rng.randint(int(gene.low), int(gene.high))
            elif (
                self.mutation_operator.name in ("gaussian", "uniform", "bit_flip")
                and gene.kind == "bool"
            ):
                mutated[index] = not bool(mutated[index])
        return apply_bounds_policy(mutated, self.gene_space, self.bounds_policy)

    def _make_offspring_python(
        self,
        working_population: Sequence[Solution],
        fitnesses: Sequence[float],
        gen: int,
        offspring_count: int,
    ) -> list[Solution]:
        mutation_sigmas = self.operators.sigma_abs_list(self._compute_sigma_fraction(gen))
        parent_count = offspring_count + (offspring_count % 2)
        parent_indices = self._select_parent_indices_python(fitnesses, parent_count, gen)
        offspring: list[Solution] = []
        child_index = 0
        for pair_index in range(parent_count // 2):
            left = list(working_population[parent_indices[pair_index * 2]].values)
            right = list(working_population[parent_indices[pair_index * 2 + 1]].values)
            children = self._crossover_children_python(
                left,
                right,
                gen=gen,
                pair_index=pair_index,
            )

            for child_values in children:
                bounded = apply_bounds_policy(child_values, self.gene_space, self.bounds_policy)
                mutated = self._mutate_child_python(
                    bounded,
                    gen=gen,
                    individual_index=child_index,
                    mutation_sigmas=mutation_sigmas,
                )
                offspring.append(
                    self.operators.decode_solution(self.operators.encode_values(mutated))
                )
                child_index += 1
                if len(offspring) >= offspring_count:
                    return offspring
        return offspring


__all__ = ["GeneticAlgorithmReproductionMixin"]
