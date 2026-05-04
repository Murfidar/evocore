from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any, Iterator, Sequence

GeneValue = float | int | bool


@dataclass
class Individual:
    genes: list[GeneValue]
    fitness: float | None = None
    fitness_valid: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def params(self) -> dict[str, GeneValue] | None:
        return self.metadata.get("params")

    def clone(self) -> "Individual":
        return Individual(
            genes=list(self.genes),
            fitness=self.fitness,
            fitness_valid=self.fitness_valid,
            metadata=dict(self.metadata),
        )


class Population(Sequence[Individual]):
    def __init__(self, individuals: Sequence[Individual]) -> None:
        self._individuals = list(individuals)

    def __len__(self) -> int:
        return len(self._individuals)

    def __iter__(self) -> Iterator[Individual]:
        return iter(self._individuals)

    def __getitem__(self, index: int) -> Individual:
        return self._individuals[index]

    def as_list(self) -> list[Individual]:
        return list(self._individuals)

    @staticmethod
    def _fitness_key(individual: Individual) -> float:
        value = individual.fitness
        if value is None or math.isnan(value):
            return float("-inf")
        if value == float("inf"):
            return float("inf")
        return value

    def best(self, n: int = 1) -> list[Individual]:
        if n <= 0:
            return []
        return sorted(self._individuals, key=self._fitness_key, reverse=True)[:n]

    def _finite_fitnesses(self) -> list[float]:
        values: list[float] = []
        for individual in self._individuals:
            if individual.fitness is not None and math.isfinite(individual.fitness):
                values.append(float(individual.fitness))
        return values

    def mean_fitness(self) -> float:
        values = self._finite_fitnesses()
        return mean(values) if values else float("nan")

    def std_fitness(self) -> float:
        values = self._finite_fitnesses()
        return pstdev(values) if len(values) > 1 else 0.0

    def diversity(self) -> list[float]:
        if not self._individuals:
            return []

        gene_count = len(self._individuals[0].genes)
        diversity_values: list[float] = []
        for gene_index in range(gene_count):
            values = [float(individual.genes[gene_index]) for individual in self._individuals]
            diversity_values.append(pstdev(values) if len(values) > 1 else 0.0)
        return diversity_values

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "Population.to_dataframe() requires pandas. Install with: pip install pandas"
            ) from exc

        rows = []
        for individual in self._individuals:
            row = {f"gene_{index}": value for index, value in enumerate(individual.genes)}
            row["fitness"] = individual.fitness
            row["fitness_valid"] = individual.fitness_valid
            rows.append(row)
        return pd.DataFrame(rows)
