"""Python-side individual and population containers."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any

GeneValue = float | int | bool


@dataclass
class Individual:
    """Represent a decoded optimization candidate and its metadata."""

    genes: list[GeneValue]
    fitness: float | None = None
    fitness_valid: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def params(self) -> dict[str, GeneValue] | None:
        """Return named parameters attached to this individual, if available."""
        return self.metadata.get("params")

    def clone(self) -> Individual:
        """Return a shallow clone of the individual state."""
        return Individual(
            genes=list(self.genes),
            fitness=self.fitness,
            fitness_valid=self.fitness_valid,
            metadata=dict(self.metadata),
        )


class Population(Sequence[Individual]):
    """Wrap a sequence of individuals with summary helpers."""

    def __init__(self, individuals: Sequence[Individual]) -> None:
        self._individuals = list(individuals)

    def __len__(self) -> int:
        return len(self._individuals)

    def __iter__(self) -> Iterator[Individual]:
        return iter(self._individuals)

    def __getitem__(self, index: int) -> Individual:
        return self._individuals[index]

    def as_list(self) -> list[Individual]:
        """Return the underlying individuals as a list copy."""
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
        """Return the best `n` individuals by fitness."""
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
        """Return the mean finite fitness across the population."""
        values = self._finite_fitnesses()
        return mean(values) if values else float("nan")

    def std_fitness(self) -> float:
        """Return the population standard deviation of finite fitness values."""
        values = self._finite_fitnesses()
        return pstdev(values) if len(values) > 1 else 0.0

    def diversity(self) -> list[float]:
        """Return per-gene population diversity as population standard deviation."""
        if not self._individuals:
            return []

        gene_count = len(self._individuals[0].genes)
        diversity_values: list[float] = []
        for gene_index in range(gene_count):
            values = [float(individual.genes[gene_index]) for individual in self._individuals]
            diversity_values.append(pstdev(values) if len(values) > 1 else 0.0)
        return diversity_values

    def to_dataframe(self):
        """Convert the population into a pandas DataFrame."""
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
