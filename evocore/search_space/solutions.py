"""Python-side solution and solution-set containers."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any

GeneValue = float | int | bool


@dataclass(init=False)
class Solution:
    """Represent a decoded optimization candidate and its metadata."""

    values: list[GeneValue]
    score: float | None = None
    score_valid: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        values: Sequence[GeneValue],
        score: float | None = None,
        score_valid: bool = False,
        metadata: dict[str, Any] | None = None,
        *,
        fitness: float | None = None,
        fitness_valid: bool | None = None,
    ) -> None:
        self.values = list(values)
        self.score = score if fitness is None else fitness
        self.score_valid = score_valid if fitness_valid is None else fitness_valid
        self.metadata = dict(metadata or {})

    @property
    def genes(self) -> list[GeneValue]:
        """Return decoded values for internal Rust-boundary compatibility."""
        return self.values

    @property
    def fitness(self) -> float | None:
        """Return score for internal GA/CMA compatibility during migration."""
        return self.score

    @fitness.setter
    def fitness(self, value: float | None) -> None:
        self.score = value

    @property
    def fitness_valid(self) -> bool:
        """Return score_valid for internal GA/CMA compatibility during migration."""
        return self.score_valid

    @fitness_valid.setter
    def fitness_valid(self, value: bool) -> None:
        self.score_valid = value

    @property
    def params(self) -> dict[str, GeneValue] | None:
        """Return named parameters attached to this solution, if available."""
        return self.metadata.get("params")

    def clone(self) -> Solution:
        """Return a shallow clone of the solution state."""
        return Solution(
            values=list(self.values),
            score=self.score,
            score_valid=self.score_valid,
            metadata=dict(self.metadata),
        )


class SolutionSet(Sequence[Solution]):
    """Wrap a sequence of solutions with summary helpers."""

    def __init__(self, solutions: Sequence[Solution]) -> None:
        self._solutions = list(solutions)

    def __len__(self) -> int:
        return len(self._solutions)

    def __iter__(self) -> Iterator[Solution]:
        return iter(self._solutions)

    def __getitem__(self, index: int) -> Solution:
        return self._solutions[index]

    def as_list(self) -> list[Solution]:
        """Return the underlying solutions as a list copy."""
        return list(self._solutions)

    @staticmethod
    def _score_key(solution: Solution) -> float:
        value = solution.score
        if value is None or math.isnan(value):
            return float("-inf")
        if value == float("inf"):
            return float("inf")
        return value

    def best(self, n: int = 1) -> list[Solution]:
        """Return the best `n` solutions by score."""
        if n <= 0:
            return []
        return sorted(self._solutions, key=self._score_key, reverse=True)[:n]

    def _finite_scores(self) -> list[float]:
        values: list[float] = []
        for solution in self._solutions:
            if solution.score is not None and math.isfinite(solution.score):
                values.append(float(solution.score))
        return values

    def mean_score(self) -> float:
        """Return the mean finite score across the solution set."""
        values = self._finite_scores()
        return mean(values) if values else float("nan")

    def std_score(self) -> float:
        """Return the solution-set standard deviation of finite scores."""
        values = self._finite_scores()
        return pstdev(values) if len(values) > 1 else 0.0

    def mean_fitness(self) -> float:
        """Return mean_score for internal GA/CMA compatibility during migration."""
        return self.mean_score()

    def std_fitness(self) -> float:
        """Return std_score for internal GA/CMA compatibility during migration."""
        return self.std_score()

    def diversity(self) -> list[float]:
        """Return per-value diversity as SolutionSet standard deviation."""
        if not self._solutions:
            return []

        value_count = len(self._solutions[0].values)
        diversity_values: list[float] = []
        for value_index in range(value_count):
            values = [float(solution.values[value_index]) for solution in self._solutions]
            diversity_values.append(pstdev(values) if len(values) > 1 else 0.0)
        return diversity_values

    def to_dataframe(self):
        """Convert the solution set into a pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "SolutionSet.to_dataframe() requires pandas. Install with: pip install pandas"
            ) from exc

        rows = []
        for solution in self._solutions:
            row = {f"value_{index}": value for index, value in enumerate(solution.values)}
            row["score"] = solution.score
            row["score_valid"] = solution.score_valid
            rows.append(row)
        return pd.DataFrame(rows)
