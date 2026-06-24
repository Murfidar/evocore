"""Mixed-variable CMA foundations for vNext."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from evocore.core.errors import ConfigurationError


@dataclass(frozen=True)
class IntegerMarginDistribution:
    """Convert continuous integer samples into margin-protected probabilities."""

    low: int
    high: int
    min_probability: float = 0.02

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ConfigurationError("IntegerMarginDistribution requires low <= high.")
        if not (0.0 < self.min_probability < 1.0):
            raise ConfigurationError(
                "IntegerMarginDistribution min_probability must be in (0, 1)."
            )

    def probabilities(self, *, mean: float, sigma: float) -> dict[int, float]:
        """Return normalized integer probabilities with a minimum margin."""
        if sigma <= 0.0 or not math.isfinite(sigma):
            raise ConfigurationError("IntegerMarginDistribution sigma must be finite and > 0.")
        log_weights: dict[int, float] = {}
        for value in range(self.low, self.high + 1):
            z = (float(value) - float(mean)) / sigma
            log_weights[value] = -0.5 * z * z
        max_log_weight = max(log_weights.values())
        raw = {
            key: math.exp(value - max_log_weight) for key, value in log_weights.items()
        }
        total = sum(raw.values())
        probabilities = {key: value / total for key, value in raw.items()}
        categories = self.high - self.low + 1
        floor_total = self.min_probability * categories
        if floor_total >= 1.0:
            raise ConfigurationError(
                "IntegerMarginDistribution min_probability is too large for range."
            )
        adjusted = {
            key: self.min_probability + (1.0 - floor_total) * value
            for key, value in probabilities.items()
        }
        drift = 1.0 - sum(adjusted.values())
        if drift:
            best_key = max(adjusted, key=adjusted.__getitem__)
            adjusted[best_key] += drift
        return adjusted


@dataclass
class CategoricalDistributionState:
    """Maintain a categorical distribution for mixed-variable CMA."""

    categories: Sequence[int]
    learning_rate: float = 0.20
    probabilities: dict[int, float] = field(init=False)

    def __post_init__(self) -> None:
        if not self.categories:
            raise ConfigurationError(
                "CategoricalDistributionState requires at least one category."
            )
        if len(set(self.categories)) != len(self.categories):
            raise ConfigurationError("CategoricalDistributionState categories must be unique.")
        if not (0.0 < self.learning_rate <= 1.0):
            raise ConfigurationError(
                "CategoricalDistributionState learning_rate must be in (0, 1]."
            )
        probability = 1.0 / len(self.categories)
        self.probabilities = dict.fromkeys(self.categories, probability)

    def update(self, *, weighted_observations: list[tuple[int, float]]) -> None:
        """Move probability mass toward weighted observed categories."""
        target = dict.fromkeys(self.categories, 0.0)
        for category, weight in weighted_observations:
            if category not in target:
                raise ConfigurationError(f"unknown category: {category!r}")
            target[category] += max(float(weight), 0.0)
        total = sum(target.values())
        if total <= 0.0:
            return
        target = {category: value / total for category, value in target.items()}
        for category in self.categories:
            self.probabilities[category] = (1.0 - self.learning_rate) * self.probabilities[
                category
            ] + self.learning_rate * target[category]
        normalizer = sum(self.probabilities.values())
        self.probabilities = {
            category: value / normalizer for category, value in self.probabilities.items()
        }
