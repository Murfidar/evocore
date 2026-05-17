"""vNext optimizer advisors."""

from __future__ import annotations

import math
from dataclasses import dataclass

from evocore.core.errors import ConfigurationError
from evocore.lifecycle import Candidate, EvaluationConfidence, EvaluationRecord
from evocore.search_space import GeneSpace, GeneValue


@dataclass(frozen=True)
class SurrogateScore:
    """Rank one candidate using an advisor."""

    candidate_id: str
    score: float
    confidence: EvaluationConfidence
    reason: str


class InverseDistanceAdvisor:
    """Pure-Python inverse-distance baseline surrogate advisor."""

    def __init__(self, gene_space: GeneSpace | None = None) -> None:
        self.gene_space = gene_space
        self._observations: list[tuple[list[GeneValue], float]] = []

    def _feature_with_space(self, genes: list[GeneValue]) -> list[float]:
        if self.gene_space is None:
            raise ConfigurationError("gene_space is required for bounded feature encoding.")
        if len(genes) != self.gene_space.length:
            raise ConfigurationError(
                f"Expected {self.gene_space.length} genes for advisor encoding, got {len(genes)}."
            )

        features: list[float] = []
        for value, gene in zip(genes, self.gene_space.genes, strict=True):
            if gene.kind == "bool":
                features.append(1.0 if bool(value) else 0.0)
                continue
            low = float(gene.low)
            high = float(gene.high)
            if high == low:
                features.append(0.0)
            else:
                normalized = (float(value) - low) / (high - low)
                features.append(min(1.0, max(0.0, normalized)))
        return features

    @staticmethod
    def _numeric_value(value: GeneValue) -> float:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        return float(value)

    def _inferred_ranges(self, candidates: list[Candidate]) -> list[tuple[float, float]]:
        all_genes = [genes for genes, _score in self._observations] + [
            list(candidate.genes) for candidate in candidates
        ]
        if not all_genes:
            return []
        width = len(all_genes[0])
        ranges: list[tuple[float, float]] = []
        for index in range(width):
            values = [self._numeric_value(genes[index]) for genes in all_genes]
            ranges.append((min(values), max(values)))
        return ranges

    def _feature_with_ranges(
        self, genes: list[GeneValue], ranges: list[tuple[float, float]]
    ) -> list[float]:
        features: list[float] = []
        for value, (low, high) in zip(genes, ranges, strict=True):
            if high == low:
                features.append(0.0)
            else:
                features.append((self._numeric_value(value) - low) / (high - low))
        return features

    def observe(
        self,
        records: list[EvaluationRecord],
        *,
        candidates: dict[str, Candidate],
    ) -> None:
        """Observe trusted records for surrogate ranking."""
        for record in records:
            if record.confidence != "trusted_full" or record.score is None:
                continue
            candidate = candidates[record.candidate_id]
            self._observations.append((list(candidate.genes), float(record.score)))

    def rank(self, candidates: list[Candidate]) -> list[SurrogateScore]:
        """Rank candidates by inverse-distance weighted known scores."""
        rankings: list[SurrogateScore] = []
        ranges = [] if self.gene_space is not None else self._inferred_ranges(candidates)
        for candidate in candidates:
            if not self._observations:
                rankings.append(
                    SurrogateScore(
                        candidate_id=candidate.candidate_id,
                        score=0.0,
                        confidence="surrogate",
                        reason="no_training_data",
                    )
                )
                continue
            genes = (
                self._feature_with_space(list(candidate.genes))
                if self.gene_space is not None
                else self._feature_with_ranges(list(candidate.genes), ranges)
            )
            weighted_sum = 0.0
            weight_total = 0.0
            for observed_genes, observed_score in self._observations:
                observed_features = (
                    self._feature_with_space(observed_genes)
                    if self.gene_space is not None
                    else self._feature_with_ranges(observed_genes, ranges)
                )
                distance = math.sqrt(
                    sum(
                        (left - right) ** 2
                        for left, right in zip(genes, observed_features, strict=True)
                    )
                )
                weight = 1.0 / max(distance, 1e-9)
                weighted_sum += observed_score * weight
                weight_total += weight
            rankings.append(
                SurrogateScore(
                    candidate_id=candidate.candidate_id,
                    score=weighted_sum / weight_total,
                    confidence="surrogate",
                    reason="inverse_distance",
                )
            )
        return sorted(rankings, key=lambda item: item.score, reverse=True)
