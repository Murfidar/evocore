"""vNext optimizer advisors."""

from __future__ import annotations

import math
from dataclasses import dataclass

from evocore.evaluation import Candidate, EvaluationConfidence, EvaluationRecord


@dataclass(frozen=True)
class AdvisorScore:
    """Rank one candidate using an advisor."""

    candidate_id: str
    score: float
    confidence: EvaluationConfidence
    reason: str


class InverseDistanceSurrogateAdvisor:
    """Pure-Python inverse-distance baseline surrogate advisor."""

    def __init__(self) -> None:
        self._observations: list[tuple[list[float], float]] = []

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
            self._observations.append(
                ([float(value) for value in candidate.genes], float(record.score))
            )

    def rank(self, candidates: list[Candidate]) -> list[AdvisorScore]:
        """Rank candidates by inverse-distance weighted known scores."""
        rankings: list[AdvisorScore] = []
        for candidate in candidates:
            if not self._observations:
                rankings.append(
                    AdvisorScore(
                        candidate_id=candidate.candidate_id,
                        score=0.0,
                        confidence="surrogate",
                        reason="no_training_data",
                    )
                )
                continue
            genes = [float(value) for value in candidate.genes]
            weighted_sum = 0.0
            weight_total = 0.0
            for observed_genes, observed_score in self._observations:
                distance = math.sqrt(
                    sum(
                        (left - right) ** 2
                        for left, right in zip(genes, observed_genes, strict=True)
                    )
                )
                weight = 1.0 / max(distance, 1e-9)
                weighted_sum += observed_score * weight
                weight_total += weight
            rankings.append(
                AdvisorScore(
                    candidate_id=candidate.candidate_id,
                    score=weighted_sum / weight_total,
                    confidence="surrogate",
                    reason="inverse_distance",
                )
            )
        return sorted(rankings, key=lambda item: item.score, reverse=True)
