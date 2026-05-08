"""vNext candidate, evaluation, and telemetry primitives."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from evocore.exceptions import ConfigurationError, FitnessError
from evocore.individual import GeneValue

CandidateOrigin = Literal[
    "random",
    "crossover",
    "mutation",
    "cma_sample",
    "surrogate_proposal",
    "memory_seed",
    "restart",
]
CandidateStatus = Literal[
    "proposed",
    "screened",
    "racing",
    "promoted",
    "trusted",
    "eliminated",
    "archived",
]
EvaluationConfidence = Literal["surrogate", "partial", "cached", "trusted_full", "rejected"]


@dataclass(frozen=True)
class Rung:
    """Describe one multi-fidelity evaluation rung."""

    name: str
    budget: float
    promote_fraction: float
    confidence: EvaluationConfidence

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ConfigurationError("rung name must be a non-empty string.")
        if not math.isfinite(float(self.budget)) or self.budget <= 0.0:
            raise ConfigurationError("rung budget must be finite and > 0.")
        if not (0.0 < float(self.promote_fraction) <= 1.0):
            raise ConfigurationError("rung promote_fraction must be in (0, 1].")
        if self.confidence not in ("surrogate", "partial", "cached", "trusted_full", "rejected"):
            raise ConfigurationError("rung confidence is invalid.")


@dataclass(frozen=True)
class CandidateScore:
    """Store one score observation for one candidate and rung."""

    score: float | None
    confidence: EvaluationConfidence
    rung: str
    cost: float
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRecord:
    """Record one evaluator result returned to an ask/tell engine."""

    candidate_id: str
    score: float | None
    confidence: EvaluationConfidence
    rung: str
    cost: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise FitnessError("EvaluationRecord candidate_id must be non-empty.")
        if not self.rung:
            raise FitnessError("EvaluationRecord rung must be non-empty.")
        if self.confidence not in ("surrogate", "partial", "cached", "trusted_full", "rejected"):
            raise FitnessError("EvaluationRecord confidence is invalid.")
        if self.confidence != "rejected":
            if self.score is None or not math.isfinite(float(self.score)):
                raise FitnessError("EvaluationRecord requires a finite score unless rejected.")
        if not math.isfinite(float(self.cost)) or self.cost < 0.0:
            raise FitnessError("EvaluationRecord cost must be finite and >= 0.")


@dataclass
class Candidate:
    """Represent a vNext optimizer candidate with lifecycle and lineage."""

    candidate_id: str
    genes: list[GeneValue]
    params: dict[str, GeneValue] | None = None
    origin: CandidateOrigin = "random"
    parents: Sequence[str] = ()
    event_index: int = 0
    generation: int | None = None
    rung: str | None = None
    status: CandidateStatus = "proposed"
    confidence: EvaluationConfidence | None = None
    cost: float = 0.0
    scores: dict[str, CandidateScore] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def apply_record(self, record: EvaluationRecord) -> None:
        """Apply an evaluation record to this candidate."""
        if record.candidate_id != self.candidate_id:
            raise FitnessError(
                f"EvaluationRecord candidate_id {record.candidate_id!r} does not match "
                f"candidate {self.candidate_id!r}."
            )
        self.rung = record.rung
        self.confidence = record.confidence
        self.cost += record.cost
        self.scores[record.rung] = CandidateScore(
            score=record.score,
            confidence=record.confidence,
            rung=record.rung,
            cost=record.cost,
            metrics=dict(record.metrics),
        )
        self.metadata["metrics"] = dict(record.metrics)
        if record.confidence == "trusted_full":
            self.status = "trusted"
        elif record.confidence == "rejected":
            self.status = "eliminated"
        elif record.confidence in ("partial", "cached"):
            self.status = "racing"
        else:
            self.status = "screened"

    def best_observed_score(self) -> float:
        """Return the best finite score observed for this candidate."""
        values = [score.score for score in self.scores.values() if score.score is not None]
        return max(values) if values else float("-inf")


@dataclass
class OptimizationTelemetry:
    """Aggregate vNext optimizer budget and trial accounting."""

    total_candidates_proposed: int = 0
    unique_candidate_hashes: set[str] = field(default_factory=set)
    candidates_screened: int = 0
    candidates_partial_evaluated: int = 0
    candidates_full_evaluated: int = 0
    promoted_by_rung: dict[str, int] = field(default_factory=dict)
    eliminated_by_rung: dict[str, int] = field(default_factory=dict)
    cost_by_rung: dict[str, float] = field(default_factory=dict)

    def record_proposed(self, count: int) -> None:
        self.total_candidates_proposed += int(count)

    def record_screened(self, count: int) -> None:
        self.candidates_screened += int(count)

    def record_partial(self, count: int, *, rung: str, cost: float) -> None:
        self.candidates_partial_evaluated += int(count)
        self.cost_by_rung[rung] = self.cost_by_rung.get(rung, 0.0) + float(cost)

    def record_full(self, count: int, *, rung: str, cost: float) -> None:
        self.candidates_full_evaluated += int(count)
        self.cost_by_rung[rung] = self.cost_by_rung.get(rung, 0.0) + float(cost)

    def record_promoted(self, count: int, *, rung: str) -> None:
        self.promoted_by_rung[rung] = self.promoted_by_rung.get(rung, 0) + int(count)

    def record_eliminated(self, count: int, *, rung: str) -> None:
        self.eliminated_by_rung[rung] = self.eliminated_by_rung.get(rung, 0) + int(count)


class Evaluator:
    """Base class for vNext evaluators."""

    def evaluate(
        self,
        candidates: Sequence[Candidate],
        rung: Rung,
    ) -> Sequence[EvaluationRecord]:
        """Evaluate candidates for a rung."""
        raise NotImplementedError("Evaluator.evaluate must be implemented by subclasses.")
