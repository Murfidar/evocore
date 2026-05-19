"""vNext candidate, evaluation, and telemetry primitives."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from evocore.core.errors import ConfigurationError, FitnessError
from evocore.search_space import GeneValue

if TYPE_CHECKING:
    from evocore.search_space import GeneSpace

Direction = Literal["maximize", "minimize"]
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
STATE_UPDATE_CONFIDENCES: tuple[EvaluationConfidence, ...] = ("trusted_full", "cached")


def is_state_update_confidence(confidence: EvaluationConfidence) -> bool:
    """Return whether a confidence value is eligible for optimizer state updates."""
    return confidence in STATE_UPDATE_CONFIDENCES


def score_for_direction(score: float, direction: Direction) -> float:
    """Return a comparison score where larger is always better."""
    if direction == "maximize":
        return float(score)
    if direction == "minimize":
        return -float(score)
    raise ConfigurationError("direction must be 'maximize' or 'minimize'.")


@dataclass(frozen=True)
class EvaluationStage:
    """Describe one multi-fidelity evaluation stage."""

    name: str
    budget: float
    promote_fraction: float
    confidence: EvaluationConfidence

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ConfigurationError("EvaluationStage name must be a non-empty string.")
        if not math.isfinite(float(self.budget)) or self.budget <= 0.0:
            raise ConfigurationError("EvaluationStage budget must be finite and > 0.")
        if not (0.0 < float(self.promote_fraction) <= 1.0):
            raise ConfigurationError("EvaluationStage promote_fraction must be in (0, 1].")
        if self.confidence not in ("surrogate", "partial", "cached", "trusted_full", "rejected"):
            raise ConfigurationError("EvaluationStage confidence is invalid.")


@dataclass(frozen=True)
class EvaluationContext:
    """Describe the evaluator call context for one ask/tell batch."""

    stage: EvaluationStage | None
    batch_id: str
    event_index: int
    direction: Direction
    budget: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.batch_id:
            raise ConfigurationError("EvaluationContext batch_id must be non-empty.")
        if int(self.event_index) < 0:
            raise ConfigurationError("EvaluationContext event_index must be >= 0.")
        if self.direction not in ("maximize", "minimize"):
            raise ConfigurationError("direction must be 'maximize' or 'minimize'.")
        if self.budget is not None and (
            not math.isfinite(float(self.budget)) or float(self.budget) <= 0.0
        ):
            raise ConfigurationError("EvaluationContext budget must be finite and > 0.")


@dataclass(frozen=True)
class ScoreObservation:
    """Store one score observation for one candidate and stage."""

    score: float | None
    confidence: EvaluationConfidence
    stage: str
    cost: float
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRecord:
    """Record one evaluator result returned to an ask/tell engine."""

    candidate_id: str
    score: float | None
    confidence: EvaluationConfidence
    stage: str
    cost: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    batch_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise FitnessError("EvaluationRecord candidate_id must be non-empty.")
        if not self.stage:
            raise FitnessError("EvaluationRecord stage must be non-empty.")
        if self.confidence not in ("surrogate", "partial", "cached", "trusted_full", "rejected"):
            raise FitnessError("EvaluationRecord confidence is invalid.")
        if self.confidence != "rejected" and (
            self.score is None or not math.isfinite(float(self.score))
        ):
            raise FitnessError("EvaluationRecord requires a finite score unless rejected.")
        if not math.isfinite(float(self.cost)) or self.cost < 0.0:
            raise FitnessError("EvaluationRecord cost must be finite and >= 0.")
        if self.confidence == "rejected":
            if self.score is not None:
                raise FitnessError(
                    "EvaluationRecord with confidence='rejected' requires score=None."
                )
        elif self.score is None or not math.isfinite(float(self.score)):
            raise FitnessError("EvaluationRecord requires a finite score unless rejected.")


@dataclass
class Candidate:
    """Represent a vNext optimizer candidate with lifecycle and lineage."""

    candidate_id: str
    genes: list[GeneValue]
    batch_id: str = ""
    params: dict[str, GeneValue] | None = None
    origin: CandidateOrigin = "random"
    parents: Sequence[str] = ()
    event_index: int = 0
    generation: int | None = None
    stage: str | None = None
    status: CandidateStatus = "proposed"
    confidence: EvaluationConfidence | None = None
    cost: float = 0.0
    scores: dict[str, ScoreObservation] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def apply_record(self, record: EvaluationRecord) -> None:
        """Apply an evaluation record to this candidate."""
        if record.candidate_id != self.candidate_id:
            raise FitnessError(
                f"EvaluationRecord candidate_id {record.candidate_id!r} does not match "
                f"candidate {self.candidate_id!r}."
            )
        if record.batch_id is not None and self.batch_id and record.batch_id != self.batch_id:
            raise FitnessError(
                f"EvaluationRecord batch_id {record.batch_id!r} does not match "
                f"candidate batch {self.batch_id!r}."
            )
        self.stage = record.stage
        self.confidence = record.confidence
        self.cost += record.cost
        self.scores[record.stage] = ScoreObservation(
            score=record.score,
            confidence=record.confidence,
            stage=record.stage,
            cost=record.cost,
            metrics=dict(record.metrics),
            metadata=dict(record.metadata),
        )
        self.metadata["metrics"] = dict(record.metrics)
        self.metadata["record_metadata"] = dict(record.metadata)
        if is_state_update_confidence(record.confidence):
            self.status = "trusted"
        elif record.confidence == "rejected":
            self.status = "eliminated"
        elif record.confidence == "partial":
            self.status = "racing"
        else:
            self.status = "screened"

    def _best_score_for_confidences(
        self,
        direction: Direction,
        confidences: tuple[EvaluationConfidence, ...] | None,
    ) -> float:
        values = [
            score.score
            for score in self.scores.values()
            if score.score is not None and (confidences is None or score.confidence in confidences)
        ]
        if not values:
            return float("inf") if direction == "minimize" else float("-inf")
        if direction == "minimize":
            return min(float(value) for value in values)
        if direction == "maximize":
            return max(float(value) for value in values)
        raise ConfigurationError("direction must be 'maximize' or 'minimize'.")

    def best_observed_score(self, direction: Direction = "maximize") -> float:
        """Return the best raw finite score observed for this candidate."""
        return self._best_score_for_confidences(direction, None)

    def comparison_score(self, direction: Direction = "maximize") -> float:
        """Return the best observed score normalized so larger is better."""
        best = self.best_observed_score(direction)
        if not math.isfinite(best):
            return best if direction == "maximize" else -best
        return score_for_direction(best, direction)

    def best_state_score(self, direction: Direction = "maximize") -> float:
        """Return the best raw score eligible for optimizer state updates."""
        return self._best_score_for_confidences(direction, STATE_UPDATE_CONFIDENCES)

    def state_comparison_score(self, direction: Direction = "maximize") -> float:
        """Return the state-eligible score normalized so larger is better."""
        best = self.best_state_score(direction)
        if not math.isfinite(best):
            return best if direction == "maximize" else -best
        return score_for_direction(best, direction)

    def candidate_hash(self, gene_space: GeneSpace | None = None) -> str:
        """Return a stable hash for this candidate's decoded genes."""
        if gene_space is not None:
            return gene_space.value_hash(self.genes)

        encoded: list[list[Any]] = []
        for value in self.genes:
            if isinstance(value, bool):
                encoded.append(["bool", value])
            elif isinstance(value, int):
                encoded.append(["int", value])
            elif isinstance(value, float):
                encoded.append(["float", value.hex()])
            else:
                encoded.append([type(value).__name__, repr(value)])
        payload = json.dumps(encoded, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
