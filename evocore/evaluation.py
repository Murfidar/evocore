"""vNext candidate, evaluation, and telemetry primitives."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from evocore.exceptions import ConfigurationError, FitnessError
from evocore.exporting import stable_json_dumps
from evocore.individual import GeneValue

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
class EvaluationContext:
    """Describe the evaluator call context for one ask/tell batch."""

    rung: Rung | None
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
class CandidateScore:
    """Store one score observation for one candidate and rung."""

    score: float | None
    confidence: EvaluationConfidence
    rung: str
    cost: float
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRecord:
    """Record one evaluator result returned to an ask/tell engine."""

    candidate_id: str
    score: float | None
    confidence: EvaluationConfidence
    rung: str
    cost: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    batch_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise FitnessError("EvaluationRecord candidate_id must be non-empty.")
        if not self.rung:
            raise FitnessError("EvaluationRecord rung must be non-empty.")
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
                raise FitnessError("EvaluationRecord with confidence='rejected' requires score=None.")
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
        if record.batch_id is not None and self.batch_id and record.batch_id != self.batch_id:
            raise FitnessError(
                f"EvaluationRecord batch_id {record.batch_id!r} does not match "
                f"candidate batch {self.batch_id!r}."
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

    def candidate_hash(self) -> str:
        """Return a stable hash for this candidate's decoded genes."""
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


@dataclass
class OptimizationTelemetry:
    """Aggregate vNext optimizer budget and trial accounting."""

    total_candidates_proposed: int = 0
    unique_candidate_hashes: set[str] = field(default_factory=set)
    candidates_screened: int = 0
    candidates_partial_evaluated: int = 0
    candidates_full_evaluated: int = 0
    candidates_cached: int = 0
    promoted_by_rung: dict[str, int] = field(default_factory=dict)
    eliminated_by_rung: dict[str, int] = field(default_factory=dict)
    cost_by_rung: dict[str, float] = field(default_factory=dict)

    def record_proposed(self, count: int) -> None:
        """Record newly proposed candidate count."""
        self.total_candidates_proposed += int(count)

    def record_proposed_candidates(self, candidates: Sequence[Candidate]) -> None:
        """Record newly proposed candidates and their unique genome hashes."""
        proposed = list(candidates)
        self.record_proposed(len(proposed))
        self.unique_candidate_hashes.update(candidate.candidate_hash() for candidate in proposed)

    def record_screened(self, count: int) -> None:
        """Record candidates scored by a surrogate or screen."""
        self.candidates_screened += int(count)

    def record_partial(self, count: int, *, rung: str, cost: float) -> None:
        """Record partial-fidelity evaluations and their cost."""
        self.candidates_partial_evaluated += int(count)
        self.cost_by_rung[rung] = self.cost_by_rung.get(rung, 0.0) + float(cost)

    def record_full(self, count: int, *, rung: str, cost: float) -> None:
        """Record full trusted evaluations and their cost."""
        self.candidates_full_evaluated += int(count)
        self.cost_by_rung[rung] = self.cost_by_rung.get(rung, 0.0) + float(cost)

    def record_cached(self, count: int, *, rung: str, cost: float) -> None:
        """Record cached trusted observations without spending fresh full-evaluation budget."""
        self.candidates_cached += int(count)
        self.cost_by_rung[rung] = self.cost_by_rung.get(rung, 0.0) + float(cost)

    def record_promoted(self, count: int, *, rung: str) -> None:
        """Record candidates promoted from a rung."""
        self.promoted_by_rung[rung] = self.promoted_by_rung.get(rung, 0) + int(count)

    def record_eliminated(self, count: int, *, rung: str) -> None:
        """Record candidates eliminated at a rung."""
        self.eliminated_by_rung[rung] = self.eliminated_by_rung.get(rung, 0) + int(count)

    def to_dict(self) -> dict[str, Any]:
        """Export stable JSON-safe telemetry fields."""
        return {
            "total_candidates_proposed": self.total_candidates_proposed,
            "unique_candidate_hashes": sorted(self.unique_candidate_hashes),
            "unique_candidate_count": len(self.unique_candidate_hashes),
            "candidates_screened": self.candidates_screened,
            "candidates_partial_evaluated": self.candidates_partial_evaluated,
            "candidates_full_evaluated": self.candidates_full_evaluated,
            "candidates_cached": self.candidates_cached,
            "promoted_by_rung": {
                key: self.promoted_by_rung[key] for key in sorted(self.promoted_by_rung)
            },
            "eliminated_by_rung": {
                key: self.eliminated_by_rung[key] for key in sorted(self.eliminated_by_rung)
            },
            "cost_by_rung": {key: self.cost_by_rung[key] for key in sorted(self.cost_by_rung)},
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Export telemetry as deterministic JSON."""
        return stable_json_dumps(self.to_dict(), indent=indent)


@dataclass(frozen=True)
class TellResult:
    """Summarize one optimizer tell() update."""

    accepted_count: int
    trusted_count: int
    partial_count: int
    surrogate_count: int
    cached_count: int
    rejected_count: int
    best_candidate_id: str | None = None
    best_score: float | None = None
    consumed_batch_ids: tuple[str, ...] = ()
    pending_batch_ids: tuple[str, ...] = ()
    telemetry: OptimizationTelemetry | None = None


@dataclass(frozen=True)
class EngineStateSummary:
    """Expose a stable read-only optimizer state summary."""

    best_candidate_id: str | None
    best_score: float | None
    event_index: int
    pending_batch_ids: tuple[str, ...]
    trusted_count: int
    telemetry: OptimizationTelemetry
