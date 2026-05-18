"""Optimizer telemetry and update summaries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from evocore.core.serialization import stable_json_dumps
from evocore.lifecycle.records import Candidate


@dataclass
class OptimizationTelemetry:
    """Aggregate vNext optimizer budget and trial accounting."""

    total_candidates_proposed: int = 0
    unique_candidate_hashes: set[str] = field(default_factory=set)
    candidates_screened: int = 0
    candidates_partial_evaluated: int = 0
    candidates_full_evaluated: int = 0
    candidates_cached: int = 0
    promoted_by_stage: dict[str, int] = field(default_factory=dict)
    eliminated_by_stage: dict[str, int] = field(default_factory=dict)
    cost_by_stage: dict[str, float] = field(default_factory=dict)

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

    def record_partial(self, count: int, *, stage: str, cost: float) -> None:
        """Record partial-fidelity evaluations and their cost."""
        self.candidates_partial_evaluated += int(count)
        self.cost_by_stage[stage] = self.cost_by_stage.get(stage, 0.0) + float(cost)

    def record_full(self, count: int, *, stage: str, cost: float) -> None:
        """Record full trusted evaluations and their cost."""
        self.candidates_full_evaluated += int(count)
        self.cost_by_stage[stage] = self.cost_by_stage.get(stage, 0.0) + float(cost)

    def record_cached(self, count: int, *, stage: str, cost: float) -> None:
        """Record cached trusted observations without spending fresh full-evaluation budget."""
        self.candidates_cached += int(count)
        self.cost_by_stage[stage] = self.cost_by_stage.get(stage, 0.0) + float(cost)

    def record_promoted(self, count: int, *, stage: str) -> None:
        """Record candidates promoted from a stage."""
        self.promoted_by_stage[stage] = self.promoted_by_stage.get(stage, 0) + int(count)

    def record_eliminated(self, count: int, *, stage: str) -> None:
        """Record candidates eliminated at a stage."""
        self.eliminated_by_stage[stage] = self.eliminated_by_stage.get(stage, 0) + int(count)

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
            "promoted_by_stage": {
                key: self.promoted_by_stage[key] for key in sorted(self.promoted_by_stage)
            },
            "eliminated_by_stage": {
                key: self.eliminated_by_stage[key] for key in sorted(self.eliminated_by_stage)
            },
            "cost_by_stage": {key: self.cost_by_stage[key] for key in sorted(self.cost_by_stage)},
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Export telemetry as deterministic JSON."""
        return stable_json_dumps(self.to_dict(), indent=indent)


@dataclass(frozen=True)
class UpdateResult:
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
class OptimizerStateSummary:
    """Expose a stable read-only optimizer state summary."""

    best_candidate_id: str | None
    best_score: float | None
    event_index: int
    pending_batch_ids: tuple[str, ...]
    trusted_count: int
    telemetry: OptimizationTelemetry


__all__ = ["OptimizationTelemetry", "OptimizerStateSummary", "UpdateResult"]
