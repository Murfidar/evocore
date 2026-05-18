"""Append-only optimizer lifecycle events."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from evocore.core.errors import ConfigurationError
from evocore.core.serialization import json_safe
from evocore.lifecycle.records import CandidateOrigin, CandidateStatus, EvaluationConfidence
from evocore.search_space import GeneValue

StopReason = Literal[
    "max_evaluations",
    "max_generations",
    "callback",
    "manual",
    "optimizer_converged",
    "target_score",
    "patience",
    "wall_time",
]


@dataclass(frozen=True)
class EventRecord:
    """Represent one append-only optimizer lifecycle observation."""

    event_index: int
    event_type: Literal["ask", "tell", "generation", "run_stop"]
    batch_id: str | None = None
    candidate_id: str | None = None
    candidate_hash: str | None = None
    generation: int | None = None
    stage: str | None = None
    confidence: EvaluationConfidence | None = None
    raw_score: float | None = None
    comparison_score: float | None = None
    cost: float = 0.0
    status: CandidateStatus | None = None
    origin: CandidateOrigin | None = None
    parents: tuple[str, ...] = ()
    genes: tuple[GeneValue, ...] = ()
    params: dict[str, GeneValue] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Export this event as one JSON-safe row."""
        return json_safe(
            {
                "event_index": self.event_index,
                "event_type": self.event_type,
                "batch_id": self.batch_id,
                "candidate_id": self.candidate_id,
                "candidate_hash": self.candidate_hash,
                "generation": self.generation,
                "stage": self.stage,
                "confidence": self.confidence,
                "raw_score": self.raw_score,
                "comparison_score": self.comparison_score,
                "cost": self.cost,
                "status": self.status,
                "origin": self.origin,
                "parents": list(self.parents),
                "genes": list(self.genes),
                "params": self.params,
                "metrics": self.metrics,
                "metadata": self.metadata,
            }
        )


class EventHistory:
    """Store append-only optimizer lifecycle events."""

    def __init__(self) -> None:
        self._events: list[EventRecord] = []

    def append(self, event: EventRecord) -> None:
        """Append one event in strict sequence order."""
        if event.event_index != len(self._events):
            raise ConfigurationError(
                "EventHistory is append-only; event_index must match the next row index."
            )
        self._events.append(event)

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[EventRecord]:
        return iter(self._events)

    def __getitem__(self, index: int) -> EventRecord:
        return self._events[index]

    def to_rows(self) -> list[dict[str, Any]]:
        """Export lifecycle events as JSON-safe row dictionaries."""
        return [event.to_dict() for event in self._events]

    def to_dict(self) -> list[dict[str, Any]]:
        """Export lifecycle events as JSON-safe row dictionaries."""
        return self.to_rows()

    def to_dataframe(self):
        """Convert event rows into a pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "EventHistory.to_dataframe() requires pandas; pip install pandas."
            ) from exc
        return pd.DataFrame(self.to_rows())


def append_run_stop_event(
    history: EventHistory,
    *,
    stop_reason: StopReason,
    max_evaluations: int | None,
    max_generations: int | None,
    n_evaluations: int,
) -> None:
    """Append one terminal run-level stop event."""
    history.append(
        EventRecord(
            event_index=len(history),
            event_type="run_stop",
            metadata={
                "stop_reason": stop_reason,
                "max_evaluations": max_evaluations,
                "max_generations": max_generations,
                "n_evaluations": n_evaluations,
            },
        )
    )


__all__ = ["EventHistory", "EventRecord", "StopReason", "append_run_stop_event"]
