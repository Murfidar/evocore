"""Structural protocols for EvoCore optimizer lifecycle APIs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from evocore.evaluation import (
    Candidate,
    Direction,
    EngineStateSummary,
    EvaluationContext,
    EvaluationRecord,
    TellResult,
)


@runtime_checkable
class Optimizer(Protocol):
    """Structural protocol implemented by ask/tell optimizers."""

    direction: Direction

    def ask(self, n: int | None = None) -> Sequence[Candidate]:
        """Return candidates for external evaluation."""
        ...

    def tell(self, records: Sequence[EvaluationRecord]) -> TellResult:
        """Apply evaluation records and return a summary of accepted records."""
        ...

    def state_summary(self) -> EngineStateSummary:
        """Return a read-only optimizer state summary."""
        ...


@runtime_checkable
class Evaluator(Protocol):
    """Structural protocol implemented by objective evaluators."""

    def evaluate(
        self,
        candidates: Sequence[Candidate],
        context: EvaluationContext,
    ) -> Sequence[EvaluationRecord]:
        """Evaluate candidates in the supplied context."""
        ...
