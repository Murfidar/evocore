"""Structural protocols for EvoCore optimizer lifecycle APIs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from evocore.lifecycle.external import (
    CandidateSnapshot,
    CmaMeanStrategy,
    ExternalStateCapabilities,
    InjectionMode,
    InjectionResult,
    PopulationSnapshot,
    SnapshotScope,
    WarmStartMode,
    WarmStartRecord,
)
from evocore.lifecycle.records import (
    TRUSTED_CONFIDENCES,
    Candidate,
    CandidateOrigin,
    Direction,
    EvaluationConfidence,
    EvaluationContext,
    EvaluationRecord,
)
from evocore.lifecycle.telemetry import OptimizerStateSummary, UpdateResult


@runtime_checkable
class Optimizer(Protocol):
    """Structural protocol implemented by ask/tell optimizers."""

    direction: Direction

    def ask(self, n: int | None = None) -> Sequence[Candidate]:
        """Return candidates for external evaluation."""
        ...

    def tell(self, records: Sequence[EvaluationRecord]) -> UpdateResult:
        """Apply evaluation records and return a summary of accepted records."""
        ...

    def state_summary(self) -> OptimizerStateSummary:
        """Return a read-only optimizer state summary."""
        ...


@runtime_checkable
class ExternalStateOptimizer(Optimizer, Protocol):
    """Structural protocol for optimizers with external-state integration APIs."""

    def external_state_capabilities(self) -> ExternalStateCapabilities:
        """Return optimizer-specific external-state support flags."""
        raise NotImplementedError

    def warm_start(
        self,
        records: Sequence[WarmStartRecord],
        *,
        deduplicate: bool = True,
        mode: WarmStartMode = "state",
        cma_mean_strategy: CmaMeanStrategy = "best",
        top_k: int | None = None,
    ) -> UpdateResult:
        """Initialize or track optimizer state from prior scored candidates."""
        raise NotImplementedError

    def inject_candidates(
        self,
        records: Sequence[WarmStartRecord],
        *,
        origin: CandidateOrigin = "memory_seed",
        mode: InjectionMode = "proposed",
        deduplicate: bool = True,
        metadata: Mapping[str, object] | None = None,
    ) -> InjectionResult:
        """Inject external candidates into the optimizer lifecycle."""
        raise NotImplementedError

    def candidate_snapshot(
        self,
        *,
        scope: SnapshotScope = "trusted",
    ) -> PopulationSnapshot:
        """Return a read-only candidate population snapshot."""
        raise NotImplementedError

    def top_candidates(
        self,
        k: int = 10,
        *,
        scope: SnapshotScope = "trusted",
        confidence: tuple[EvaluationConfidence, ...] = TRUSTED_CONFIDENCES,
    ) -> tuple[CandidateSnapshot, ...]:
        """Return top-k read-only candidate snapshots."""
        raise NotImplementedError


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
