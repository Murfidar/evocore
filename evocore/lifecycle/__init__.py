"""Ask/tell lifecycle contracts shared by EvoCore optimizers."""

from evocore.lifecycle.batches import CandidateBatch, batch_id_from_seed
from evocore.lifecycle.checkpointing import (
    batch_from_checkpoint,
    batch_to_checkpoint,
    candidate_from_checkpoint,
    candidate_to_checkpoint,
    event_history_from_checkpoint,
    event_history_to_checkpoint,
    telemetry_from_checkpoint,
    telemetry_to_checkpoint,
)
from evocore.lifecycle.conversion import candidate_to_solution, solution_to_candidate
from evocore.lifecycle.events import EventHistory, EventRecord, StopReason, append_run_stop_event
from evocore.lifecycle.policies import BudgetPolicy
from evocore.lifecycle.protocols import Evaluator, Optimizer
from evocore.lifecycle.records import (
    Candidate,
    CandidateOrigin,
    CandidateStatus,
    Direction,
    EvaluationConfidence,
    EvaluationContext,
    EvaluationRecord,
    EvaluationStage,
    ScoreObservation,
    is_state_update_confidence,
    score_for_direction,
)
from evocore.lifecycle.scheduler import BudgetScheduler
from evocore.lifecycle.telemetry import OptimizationTelemetry, OptimizerStateSummary, UpdateResult

__all__ = [
    "BudgetPolicy",
    "BudgetScheduler",
    "Candidate",
    "CandidateBatch",
    "CandidateOrigin",
    "CandidateStatus",
    "Direction",
    "EvaluationConfidence",
    "EvaluationContext",
    "EvaluationRecord",
    "EvaluationStage",
    "Evaluator",
    "EventHistory",
    "EventRecord",
    "OptimizationTelemetry",
    "Optimizer",
    "OptimizerStateSummary",
    "ScoreObservation",
    "StopReason",
    "UpdateResult",
    "append_run_stop_event",
    "batch_from_checkpoint",
    "batch_id_from_seed",
    "batch_to_checkpoint",
    "candidate_from_checkpoint",
    "candidate_to_checkpoint",
    "candidate_to_solution",
    "event_history_from_checkpoint",
    "event_history_to_checkpoint",
    "is_state_update_confidence",
    "score_for_direction",
    "solution_to_candidate",
    "telemetry_from_checkpoint",
    "telemetry_to_checkpoint",
]
