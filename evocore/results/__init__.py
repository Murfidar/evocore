"""Completed-run result and reporting types."""

from evocore.lifecycle.events import EventHistory, EventRecord, StopReason, append_run_stop_event
from evocore.results.generation import GenerationHistory, GenerationRecord
from evocore.results.reproducibility import (
    ReproducibilityMetadata,
    gene_space_hash,
    gene_space_signature,
)
from evocore.results.run import OptimizationBatchResult, OptimizationResult

__all__ = [
    "EventHistory",
    "EventRecord",
    "GenerationHistory",
    "GenerationRecord",
    "OptimizationBatchResult",
    "OptimizationResult",
    "ReproducibilityMetadata",
    "StopReason",
    "append_run_stop_event",
    "gene_space_hash",
    "gene_space_signature",
]
