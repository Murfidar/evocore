"""Completed-run result and reporting types."""

from evocore.lifecycle.events import EventHistory, EventRecord, StopReason, append_run_stop_event
from evocore.results.checkpointing import (
    CHECKPOINT_KIND,
    CHECKPOINT_SCHEMA_VERSION,
    SEED_DERIVATION_ALGORITHM,
    SEED_DERIVATION_VERSION,
    CheckpointSnapshot,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_envelope,
    validate_checkpoint_identity,
)
from evocore.results.generation import GenerationHistory, GenerationRecord
from evocore.results.reproducibility import (
    ReproducibilityMetadata,
    gene_space_hash,
    gene_space_signature,
)
from evocore.results.run import OptimizationBatchResult, OptimizationResult

__all__ = [
    "CHECKPOINT_KIND",
    "CHECKPOINT_SCHEMA_VERSION",
    "SEED_DERIVATION_ALGORITHM",
    "SEED_DERIVATION_VERSION",
    "CheckpointSnapshot",
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
    "load_checkpoint",
    "save_checkpoint",
    "validate_checkpoint_envelope",
    "validate_checkpoint_identity",
]
