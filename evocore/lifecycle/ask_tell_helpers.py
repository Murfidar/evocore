"""Shared ask/tell lifecycle helpers for optimizer implementations."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Literal

from evocore.core.errors import FitnessError
from evocore.lifecycle.batches import CandidateBatch
from evocore.lifecycle.events import EventHistory, EventRecord
from evocore.lifecycle.records import (
    Candidate,
    Direction,
    EvaluationContext,
    EvaluationRecord,
    EvaluationStage,
    score_for_direction,
)
from evocore.lifecycle.telemetry import OptimizationTelemetry
from evocore.search_space import GeneSpace

TelemetryLabel = Literal["trusted", "partial", "surrogate", "cached", "rejected"]


def candidate_and_batch_for_record(
    record: EvaluationRecord,
    candidates_by_id: Mapping[str, Candidate],
    batches_by_id: Mapping[str, CandidateBatch],
) -> tuple[Candidate, CandidateBatch]:
    """Return the candidate and batch referenced by a tell record."""
    candidate = candidates_by_id.get(record.candidate_id)
    if candidate is None:
        raise FitnessError(f"tell() received unknown candidate_id: {record.candidate_id!r}")
    if record.batch_id is not None and record.batch_id not in batches_by_id:
        raise FitnessError(f"tell() received unknown batch_id: {record.batch_id!r}")
    batch = batches_by_id.get(candidate.batch_id)
    if batch is None:
        raise FitnessError(f"tell() received unknown batch_id: {candidate.batch_id!r}")
    return candidate, batch


def append_candidate_ask_events(
    events: EventHistory,
    candidates: Sequence[Candidate],
    gene_space: GeneSpace,
) -> None:
    """Append one ask event for each proposed candidate."""
    for candidate in candidates:
        events.append(
            EventRecord(
                event_index=len(events),
                event_type="ask",
                batch_id=candidate.batch_id,
                candidate_id=candidate.candidate_id,
                candidate_hash=candidate.candidate_hash(gene_space),
                generation=candidate.generation,
                origin=candidate.origin,
                parents=tuple(candidate.parents),
                genes=tuple(candidate.genes),
                params=dict(candidate.params) if candidate.params is not None else None,
                metadata=dict(candidate.metadata),
            )
        )


def append_candidate_tell_event(
    events: EventHistory,
    candidate: Candidate,
    record: EvaluationRecord,
    gene_space: GeneSpace,
    direction: Direction,
    *,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Append a tell event for an applied evaluation record."""
    raw_score = float(record.score) if record.score is not None else None
    comparison_score = (
        score_for_direction(raw_score, direction)
        if raw_score is not None and math.isfinite(raw_score)
        else None
    )
    event_metadata = dict(record.metadata)
    event_metadata.update(dict(metadata or {}))
    events.append(
        EventRecord(
            event_index=len(events),
            event_type="tell",
            batch_id=candidate.batch_id,
            candidate_id=candidate.candidate_id,
            candidate_hash=candidate.candidate_hash(gene_space),
            generation=candidate.generation,
            stage=record.stage,
            confidence=record.confidence,
            raw_score=raw_score,
            comparison_score=comparison_score,
            cost=record.cost,
            status=candidate.status,
            origin=candidate.origin,
            parents=tuple(candidate.parents),
            genes=tuple(candidate.genes),
            params=dict(candidate.params) if candidate.params is not None else None,
            metrics=dict(record.metrics),
            metadata=event_metadata,
        )
    )


def record_evaluation_telemetry(
    telemetry: OptimizationTelemetry,
    record: EvaluationRecord,
) -> TelemetryLabel:
    """Record telemetry for one evaluation record and return its count label."""
    if record.confidence == "trusted_full":
        telemetry.record_full(1, stage=record.stage, cost=record.cost)
        return "trusted"
    if record.confidence == "cached":
        telemetry.record_cached(1, stage=record.stage, cost=record.cost)
        return "cached"
    if record.confidence == "partial":
        telemetry.record_partial(1, stage=record.stage, cost=record.cost)
        return "partial"
    if record.confidence == "surrogate":
        telemetry.record_screened(1)
        return "surrogate"
    telemetry.record_eliminated(1, stage=record.stage)
    return "rejected"


def evaluation_context_for_candidates(
    assigned: Sequence[Candidate],
    stage: EvaluationStage,
    *,
    direction: Direction,
    fallback_event_index: int,
    batch_error_message: str,
) -> EvaluationContext:
    """Build an evaluation context for one synchronous candidate batch."""
    batch_ids = {candidate.batch_id for candidate in assigned}
    if len(batch_ids) != 1:
        raise FitnessError(batch_error_message)
    return EvaluationContext(
        stage=stage,
        batch_id=next(iter(batch_ids)),
        event_index=assigned[0].event_index if assigned else fallback_event_index,
        direction=direction,
        budget=stage.budget,
    )


def validate_evaluator_records(
    assigned: Sequence[Candidate],
    records: Sequence[EvaluationRecord],
    *,
    batch_error_message: str,
) -> None:
    """Reject incomplete or mismatched synchronous evaluator records."""
    expected_ids = [candidate.candidate_id for candidate in assigned]
    returned_ids = [record.candidate_id for record in records]
    expected_counts = Counter(expected_ids)
    returned_counts = Counter(returned_ids)

    missing_ids = [
        candidate_id for candidate_id in expected_ids if returned_counts[candidate_id] == 0
    ]
    unexpected_ids = [
        candidate_id for candidate_id in returned_counts if candidate_id not in expected_counts
    ]
    duplicate_ids = [
        candidate_id
        for candidate_id, count in returned_counts.items()
        if count > expected_counts[candidate_id]
    ]

    if missing_ids:
        raise FitnessError(
            "Evaluator returned missing evaluation records for candidate_ids: "
            f"{sorted(set(missing_ids))!r}."
        )
    if unexpected_ids:
        raise FitnessError(
            "Evaluator returned unknown evaluation records for candidate_ids: "
            f"{sorted(unexpected_ids)!r}."
        )
    if duplicate_ids:
        raise FitnessError(
            "Evaluator returned duplicate evaluation records for candidate_ids: "
            f"{sorted(duplicate_ids)!r}."
        )

    batch_ids = {candidate.batch_id for candidate in assigned}
    if len(batch_ids) != 1:
        raise FitnessError(batch_error_message)
    expected_batch_id = next(iter(batch_ids))
    for record in records:
        if record.batch_id is not None and record.batch_id != expected_batch_id:
            raise FitnessError(
                f"Evaluator returned record batch_id {record.batch_id!r} for batch "
                f"{expected_batch_id!r}."
            )


__all__ = [
    "TelemetryLabel",
    "append_candidate_ask_events",
    "append_candidate_tell_event",
    "candidate_and_batch_for_record",
    "evaluation_context_for_candidates",
    "record_evaluation_telemetry",
    "validate_evaluator_records",
]
