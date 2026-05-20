"""Stable checkpoint helpers for lifecycle runtime state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from evocore.core.errors import CheckpointError, FitnessError
from evocore.core.serialization import json_safe
from evocore.lifecycle.batches import CandidateBatch
from evocore.lifecycle.events import EventHistory, EventRecord
from evocore.lifecycle.records import Candidate, EvaluationRecord, ScoreObservation
from evocore.lifecycle.telemetry import OptimizationTelemetry


def _require_mapping(payload: object, label: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise CheckpointError(f"checkpoint {label} must be an object.")
    return payload


def _require_list(payload: Mapping[str, Any], key: str, label: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise CheckpointError(f"checkpoint {label}.{key} must be a list.")
    return value


def score_observation_to_checkpoint(observation: ScoreObservation) -> dict[str, Any]:
    """Serialise a ScoreObservation to a JSON-safe checkpoint dict."""
    return json_safe(
        {
            "score": observation.score,
            "confidence": observation.confidence,
            "stage": observation.stage,
            "cost": observation.cost,
            "metrics": dict(observation.metrics),
            "metadata": dict(observation.metadata),
        }
    )


def score_observation_from_checkpoint(payload: object) -> ScoreObservation:
    """Deserialise a ScoreObservation from a checkpoint payload."""
    data = _require_mapping(payload, "score observation")
    return ScoreObservation(
        score=data.get("score"),
        confidence=data.get("confidence"),
        stage=str(data.get("stage") or ""),
        cost=float(data.get("cost", 0.0)),
        metrics=dict(data.get("metrics") or {}),
        metadata=dict(data.get("metadata") or {}),
    )


def candidate_to_checkpoint(candidate: Candidate) -> dict[str, Any]:
    """Serialise a Candidate to a JSON-safe checkpoint dict."""
    return json_safe(
        {
            "candidate_id": candidate.candidate_id,
            "genes": list(candidate.genes),
            "batch_id": candidate.batch_id,
            "params": dict(candidate.params) if candidate.params is not None else None,
            "origin": candidate.origin,
            "parents": list(candidate.parents),
            "event_index": candidate.event_index,
            "generation": candidate.generation,
            "stage": candidate.stage,
            "status": candidate.status,
            "confidence": candidate.confidence,
            "cost": candidate.cost,
            "scores": {
                stage: score_observation_to_checkpoint(observation)
                for stage, observation in sorted(candidate.scores.items())
            },
            "metadata": dict(candidate.metadata),
        }
    )


def candidate_from_checkpoint(payload: object) -> Candidate:
    """Deserialise a Candidate from a checkpoint payload."""
    data = _require_mapping(payload, "candidate")
    genes = _require_list(data, "genes", "candidate")
    scores_payload = data.get("scores") or {}
    if not isinstance(scores_payload, Mapping):
        raise CheckpointError("checkpoint candidate.scores must be an object.")
    candidate = Candidate(
        candidate_id=str(data.get("candidate_id") or ""),
        genes=list(genes),
        batch_id=str(data.get("batch_id") or ""),
        params=dict(data["params"]) if isinstance(data.get("params"), Mapping) else None,
        origin=data.get("origin", "random"),
        parents=tuple(data.get("parents") or ()),
        event_index=int(data.get("event_index", 0)),
        generation=data.get("generation"),
        stage=data.get("stage"),
        status=data.get("status", "proposed"),
        confidence=data.get("confidence"),
        cost=float(data.get("cost", 0.0)),
        metadata=dict(data.get("metadata") or {}),
    )
    candidate.scores = {
        str(stage): score_observation_from_checkpoint(observation)
        for stage, observation in scores_payload.items()
    }
    return candidate


def evaluation_record_to_checkpoint(record: EvaluationRecord) -> dict[str, Any]:
    """Serialise an EvaluationRecord to a JSON-safe checkpoint dict."""
    return json_safe(
        {
            "candidate_id": record.candidate_id,
            "batch_id": record.batch_id,
            "score": record.score,
            "confidence": record.confidence,
            "stage": record.stage,
            "cost": record.cost,
            "metrics": dict(record.metrics),
            "metadata": dict(record.metadata),
        }
    )


def evaluation_record_from_checkpoint(payload: object) -> EvaluationRecord:
    """Deserialise an EvaluationRecord from a checkpoint payload."""
    data = _require_mapping(payload, "evaluation record")
    return EvaluationRecord(
        candidate_id=str(data.get("candidate_id") or ""),
        batch_id=data.get("batch_id"),
        score=data.get("score"),
        confidence=data.get("confidence"),
        stage=str(data.get("stage") or ""),
        cost=float(data.get("cost", 0.0)),
        metrics=dict(data.get("metrics") or {}),
        metadata=dict(data.get("metadata") or {}),
    )


def batch_to_checkpoint(batch: CandidateBatch) -> dict[str, Any]:
    """Serialise a CandidateBatch to a JSON-safe checkpoint dict."""
    records = [
        evaluation_record_to_checkpoint(record)
        for _, record in sorted(batch.records_by_key.items())
    ]
    return json_safe(
        {
            "batch_id": batch.batch_id,
            "candidate_ids": list(batch.candidate_ids),
            "records": records,
            "consumed": bool(batch.consumed),
            "continuous_samples_by_id": {
                candidate_id: list(sample)
                for candidate_id, sample in sorted(batch.continuous_samples_by_id.items())
            },
        }
    )


def batch_from_checkpoint(payload: object) -> CandidateBatch:
    """Deserialise a CandidateBatch from a checkpoint payload."""
    data = _require_mapping(payload, "batch")
    candidate_ids = tuple(
        str(candidate_id) for candidate_id in _require_list(data, "candidate_ids", "batch")
    )
    samples_payload = data.get("continuous_samples_by_id") or {}
    if not isinstance(samples_payload, Mapping):
        raise CheckpointError("checkpoint batch.continuous_samples_by_id must be an object.")
    batch = CandidateBatch(
        batch_id=str(data.get("batch_id") or ""),
        candidate_ids=candidate_ids,
        continuous_samples_by_id={
            str(candidate_id): [float(value) for value in values]
            for candidate_id, values in samples_payload.items()
        },
    )
    for record_payload in _require_list(data, "records", "batch"):
        try:
            batch.accept_record(evaluation_record_from_checkpoint(record_payload))
        except FitnessError as exc:
            raise CheckpointError(str(exc)) from exc
    batch.consumed = bool(data.get("consumed", False))
    return batch


def telemetry_to_checkpoint(telemetry: OptimizationTelemetry) -> dict[str, Any]:
    """Serialise OptimizationTelemetry to a JSON-safe checkpoint dict."""
    return telemetry.to_dict()


def telemetry_from_checkpoint(payload: object) -> OptimizationTelemetry:
    """Deserialise OptimizationTelemetry from a checkpoint payload."""
    data = _require_mapping(payload, "telemetry")
    return OptimizationTelemetry(
        total_candidates_proposed=int(data.get("total_candidates_proposed", 0)),
        unique_candidate_hashes=set(data.get("unique_candidate_hashes") or ()),
        candidates_screened=int(data.get("candidates_screened", 0)),
        candidates_partial_evaluated=int(data.get("candidates_partial_evaluated", 0)),
        candidates_full_evaluated=int(data.get("candidates_full_evaluated", 0)),
        candidates_cached=int(data.get("candidates_cached", 0)),
        promoted_by_stage={
            str(stage): int(count)
            for stage, count in dict(data.get("promoted_by_stage") or {}).items()
        },
        eliminated_by_stage={
            str(stage): int(count)
            for stage, count in dict(data.get("eliminated_by_stage") or {}).items()
        },
        cost_by_stage={
            str(stage): float(cost)
            for stage, cost in dict(data.get("cost_by_stage") or {}).items()
        },
    )


def event_record_to_checkpoint(event: EventRecord) -> dict[str, Any]:
    """Serialise an EventRecord to a JSON-safe checkpoint dict."""
    return event.to_dict()


def event_record_from_checkpoint(payload: object) -> EventRecord:
    """Deserialise an EventRecord from a checkpoint payload."""
    data = _require_mapping(payload, "event")
    return EventRecord(
        event_index=int(data.get("event_index", 0)),
        event_type=data.get("event_type"),
        batch_id=data.get("batch_id"),
        candidate_id=data.get("candidate_id"),
        candidate_hash=data.get("candidate_hash"),
        generation=data.get("generation"),
        stage=data.get("stage"),
        confidence=data.get("confidence"),
        raw_score=data.get("raw_score"),
        comparison_score=data.get("comparison_score"),
        cost=float(data.get("cost", 0.0)),
        status=data.get("status"),
        origin=data.get("origin"),
        parents=tuple(data.get("parents") or ()),
        genes=tuple(data.get("genes") or ()),
        params=dict(data["params"]) if isinstance(data.get("params"), Mapping) else None,
        metrics=dict(data.get("metrics") or {}),
        metadata=dict(data.get("metadata") or {}),
    )


def event_history_to_checkpoint(history: EventHistory) -> list[dict[str, Any]]:
    """Serialise an EventHistory to a list of JSON-safe checkpoint dicts."""
    return [event_record_to_checkpoint(event) for event in history]


def event_history_from_checkpoint(payload: object) -> EventHistory:
    """Deserialise an EventHistory from a checkpoint payload."""
    if not isinstance(payload, list):
        raise CheckpointError("checkpoint events must be a list.")
    history = EventHistory()
    for row in payload:
        history.append(event_record_from_checkpoint(row))
    return history
