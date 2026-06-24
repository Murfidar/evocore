"""Stable checkpoint helpers for lifecycle runtime state."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from evocore.core.errors import CheckpointError, ConfigurationError, FitnessError
from evocore.core.serialization import json_safe
from evocore.lifecycle.batches import CandidateBatch
from evocore.lifecycle.events import EventHistory, EventRecord
from evocore.lifecycle.records import Candidate, EvaluationRecord, ScoreObservation
from evocore.lifecycle.telemetry import OptimizationTelemetry

_CANDIDATE_ORIGINS = {
    "random",
    "crossover",
    "mutation",
    "cma_sample",
    "surrogate_proposal",
    "memory_seed",
    "restart",
}
_CANDIDATE_STATUSES = {
    "proposed",
    "screened",
    "racing",
    "promoted",
    "trusted",
    "eliminated",
    "archived",
}
_EVALUATION_CONFIDENCES = {
    "surrogate",
    "partial",
    "cached",
    "trusted_full",
    "constraint_penalty",
    "rejected",
}
_EVENT_TYPES = {"ask", "tell", "generation", "run_stop"}


def _require_mapping(payload: object, label: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise CheckpointError(f"checkpoint {label} must be an object.")
    return payload


def _require_list(payload: Mapping[str, Any], key: str, label: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise CheckpointError(f"checkpoint {label}.{key} must be a list.")
    return value


def _require_non_empty_str(payload: Mapping[str, Any], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise CheckpointError(f"checkpoint {label}.{key} must be a non-empty string.")
    return value


def _optional_str(payload: Mapping[str, Any], key: str, label: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise CheckpointError(f"checkpoint {label}.{key} must be a string when provided.")
    return value


def _require_mapping_value(payload: Mapping[str, Any], key: str, label: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise CheckpointError(f"checkpoint {label}.{key} must be an object.")
    return value


def _optional_mapping_value(
    payload: Mapping[str, Any],
    key: str,
    label: str,
) -> Mapping[str, Any] | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise CheckpointError(f"checkpoint {label}.{key} must be an object when provided.")
    return value


def _finite_float(payload: Mapping[str, Any], key: str, label: str) -> float:
    try:
        value = float(payload.get(key, 0.0))
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"checkpoint {label}.{key} must be a finite number.") from exc
    if not math.isfinite(value):
        raise CheckpointError(f"checkpoint {label}.{key} must be a finite number.")
    return value


def _optional_finite_float(
    payload: Mapping[str, Any],
    key: str,
    label: str,
) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"checkpoint {label}.{key} must be a finite number.") from exc
    if not math.isfinite(number):
        raise CheckpointError(f"checkpoint {label}.{key} must be a finite number.")
    return number


def _non_negative_int(payload: Mapping[str, Any], key: str, label: str) -> int:
    try:
        value = int(payload.get(key, 0))
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"checkpoint {label}.{key} must be a non-negative integer.") from exc
    if value < 0:
        raise CheckpointError(f"checkpoint {label}.{key} must be a non-negative integer.")
    return value


def _literal_value(
    value: object,
    *,
    key: str,
    label: str,
    allowed: set[str],
    optional: bool = False,
) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise CheckpointError(f"checkpoint {label}.{key} must be one of: {choices}.")
    return value


def _counter_mapping(payload: Mapping[str, Any], key: str, label: str) -> dict[str, int]:
    value = payload.get(key) or {}
    if not isinstance(value, Mapping):
        raise CheckpointError(f"checkpoint {label}.{key} must be an object.")
    result: dict[str, int] = {}
    for stage, count in value.items():
        try:
            parsed = int(count)
        except (TypeError, ValueError) as exc:
            raise CheckpointError(
                f"checkpoint {label}.{key}.{stage} must be a non-negative integer."
            ) from exc
        if parsed < 0:
            raise CheckpointError(
                f"checkpoint {label}.{key}.{stage} must be a non-negative integer."
            )
        result[str(stage)] = parsed
    return result


def _cost_mapping(payload: Mapping[str, Any], key: str, label: str) -> dict[str, float]:
    value = payload.get(key) or {}
    if not isinstance(value, Mapping):
        raise CheckpointError(f"checkpoint {label}.{key} must be an object.")
    result: dict[str, float] = {}
    for stage, cost in value.items():
        try:
            parsed = float(cost)
        except (TypeError, ValueError) as exc:
            raise CheckpointError(
                f"checkpoint {label}.{key}.{stage} must be a finite number."
            ) from exc
        if not math.isfinite(parsed):
            raise CheckpointError(f"checkpoint {label}.{key}.{stage} must be a finite number.")
        result[str(stage)] = parsed
    return result


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
        score=_optional_finite_float(data, "score", "score observation"),
        confidence=_literal_value(
            data.get("confidence"),
            key="confidence",
            label="score observation",
            allowed=_EVALUATION_CONFIDENCES,
        ),
        stage=_require_non_empty_str(data, "stage", "score observation"),
        cost=_finite_float(data, "cost", "score observation"),
        metrics=dict(_require_mapping_value(data, "metrics", "score observation")),
        metadata=dict(_require_mapping_value(data, "metadata", "score observation")),
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
    params = _optional_mapping_value(data, "params", "candidate")
    candidate = Candidate(
        candidate_id=_require_non_empty_str(data, "candidate_id", "candidate"),
        genes=list(genes),
        batch_id=_require_non_empty_str(data, "batch_id", "candidate"),
        params=dict(params) if params is not None else None,
        origin=_literal_value(
            data.get("origin"),
            key="origin",
            label="candidate",
            allowed=_CANDIDATE_ORIGINS,
        ),
        parents=tuple(data.get("parents") or ()),
        event_index=_non_negative_int(data, "event_index", "candidate"),
        generation=data.get("generation"),
        stage=_optional_str(data, "stage", "candidate"),
        status=_literal_value(
            data.get("status"),
            key="status",
            label="candidate",
            allowed=_CANDIDATE_STATUSES,
        ),
        confidence=_literal_value(
            data.get("confidence"),
            key="confidence",
            label="candidate",
            allowed=_EVALUATION_CONFIDENCES,
            optional=True,
        ),
        cost=_finite_float(data, "cost", "candidate"),
        metadata=dict(_require_mapping_value(data, "metadata", "candidate")),
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
    try:
        return EvaluationRecord(
            candidate_id=_require_non_empty_str(data, "candidate_id", "evaluation record"),
            batch_id=_optional_str(data, "batch_id", "evaluation record"),
            score=_optional_finite_float(data, "score", "evaluation record"),
            confidence=_literal_value(
                data.get("confidence"),
                key="confidence",
                label="evaluation record",
                allowed=_EVALUATION_CONFIDENCES,
            ),
            stage=_require_non_empty_str(data, "stage", "evaluation record"),
            cost=_finite_float(data, "cost", "evaluation record"),
            metrics=dict(_require_mapping_value(data, "metrics", "evaluation record")),
            metadata=dict(_require_mapping_value(data, "metadata", "evaluation record")),
        )
    except FitnessError as exc:
        raise CheckpointError(str(exc)) from exc


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
    continuous_samples_by_id = {}
    for candidate_id, values in samples_payload.items():
        if not isinstance(values, list):
            raise CheckpointError(
                "checkpoint batch.continuous_samples_by_id values must be lists."
            )
        continuous_samples_by_id[str(candidate_id)] = [float(value) for value in values]
    batch = CandidateBatch(
        batch_id=_require_non_empty_str(data, "batch_id", "batch"),
        candidate_ids=candidate_ids,
        continuous_samples_by_id=continuous_samples_by_id,
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
    unique_hashes = data.get("unique_candidate_hashes") or []
    if not isinstance(unique_hashes, list):
        raise CheckpointError("checkpoint telemetry.unique_candidate_hashes must be a list.")
    return OptimizationTelemetry(
        total_candidates_proposed=_non_negative_int(
            data, "total_candidates_proposed", "telemetry"
        ),
        unique_candidate_hashes=set(unique_hashes),
        candidates_screened=_non_negative_int(data, "candidates_screened", "telemetry"),
        candidates_partial_evaluated=_non_negative_int(
            data, "candidates_partial_evaluated", "telemetry"
        ),
        candidates_full_evaluated=_non_negative_int(
            data, "candidates_full_evaluated", "telemetry"
        ),
        candidates_cached=_non_negative_int(data, "candidates_cached", "telemetry"),
        candidates_constraint_penalized=_non_negative_int(
            data, "candidates_constraint_penalized", "telemetry"
        ),
        promoted_by_stage=_counter_mapping(data, "promoted_by_stage", "telemetry"),
        eliminated_by_stage=_counter_mapping(data, "eliminated_by_stage", "telemetry"),
        cost_by_stage=_cost_mapping(data, "cost_by_stage", "telemetry"),
    )


def event_record_to_checkpoint(event: EventRecord) -> dict[str, Any]:
    """Serialise an EventRecord to a JSON-safe checkpoint dict."""
    return event.to_dict()


def event_record_from_checkpoint(payload: object) -> EventRecord:
    """Deserialise an EventRecord from a checkpoint payload."""
    data = _require_mapping(payload, "event")
    params = _optional_mapping_value(data, "params", "event")
    try:
        return EventRecord(
            event_index=_non_negative_int(data, "event_index", "event"),
            event_type=_literal_value(
                data.get("event_type"),
                key="event_type",
                label="event",
                allowed=_EVENT_TYPES,
            ),
            batch_id=_optional_str(data, "batch_id", "event"),
            candidate_id=_optional_str(data, "candidate_id", "event"),
            candidate_hash=_optional_str(data, "candidate_hash", "event"),
            generation=data.get("generation"),
            stage=_optional_str(data, "stage", "event"),
            confidence=_literal_value(
                data.get("confidence"),
                key="confidence",
                label="event",
                allowed=_EVALUATION_CONFIDENCES,
                optional=True,
            ),
            raw_score=_optional_finite_float(data, "raw_score", "event"),
            comparison_score=_optional_finite_float(data, "comparison_score", "event"),
            cost=_finite_float(data, "cost", "event"),
            status=_literal_value(
                data.get("status"),
                key="status",
                label="event",
                allowed=_CANDIDATE_STATUSES,
                optional=True,
            ),
            origin=_literal_value(
                data.get("origin"),
                key="origin",
                label="event",
                allowed=_CANDIDATE_ORIGINS,
                optional=True,
            ),
            parents=tuple(data.get("parents") or ()),
            genes=tuple(data.get("genes") or ()),
            params=dict(params) if params is not None else None,
            metrics=dict(_require_mapping_value(data, "metrics", "event")),
            metadata=dict(_require_mapping_value(data, "metadata", "event")),
        )
    except ConfigurationError as exc:
        raise CheckpointError(str(exc)) from exc


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
