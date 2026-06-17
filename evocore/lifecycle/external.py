"""External-state integration helpers for ask/tell optimizers."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from evocore.core.errors import ConfigurationError, FitnessError
from evocore.core.serialization import json_safe
from evocore.lifecycle.records import (
    STATE_UPDATE_CONFIDENCES,
    Candidate,
    CandidateOrigin,
    CandidateStatus,
    Direction,
    EvaluationConfidence,
    EvaluationRecord,
    ScoreObservation,
    score_for_direction,
)
from evocore.lifecycle.telemetry import AcceptanceDecision, OptimizationTelemetry
from evocore.search_space import GeneSpace, GeneValue

WarmStartMode = Literal["state", "tracked"]
InjectionMode = Literal["proposed", "tracked"]
SnapshotScope = Literal["trusted", "known", "pending", "scored"]
CmaMeanStrategy = Literal["best", "top_k_centroid"]
CacheValue = float | Mapping[str, object]
CacheLookup = Mapping[str, CacheValue] | Callable[["CandidateSnapshot"], CacheValue | None]


def json_safe_mapping(
    value: Mapping[str, object] | None,
    *,
    field_name: str,
) -> dict[str, object]:
    """Return a JSON-safe mapping copy."""
    payload = json_safe(dict(value or {}))
    if not isinstance(payload, dict):
        raise ConfigurationError(f"{field_name} must be a JSON-safe mapping.")
    return payload


@dataclass(frozen=True)
class WarmStartRecord:
    """Describe one scored candidate supplied from external trusted state."""

    values: tuple[GeneValue, ...] | None = None
    params: Mapping[str, GeneValue] | None = None
    score: float = 0.0
    confidence: EvaluationConfidence = "cached"
    stage: str = "warm_start"
    cost: float = 0.0
    metrics: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        has_values = self.values is not None
        has_params = self.params is not None
        if has_values == has_params:
            raise ConfigurationError("WarmStartRecord requires values or params, not both.")
        if self.confidence not in STATE_UPDATE_CONFIDENCES:
            raise ConfigurationError("WarmStartRecord confidence must be trusted_full or cached.")
        if not isinstance(self.stage, str) or not self.stage:
            raise ConfigurationError("WarmStartRecord stage must be a non-empty string.")
        if not math.isfinite(float(self.score)):
            raise ConfigurationError("WarmStartRecord score must be finite.")
        if not math.isfinite(float(self.cost)) or float(self.cost) < 0.0:
            raise ConfigurationError("WarmStartRecord cost must be finite and >= 0.")

        if has_values:
            object.__setattr__(self, "values", tuple(self.values or ()))
        if has_params:
            object.__setattr__(self, "params", json_safe_mapping(self.params, field_name="params"))
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "cost", float(self.cost))
        object.__setattr__(
            self,
            "metrics",
            json_safe_mapping(self.metrics, field_name="metrics"),
        )
        object.__setattr__(
            self,
            "metadata",
            json_safe_mapping(self.metadata, field_name="metadata"),
        )


@dataclass(frozen=True)
class CandidateSnapshot:
    """Read-only candidate snapshot for external reporting and promotion logic."""

    candidate_id: str
    candidate_hash: str
    values: tuple[GeneValue, ...]
    params: Mapping[str, GeneValue] | None
    origin: CandidateOrigin
    batch_id: str
    event_index: int
    generation: int | None
    status: CandidateStatus
    stage: str | None
    confidence: EvaluationConfidence | None
    score: float | None
    scores: Mapping[str, ScoreObservation]
    cost: float
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class PopulationSnapshot:
    """Read-only optimizer candidate collection snapshot."""

    optimizer_type: str
    direction: Direction
    event_index: int
    pending_batch_ids: tuple[str, ...]
    trusted_count: int
    candidates: tuple[CandidateSnapshot, ...]
    telemetry: OptimizationTelemetry
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ExternalStateCapabilities:
    """Describe optimizer-specific external-state support."""

    warm_start_before_ask: bool
    warm_start_after_ask: bool
    proposed_candidate_injection: bool
    state_candidate_injection: bool
    tracked_only_injection: bool
    population_snapshots: bool
    top_candidate_snapshots: bool
    cached_record_helpers: bool


@dataclass(frozen=True)
class InjectionResult:
    """Summarize one external candidate injection call."""

    accepted: tuple[CandidateSnapshot, ...]
    skipped_duplicates: tuple[CandidateSnapshot, ...]
    rejected: tuple[Mapping[str, object], ...]
    acceptance_decisions: tuple[AcceptanceDecision, ...] = ()


def resolve_warm_start_values(
    record: WarmStartRecord,
    gene_space: GeneSpace,
) -> tuple[GeneValue, ...]:
    """Resolve a warm-start record into decoded values for a gene space."""
    if record.values is not None:
        values = tuple(record.values)
    else:
        params = dict(record.params or {})
        expected = set(gene_space.names)
        provided = set(params)
        unknown = sorted(provided - expected)
        missing = sorted(expected - provided)
        if unknown:
            raise ConfigurationError(
                f"WarmStartRecord contains unknown parameter(s): {unknown!r}."
            )
        if missing:
            raise ConfigurationError(f"WarmStartRecord missing parameter(s): {missing!r}.")
        values = tuple(params[name] for name in gene_space.names)
    gene_space.validate_genes(values)
    return values


def _best_raw_score(
    candidate: Candidate,
    direction: Direction,
    confidences: tuple[EvaluationConfidence, ...] | None,
) -> float | None:
    values = [
        observation.score
        for observation in candidate.scores.values()
        if observation.score is not None
        and (confidences is None or observation.confidence in confidences)
    ]
    if not values:
        return None
    if direction == "minimize":
        return min(float(value) for value in values)
    if direction == "maximize":
        return max(float(value) for value in values)
    raise ConfigurationError("direction must be 'maximize' or 'minimize'.")


def build_candidate_snapshot(
    candidate: Candidate,
    *,
    gene_space: GeneSpace,
    direction: Direction,
) -> CandidateSnapshot:
    """Build a detached read-only snapshot for one candidate."""
    score = _best_raw_score(candidate, direction, STATE_UPDATE_CONFIDENCES)
    if score is None:
        score = _best_raw_score(candidate, direction, None)
    if score is not None and not math.isfinite(float(score)):
        score = None

    return CandidateSnapshot(
        candidate_id=candidate.candidate_id,
        candidate_hash=candidate.candidate_hash(gene_space),
        values=tuple(copy.deepcopy(candidate.genes)),
        params=copy.deepcopy(candidate.params),
        origin=candidate.origin,
        batch_id=candidate.batch_id,
        event_index=candidate.event_index,
        generation=candidate.generation,
        status=candidate.status,
        stage=candidate.stage,
        confidence=candidate.confidence,
        score=None if score is None else float(score),
        scores=copy.deepcopy(candidate.scores),
        cost=float(candidate.cost),
        metadata=copy.deepcopy(candidate.metadata),
    )


def build_population_snapshot(
    *,
    optimizer_type: str,
    direction: Direction,
    event_index: int,
    pending_batch_ids: Sequence[str],
    trusted_count: int,
    candidates: Sequence[Candidate],
    gene_space: GeneSpace,
    telemetry: OptimizationTelemetry,
    metadata: Mapping[str, object] | None = None,
) -> PopulationSnapshot:
    """Build a detached read-only optimizer population snapshot."""
    return PopulationSnapshot(
        optimizer_type=str(optimizer_type),
        direction=direction,
        event_index=int(event_index),
        pending_batch_ids=tuple(str(batch_id) for batch_id in pending_batch_ids),
        trusted_count=int(trusted_count),
        candidates=tuple(
            build_candidate_snapshot(candidate, gene_space=gene_space, direction=direction)
            for candidate in candidates
        ),
        telemetry=copy.deepcopy(telemetry),
        metadata=json_safe_mapping(metadata, field_name="metadata"),
    )


def top_candidate_snapshots(
    candidates: Sequence[Candidate],
    *,
    k: int,
    gene_space: GeneSpace,
    direction: Direction,
    confidence: tuple[EvaluationConfidence, ...],
) -> tuple[CandidateSnapshot, ...]:
    """Return top-k detached snapshots by score for the requested confidence set."""
    count = int(k)
    if count < 0:
        raise ConfigurationError("top candidate count k must be >= 0.")
    if count == 0:
        return ()

    ranked: list[tuple[float, Candidate]] = []
    for candidate in candidates:
        raw_score = _best_raw_score(candidate, direction, confidence)
        if raw_score is None or not math.isfinite(float(raw_score)):
            continue
        ranked.append((score_for_direction(float(raw_score), direction), candidate))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return tuple(
        build_candidate_snapshot(candidate, gene_space=gene_space, direction=direction)
        for _, candidate in ranked[:count]
    )


def cached_records(
    candidates: Sequence[Candidate],
    *,
    gene_space: GeneSpace,
    cache: CacheLookup,
    stage: str = "cached",
    cost: float = 0.0,
    metadata: Mapping[str, object] | None = None,
) -> tuple[EvaluationRecord, ...]:
    """Convert cached candidate scores into cached evaluation records."""
    if not stage:
        raise FitnessError("cached_records stage must be non-empty.")
    if not math.isfinite(float(cost)) or float(cost) < 0.0:
        raise FitnessError("cached_records cost must be finite and >= 0.")

    helper_metadata = json_safe_mapping(metadata, field_name="metadata")
    output: list[EvaluationRecord] = []
    for candidate in candidates:
        snapshot = build_candidate_snapshot(candidate, gene_space=gene_space, direction="maximize")
        cache_key = snapshot.candidate_hash
        raw = cache(snapshot) if callable(cache) else cache.get(cache_key)
        if raw is None:
            continue

        metrics: dict[str, object] = {}
        entry_metadata: dict[str, object] = {}
        if isinstance(raw, Mapping):
            if "score" not in raw:
                raise FitnessError("cached record mapping requires a score.")
            score_value = raw["score"]
            raw_metrics = raw.get("metrics", {})
            raw_metadata = raw.get("metadata", {})
            if not isinstance(raw_metrics, Mapping):
                raise FitnessError("cached record metrics must be a mapping.")
            if not isinstance(raw_metadata, Mapping):
                raise FitnessError("cached record metadata must be a mapping.")
            metrics = json_safe_mapping(raw_metrics, field_name="metrics")
            entry_metadata = json_safe_mapping(raw_metadata, field_name="metadata")
        else:
            score_value = raw

        score = float(score_value)
        if not math.isfinite(score):
            raise FitnessError("cached record requires a finite score.")

        record_metadata = dict(helper_metadata)
        record_metadata["cache_key"] = cache_key
        record_metadata.update(entry_metadata)
        output.append(
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=score,
                confidence="cached",
                stage=stage,
                cost=float(cost),
                metrics=metrics,
                metadata=record_metadata,
            )
        )
    return tuple(output)


__all__ = [
    "CacheLookup",
    "CacheValue",
    "CandidateSnapshot",
    "CmaMeanStrategy",
    "ExternalStateCapabilities",
    "InjectionMode",
    "InjectionResult",
    "PopulationSnapshot",
    "SnapshotScope",
    "WarmStartMode",
    "WarmStartRecord",
    "build_candidate_snapshot",
    "build_population_snapshot",
    "cached_records",
    "json_safe_mapping",
    "resolve_warm_start_values",
    "top_candidate_snapshots",
]
