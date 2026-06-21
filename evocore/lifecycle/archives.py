"""Archive utilities for expensive external optimization workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, cast

from evocore.core.errors import ConfigurationError
from evocore.core.serialization import json_safe, stable_json_dumps
from evocore.lifecycle.external import CandidateSnapshot, PopulationSnapshot, WarmStartRecord
from evocore.lifecycle.records import (
    STATE_UPDATE_CONFIDENCES,
    Direction,
    EvaluationConfidence,
    ScoreObservation,
    score_for_direction,
)
from evocore.search_space import GeneValue

ARCHIVE_SCHEMA_VERSION = 1
DuplicatePolicy = Literal["keep_first", "keep_latest", "keep_best"]


def _validate_direction(direction: Direction) -> Direction:
    if direction not in ("maximize", "minimize"):
        raise ConfigurationError("score_direction must be 'maximize' or 'minimize'.")
    return direction


def _validate_duplicate_policy(policy: DuplicatePolicy) -> DuplicatePolicy:
    if policy not in ("keep_first", "keep_latest", "keep_best"):
        raise ConfigurationError(
            "duplicate_policy must be 'keep_first', 'keep_latest', or 'keep_best'."
        )
    return policy


def _json_mapping(value: object, *, field_name: str) -> dict[str, object]:
    payload = json_safe(value)
    if not isinstance(payload, dict):
        raise ConfigurationError(f"{field_name} must be a JSON-safe mapping.")
    return payload


def _snapshot_score_observation(snapshot: CandidateSnapshot) -> ScoreObservation | None:
    if snapshot.score is None:
        return None

    def matches(observation: ScoreObservation | None) -> bool:
        return bool(
            observation is not None
            and observation.score is not None
            and observation.confidence in STATE_UPDATE_CONFIDENCES
            and float(observation.score) == float(snapshot.score)
        )

    if snapshot.stage is not None:
        current = snapshot.scores.get(snapshot.stage)
        if matches(current):
            return current
    for observation in reversed(tuple(snapshot.scores.values())):
        if matches(observation):
            return observation
    return None


@dataclass(frozen=True)
class ArchiveEntry:
    """Stored scored candidate snapshot in a user-owned archive."""

    candidate_id: str
    candidate_hash: str
    values: tuple[GeneValue, ...]
    params: dict[str, GeneValue] | None
    score: float
    confidence: EvaluationConfidence
    stage: str
    cost: float
    metrics: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    source: str = "archive"
    inserted_index: int = 0

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ConfigurationError("ArchiveEntry candidate_id must be non-empty.")
        if not self.candidate_hash:
            raise ConfigurationError("ArchiveEntry candidate_hash must be non-empty.")
        if self.confidence not in ("trusted_full", "cached"):
            raise ConfigurationError("ArchiveEntry confidence must be trusted_full or cached.")
        if not self.stage:
            raise ConfigurationError("ArchiveEntry stage must be non-empty.")
        if not math.isfinite(float(self.score)):
            raise ConfigurationError("ArchiveEntry score must be finite.")
        if not math.isfinite(float(self.cost)) or float(self.cost) < 0.0:
            raise ConfigurationError("ArchiveEntry cost must be finite and >= 0.")
        if not self.source:
            raise ConfigurationError("ArchiveEntry source must be non-empty.")
        object.__setattr__(self, "values", tuple(self.values))
        if self.params is not None:
            object.__setattr__(self, "params", _json_mapping(self.params, field_name="params"))
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "cost", float(self.cost))
        object.__setattr__(self, "metrics", _json_mapping(self.metrics, field_name="metrics"))
        object.__setattr__(
            self,
            "metadata",
            _json_mapping(self.metadata, field_name="metadata"),
        )
        object.__setattr__(self, "inserted_index", int(self.inserted_index))

    @classmethod
    def from_snapshot(
        cls,
        snapshot: CandidateSnapshot,
        *,
        source: str,
        inserted_index: int,
    ) -> ArchiveEntry:
        """Create an archive entry from a detached scored candidate snapshot."""
        if snapshot.score is None or not math.isfinite(float(snapshot.score)):
            raise ConfigurationError("ArchiveEntry requires a finite score.")
        observation = _snapshot_score_observation(snapshot)
        confidence = snapshot.confidence if observation is None else observation.confidence
        stage = snapshot.stage if observation is None else observation.stage
        if confidence not in ("trusted_full", "cached"):
            raise ConfigurationError("ArchiveEntry confidence must be trusted_full or cached.")
        if not stage:
            raise ConfigurationError("ArchiveEntry stage must be non-empty.")
        metrics = {} if observation is None else dict(observation.metrics)
        metadata = dict(snapshot.metadata)
        if observation is not None:
            metadata["metrics"] = dict(observation.metrics)
            metadata["record_metadata"] = dict(observation.metadata)
        return cls(
            candidate_id=snapshot.candidate_id,
            candidate_hash=snapshot.candidate_hash,
            values=tuple(snapshot.values),
            params=None if snapshot.params is None else dict(snapshot.params),
            score=float(snapshot.score),
            confidence=confidence,
            stage=stage,
            cost=float(snapshot.cost),
            metrics=metrics,
            metadata=metadata,
            source=source,
            inserted_index=inserted_index,
        )

    def to_warm_start_record(
        self,
        *,
        stage: str | None = None,
        confidence: EvaluationConfidence | None = None,
    ) -> WarmStartRecord:
        """Export this archive entry through the existing warm-start record type."""
        metadata = dict(self.metadata)
        metadata.update(
            {
                "archive_candidate_id": self.candidate_id,
                "archive_candidate_hash": self.candidate_hash,
                "archive_source": self.source,
            }
        )
        return WarmStartRecord(
            values=None if self.params is not None else self.values,
            params=self.params,
            score=self.score,
            confidence=confidence or self.confidence,
            stage=stage or self.stage,
            cost=self.cost,
            metrics=dict(self.metrics),
            metadata=metadata,
        )


@dataclass(frozen=True)
class ArchiveExport:
    """Warm-start records exported from an archive with selected identity data."""

    records: tuple[WarmStartRecord, ...]
    selected_hashes: tuple[str, ...]


@dataclass(frozen=True)
class ArchivePolicy:
    """Configuration for archive duplicate handling and score direction."""

    duplicate_policy: DuplicatePolicy = "keep_best"
    score_direction: Direction = "maximize"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "duplicate_policy",
            _validate_duplicate_policy(self.duplicate_policy),
        )
        object.__setattr__(self, "score_direction", _validate_direction(self.score_direction))


class CandidateArchive:
    """User-owned durable archive for scored candidate snapshots."""

    def __init__(
        self,
        *,
        duplicate_policy: DuplicatePolicy = "keep_best",
        score_direction: Direction = "maximize",
    ) -> None:
        self.duplicate_policy = _validate_duplicate_policy(duplicate_policy)
        self.score_direction = _validate_direction(score_direction)
        self._entries_by_hash: dict[str, ArchiveEntry] = {}
        self._next_inserted_index = 0

    @property
    def entries(self) -> tuple[ArchiveEntry, ...]:
        """Return archive entries in insertion order."""
        return tuple(
            sorted(self._entries_by_hash.values(), key=lambda entry: entry.inserted_index)
        )

    def add_population(
        self,
        snapshot: PopulationSnapshot,
        *,
        source: str,
    ) -> tuple[ArchiveEntry, ...]:
        """Add all candidates from a population snapshot and inherit its direction."""
        if self._entries_by_hash and snapshot.direction != self.score_direction:
            raise ConfigurationError(
                "PopulationSnapshot direction cannot change a non-empty CandidateArchive."
            )
        self.score_direction = snapshot.direction
        return self.add_candidates(snapshot.candidates, source=source)

    def add_candidates(
        self,
        candidates: tuple[CandidateSnapshot, ...] | list[CandidateSnapshot],
        *,
        source: str,
    ) -> tuple[ArchiveEntry, ...]:
        """Add scored snapshots according to the configured duplicate policy."""
        if not source:
            raise ConfigurationError("source must be non-empty.")
        accepted: list[ArchiveEntry] = []
        for snapshot in candidates:
            entry = ArchiveEntry.from_snapshot(
                snapshot,
                source=source,
                inserted_index=self._next_inserted_index,
            )
            self._next_inserted_index += 1
            existing = self._entries_by_hash.get(entry.candidate_hash)
            if existing is None or self._should_replace(existing, entry):
                self._entries_by_hash[entry.candidate_hash] = entry
                accepted.append(entry)
        return tuple(accepted)

    def _should_replace(self, existing: ArchiveEntry, incoming: ArchiveEntry) -> bool:
        if self.duplicate_policy == "keep_first":
            return False
        if self.duplicate_policy == "keep_latest":
            return True
        existing_score = score_for_direction(existing.score, self.score_direction)
        incoming_score = score_for_direction(incoming.score, self.score_direction)
        if incoming_score == existing_score:
            return incoming.inserted_index > existing.inserted_index
        return incoming_score > existing_score

    def ranked_entries(self) -> tuple[ArchiveEntry, ...]:
        """Return archive entries sorted best-first for the configured direction."""
        return tuple(
            sorted(
                self.entries,
                key=lambda entry: (
                    score_for_direction(entry.score, self.score_direction),
                    -entry.inserted_index,
                ),
                reverse=True,
            )
        )

    def to_warm_start_records(
        self,
        *,
        k: int | None = None,
        stage: str | None = None,
        confidence: EvaluationConfidence | None = None,
    ) -> tuple[WarmStartRecord, ...]:
        """Export best archive entries to existing warm-start records."""
        if k is not None and int(k) < 0:
            raise ConfigurationError("k must be >= 0.")
        entries = self.ranked_entries() if k is None else self.ranked_entries()[: int(k)]
        return tuple(
            entry.to_warm_start_record(stage=stage, confidence=confidence) for entry in entries
        )

    def to_dict(self) -> dict[str, object]:
        """Export this archive as a schema-versioned JSON-safe payload."""
        return {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "duplicate_policy": self.duplicate_policy,
            "score_direction": self.score_direction,
            "next_inserted_index": self._next_inserted_index,
            "entries": [
                {
                    "candidate_id": entry.candidate_id,
                    "candidate_hash": entry.candidate_hash,
                    "values": list(entry.values),
                    "params": entry.params,
                    "score": entry.score,
                    "confidence": entry.confidence,
                    "stage": entry.stage,
                    "cost": entry.cost,
                    "metrics": entry.metrics,
                    "metadata": entry.metadata,
                    "source": entry.source,
                    "inserted_index": entry.inserted_index,
                }
                for entry in self.entries
            ],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Export this archive as deterministic JSON."""
        return stable_json_dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> CandidateArchive:
        """Restore a candidate archive from a schema-versioned mapping."""
        if payload.get("schema_version") != ARCHIVE_SCHEMA_VERSION:
            raise ConfigurationError("Unsupported CandidateArchive schema_version.")
        archive = cls(
            duplicate_policy=cast("DuplicatePolicy", payload.get("duplicate_policy", "keep_best")),
            score_direction=cast("Direction", payload.get("score_direction", "maximize")),
        )
        entries = payload.get("entries", [])
        if not isinstance(entries, list):
            raise ConfigurationError("CandidateArchive entries must be a list.")
        for raw in entries:
            if not isinstance(raw, dict):
                raise ConfigurationError("CandidateArchive entry must be a mapping.")
            values = raw.get("values", ())
            if not isinstance(values, list | tuple):
                raise ConfigurationError("CandidateArchive entry values must be a sequence.")
            raw_params = raw.get("params")
            if raw_params is not None and not isinstance(raw_params, dict):
                raise ConfigurationError("CandidateArchive entry params must be a mapping.")
            entry = ArchiveEntry(
                candidate_id=str(raw["candidate_id"]),
                candidate_hash=str(raw["candidate_hash"]),
                values=tuple(values),
                params=None if raw_params is None else dict(raw_params),
                score=float(cast("float", raw["score"])),
                confidence=cast("EvaluationConfidence", raw["confidence"]),
                stage=str(raw["stage"]),
                cost=float(raw.get("cost", 0.0)),
                metrics=dict(cast("dict[str, object]", raw.get("metrics", {}))),
                metadata=dict(cast("dict[str, object]", raw.get("metadata", {}))),
                source=str(raw.get("source", "archive")),
                inserted_index=int(raw.get("inserted_index", len(archive._entries_by_hash))),
            )
            archive._entries_by_hash[entry.candidate_hash] = entry
            archive._next_inserted_index = max(
                archive._next_inserted_index,
                entry.inserted_index + 1,
            )
        raw_next_inserted_index = payload.get("next_inserted_index")
        if raw_next_inserted_index is not None:
            if not isinstance(raw_next_inserted_index, int) or isinstance(
                raw_next_inserted_index, bool
            ):
                raise ConfigurationError(
                    "CandidateArchive next_inserted_index must be an integer."
                )
            if raw_next_inserted_index < archive._next_inserted_index:
                raise ConfigurationError(
                    "CandidateArchive next_inserted_index cannot precede stored entries."
                )
            archive._next_inserted_index = raw_next_inserted_index
        return archive


__all__ = [
    "ARCHIVE_SCHEMA_VERSION",
    "ArchiveEntry",
    "ArchiveExport",
    "ArchivePolicy",
    "CandidateArchive",
    "DuplicatePolicy",
]
