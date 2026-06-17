"""Selection utilities for public candidate snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from evocore.core.errors import ConfigurationError
from evocore.core.serialization import json_safe
from evocore.lifecycle.external import CandidateSnapshot, WarmStartRecord
from evocore.lifecycle.records import Direction, EvaluationConfidence, score_for_direction

DuplicateSelectionPolicy = Literal["allow", "suppress"]
MissingMetadataPolicy = Literal["unknown", "error"]


def _json_mapping(value: object, *, field_name: str) -> dict[str, object]:
    payload = json_safe(value)
    if not isinstance(payload, dict):
        raise ConfigurationError(f"{field_name} must be a JSON-safe mapping.")
    return payload


@dataclass(frozen=True)
class FamilyQuota:
    """Maximum selected candidates per metadata family bucket."""

    metadata_key: str
    max_count: int
    unknown_value: str = "unknown"

    def __post_init__(self) -> None:
        if not self.metadata_key:
            raise ConfigurationError("FamilyQuota metadata_key must be non-empty.")
        if int(self.max_count) <= 0:
            raise ConfigurationError("FamilyQuota max_count must be positive.")
        object.__setattr__(self, "max_count", int(self.max_count))


@dataclass(frozen=True)
class SpecialistCap:
    """Maximum selected candidates per metadata specialist bucket."""

    metadata_key: str
    max_count: int
    unknown_value: str = "unknown"

    def __post_init__(self) -> None:
        if not self.metadata_key:
            raise ConfigurationError("SpecialistCap metadata_key must be non-empty.")
        if int(self.max_count) <= 0:
            raise ConfigurationError("SpecialistCap max_count must be positive.")
        object.__setattr__(self, "max_count", int(self.max_count))


@dataclass(frozen=True)
class SelectionDecision:
    """Explain why one candidate was selected, rejected, or skipped."""

    candidate_id: str
    candidate_hash: str
    selected: bool
    reason: str
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ConfigurationError("SelectionDecision candidate_id must be non-empty.")
        if not self.candidate_hash:
            raise ConfigurationError("SelectionDecision candidate_hash must be non-empty.")
        if not self.reason:
            raise ConfigurationError("SelectionDecision reason must be non-empty.")
        object.__setattr__(
            self,
            "metadata",
            _json_mapping(self.metadata, field_name="metadata"),
        )


@dataclass(frozen=True)
class SelectionResult:
    """Result of deterministic survivor selection over candidate snapshots."""

    selected: tuple[CandidateSnapshot, ...]
    rejected: tuple[CandidateSnapshot, ...]
    skipped: tuple[CandidateSnapshot, ...]
    decisions: tuple[SelectionDecision, ...]
    summary: dict[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected", tuple(self.selected))
        object.__setattr__(self, "rejected", tuple(self.rejected))
        object.__setattr__(self, "skipped", tuple(self.skipped))
        object.__setattr__(self, "decisions", tuple(self.decisions))
        object.__setattr__(self, "summary", _json_mapping(self.summary, field_name="summary"))

    def to_warm_start_records(
        self,
        *,
        stage: str,
        confidence: EvaluationConfidence = "cached",
    ) -> tuple[WarmStartRecord, ...]:
        """Export selected candidates through the existing warm-start record type."""
        return tuple(
            WarmStartRecord(
                values=None if candidate.params is not None else candidate.values,
                params=candidate.params,
                score=float(candidate.score),
                confidence=confidence,
                stage=stage,
                cost=float(candidate.cost),
                metadata=dict(candidate.metadata),
            )
            for candidate in self.selected
        )


def _metadata_bucket(
    candidate: CandidateSnapshot,
    key: str,
    *,
    missing_metadata: MissingMetadataPolicy,
    unknown_value: str,
) -> str:
    if key not in candidate.metadata:
        if missing_metadata == "error":
            raise ConfigurationError(
                f"Candidate {candidate.candidate_id!r} missing metadata key {key!r}."
            )
        return unknown_value
    return str(candidate.metadata[key])


def _ranked_candidates(
    candidates: list[CandidateSnapshot],
    *,
    score_direction: Direction,
) -> list[CandidateSnapshot]:
    if score_direction not in ("maximize", "minimize"):
        raise ConfigurationError("score_direction must be 'maximize' or 'minimize'.")
    return sorted(
        candidates,
        key=lambda item: (
            -score_for_direction(float(item.score), score_direction),
            int(item.event_index),
            item.candidate_id,
        ),
    )


def _summary(
    selected: list[CandidateSnapshot],
    rejected: list[CandidateSnapshot],
    skipped: list[CandidateSnapshot],
) -> dict[str, object]:
    selected_by_family: dict[str, int] = {}
    selected_by_confidence: dict[str, int] = {}
    for candidate in selected:
        family = str(candidate.metadata.get("family", "unknown"))
        selected_by_family[family] = selected_by_family.get(family, 0) + 1
        confidence = str(candidate.confidence or "unknown")
        selected_by_confidence[confidence] = selected_by_confidence.get(confidence, 0) + 1
    return {
        "selected": len(selected),
        "rejected": len(rejected),
        "skipped": len(skipped),
        "selected_by_family": selected_by_family,
        "selected_by_confidence": selected_by_confidence,
    }


def _decision(
    candidate: CandidateSnapshot,
    *,
    selected: bool,
    reason: str,
    metadata: dict[str, object] | None = None,
) -> SelectionDecision:
    return SelectionDecision(
        candidate_id=candidate.candidate_id,
        candidate_hash=candidate.candidate_hash,
        selected=selected,
        reason=reason,
        metadata=metadata or {},
    )


def _split_scored_candidates(
    candidates: list[CandidateSnapshot] | tuple[CandidateSnapshot, ...],
) -> tuple[list[CandidateSnapshot], list[CandidateSnapshot], list[SelectionDecision]]:
    scored: list[CandidateSnapshot] = []
    rejected: list[CandidateSnapshot] = []
    decisions: list[SelectionDecision] = []
    for candidate in candidates:
        if candidate.score is None:
            rejected.append(candidate)
            decisions.append(_decision(candidate, selected=False, reason="no_score"))
        else:
            scored.append(candidate)
    return scored, rejected, decisions


def _blocked_reason(
    candidate: CandidateSnapshot,
    *,
    quotas: tuple[FamilyQuota, ...],
    caps: tuple[SpecialistCap, ...],
    missing_metadata: MissingMetadataPolicy,
    quota_counts: dict[tuple[str, str], int],
    cap_counts: dict[tuple[str, str], int],
) -> str | None:
    for quota in quotas:
        bucket = _metadata_bucket(
            candidate,
            quota.metadata_key,
            missing_metadata=missing_metadata,
            unknown_value=quota.unknown_value,
        )
        key = (quota.metadata_key, bucket)
        if quota_counts.get(key, 0) >= quota.max_count:
            return f"quota:{quota.metadata_key}"
    for cap in caps:
        bucket = _metadata_bucket(
            candidate,
            cap.metadata_key,
            missing_metadata=missing_metadata,
            unknown_value=cap.unknown_value,
        )
        key = (cap.metadata_key, bucket)
        if cap_counts.get(key, 0) >= cap.max_count:
            return f"cap:{cap.metadata_key}"
    return None


def _record_bucket_counts(
    candidate: CandidateSnapshot,
    *,
    quotas: tuple[FamilyQuota, ...],
    caps: tuple[SpecialistCap, ...],
    missing_metadata: MissingMetadataPolicy,
    quota_counts: dict[tuple[str, str], int],
    cap_counts: dict[tuple[str, str], int],
) -> None:
    for quota in quotas:
        bucket = _metadata_bucket(
            candidate,
            quota.metadata_key,
            missing_metadata=missing_metadata,
            unknown_value=quota.unknown_value,
        )
        key = (quota.metadata_key, bucket)
        quota_counts[key] = quota_counts.get(key, 0) + 1
    for cap in caps:
        bucket = _metadata_bucket(
            candidate,
            cap.metadata_key,
            missing_metadata=missing_metadata,
            unknown_value=cap.unknown_value,
        )
        key = (cap.metadata_key, bucket)
        cap_counts[key] = cap_counts.get(key, 0) + 1


def select_candidates(
    candidates: list[CandidateSnapshot] | tuple[CandidateSnapshot, ...],
    *,
    k: int,
    score_direction: Direction,
    duplicate_policy: DuplicateSelectionPolicy = "suppress",
    quotas: list[FamilyQuota] | tuple[FamilyQuota, ...] = (),
    caps: list[SpecialistCap] | tuple[SpecialistCap, ...] = (),
    missing_metadata: MissingMetadataPolicy = "unknown",
) -> SelectionResult:
    """Select candidate snapshots with deterministic duplicate, quota, and cap handling."""
    if int(k) < 0:
        raise ConfigurationError("k must be >= 0.")
    if duplicate_policy not in ("allow", "suppress"):
        raise ConfigurationError("duplicate_policy must be 'allow' or 'suppress'.")
    if missing_metadata not in ("unknown", "error"):
        raise ConfigurationError("missing_metadata must be 'unknown' or 'error'.")

    quota_tuple = tuple(quotas)
    cap_tuple = tuple(caps)
    scored, rejected, decisions = _split_scored_candidates(candidates)
    skipped: list[CandidateSnapshot] = []
    selected: list[CandidateSnapshot] = []
    seen_hashes: set[str] = set()
    quota_counts: dict[tuple[str, str], int] = {}
    cap_counts: dict[tuple[str, str], int] = {}

    for candidate in _ranked_candidates(scored, score_direction=score_direction):
        decision_metadata = _json_mapping(dict(candidate.metadata), field_name="metadata")
        if duplicate_policy == "suppress" and candidate.candidate_hash in seen_hashes:
            rejected.append(candidate)
            decisions.append(
                _decision(
                    candidate,
                    selected=False,
                    reason="duplicate",
                    metadata=decision_metadata,
                )
            )
            continue

        blocked_reason = _blocked_reason(
            candidate,
            quotas=quota_tuple,
            caps=cap_tuple,
            missing_metadata=missing_metadata,
            quota_counts=quota_counts,
            cap_counts=cap_counts,
        )
        if blocked_reason is not None:
            rejected.append(candidate)
            decisions.append(
                _decision(
                    candidate,
                    selected=False,
                    reason=blocked_reason,
                    metadata=decision_metadata,
                )
            )
            continue

        if len(selected) >= int(k):
            skipped.append(candidate)
            decisions.append(
                _decision(
                    candidate,
                    selected=False,
                    reason="overflow",
                    metadata=decision_metadata,
                )
            )
            continue

        selected.append(candidate)
        seen_hashes.add(candidate.candidate_hash)
        _record_bucket_counts(
            candidate,
            quotas=quota_tuple,
            caps=cap_tuple,
            missing_metadata=missing_metadata,
            quota_counts=quota_counts,
            cap_counts=cap_counts,
        )
        decisions.append(
            _decision(
                candidate,
                selected=True,
                reason="selected",
                metadata=decision_metadata,
            )
        )

    return SelectionResult(
        selected=tuple(selected),
        rejected=tuple(rejected),
        skipped=tuple(skipped),
        decisions=tuple(decisions),
        summary=_summary(selected, rejected, skipped),
    )


__all__ = [
    "DuplicateSelectionPolicy",
    "FamilyQuota",
    "MissingMetadataPolicy",
    "SelectionDecision",
    "SelectionResult",
    "SpecialistCap",
    "select_candidates",
]
