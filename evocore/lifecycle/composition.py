"""Composition helpers for nested expensive optimization workflows."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

from evocore import _core
from evocore.core.errors import ConfigurationError, FitnessError
from evocore.core.serialization import json_safe
from evocore.lifecycle.external import CandidateSnapshot
from evocore.lifecycle.records import Candidate, EvaluationConfidence, EvaluationRecord
from evocore.search_space import GeneSpace


def _json_metadata(value: Mapping[str, object] | None, *, field_name: str) -> dict[str, object]:
    payload = json_safe(dict(value or {}))
    if not isinstance(payload, dict):
        raise ConfigurationError(f"{field_name} must be a JSON-safe mapping.")
    return payload


def _outer_identity(
    outer_candidate: Candidate | CandidateSnapshot,
    *,
    gene_space: GeneSpace | None = None,
) -> tuple[str, str, str | None]:
    if isinstance(outer_candidate, CandidateSnapshot):
        return (
            outer_candidate.candidate_id,
            outer_candidate.candidate_hash,
            outer_candidate.batch_id,
        )
    if isinstance(outer_candidate, Candidate):
        if gene_space is None:
            raise ConfigurationError("gene_space is required when outer_candidate is Candidate.")
        return (
            outer_candidate.candidate_id,
            outer_candidate.candidate_hash(gene_space),
            outer_candidate.batch_id or None,
        )
    raise ConfigurationError("outer_candidate must be Candidate or CandidateSnapshot.")


def derive_child_seed(
    *,
    parent_seed: int,
    candidate_hash: str,
    stage: str,
) -> int:
    """Derive a deterministic nested optimizer seed from parent seed, hash, and stage."""
    if not candidate_hash:
        raise ConfigurationError("candidate_hash must be non-empty.")
    if not stage:
        raise ConfigurationError("stage must be non-empty.")
    digest = hashlib.sha256(f"{candidate_hash}:{stage}".encode()).digest()
    child_index = int.from_bytes(digest[:4], "big", signed=False)
    return int(_core.py_derive_seed(int(parent_seed), 0, child_index, _core.OP_MULTI_RUN)) % (
        2**32
    )


def lineage_metadata(
    *,
    outer_candidate: Candidate | CandidateSnapshot,
    inner_optimizer_type: str,
    inner_seed: int,
    stage: str,
    gene_space: GeneSpace | None = None,
    checkpoint_path: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build JSON-safe lineage metadata for a nested optimizer run."""
    if not inner_optimizer_type:
        raise ConfigurationError("inner_optimizer_type must be non-empty.")
    if not stage:
        raise ConfigurationError("stage must be non-empty.")
    candidate_id, candidate_hash, batch_id = _outer_identity(
        outer_candidate,
        gene_space=gene_space,
    )
    payload = {
        "outer_candidate_id": candidate_id,
        "outer_candidate_hash": candidate_hash,
        "inner_optimizer_type": inner_optimizer_type,
        "inner_seed": int(inner_seed),
        "composition_stage": stage,
    }
    if batch_id is not None:
        payload["outer_batch_id"] = batch_id
    if checkpoint_path is not None:
        payload["inner_checkpoint_path"] = str(checkpoint_path)
    payload.update(_json_metadata(metadata, field_name="metadata"))
    return _json_metadata(payload, field_name="lineage metadata")


def inner_result_record(
    *,
    outer_candidate: Candidate | CandidateSnapshot,
    score: float,
    confidence: EvaluationConfidence,
    stage: str,
    cost: float = 0.0,
    metrics: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
    gene_space: GeneSpace | None = None,
) -> EvaluationRecord:
    """Convert an inner optimizer result into an outer candidate evaluation record."""
    if not math.isfinite(float(score)):
        raise FitnessError("inner_result_record score must be finite.")
    candidate_id, _, batch_id = _outer_identity(outer_candidate, gene_space=gene_space)
    return EvaluationRecord(
        candidate_id=candidate_id,
        batch_id=batch_id,
        score=float(score),
        confidence=confidence,
        stage=stage,
        cost=float(cost),
        metrics=_json_metadata(metrics, field_name="metrics"),
        metadata=_json_metadata(metadata, field_name="metadata"),
    )


__all__ = [
    "derive_child_seed",
    "inner_result_record",
    "lineage_metadata",
]
