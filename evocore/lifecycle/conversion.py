"""Conversions between lifecycle candidates and result solutions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from evocore.lifecycle.records import (
    STATE_UPDATE_CONFIDENCES,
    Candidate,
    CandidateOrigin,
    Direction,
)
from evocore.search_space.genes import GeneSpace
from evocore.search_space.solutions import GeneValue, Solution


def _has_state_observation(candidate: Candidate) -> bool:
    return any(
        observation.score is not None and observation.confidence in STATE_UPDATE_CONFIDENCES
        for observation in candidate.scores.values()
    )


def candidate_to_solution(
    candidate: Candidate,
    *,
    direction: Direction,
    gene_space: GeneSpace | None = None,
    include_provenance: bool = True,
) -> Solution:
    """Convert a lifecycle candidate into a population/result solution."""
    score_valid = _has_state_observation(candidate)
    score = candidate.best_state_score(direction) if score_valid else None
    metadata: dict[str, Any] = {}

    if candidate.params is not None:
        metadata["params"] = dict(candidate.params)

    if include_provenance:
        metadata["candidate_id"] = candidate.candidate_id
        if gene_space is not None:
            metadata["candidate_hash"] = gene_space.value_hash(candidate.genes)
        if candidate.batch_id:
            metadata["batch_id"] = candidate.batch_id
        metadata["origin"] = candidate.origin
        if candidate.generation is not None:
            metadata["generation"] = candidate.generation

    return Solution(
        list(candidate.genes),
        score=score,
        score_valid=score_valid,
        metadata=metadata,
    )


def solution_to_candidate(
    solution: Solution,
    *,
    gene_space: GeneSpace,
    candidate_id: str,
    batch_id: str,
    origin: CandidateOrigin,
    event_index: int,
    parents: Sequence[str] = (),
    generation: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Candidate:
    """Convert a population/result solution into a fresh lifecycle candidate."""
    values: list[GeneValue] = list(solution.values)
    gene_space.validate_genes(values)
    return Candidate(
        candidate_id=candidate_id,
        genes=values,
        batch_id=batch_id,
        params=gene_space.params_for(values),
        origin=origin,
        parents=tuple(parents),
        event_index=event_index,
        generation=generation,
        metadata=dict(metadata or {}),
    )


__all__ = ["candidate_to_solution", "solution_to_candidate"]
