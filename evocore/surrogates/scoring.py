"""Surrogate ranking result types."""

from __future__ import annotations

from dataclasses import dataclass

from evocore.lifecycle import EvaluationConfidence


@dataclass(frozen=True)
class SurrogateScore:
    """Rank one candidate using an advisor."""

    candidate_id: str
    score: float
    confidence: EvaluationConfidence
    reason: str
