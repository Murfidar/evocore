"""vNext multi-fidelity schedulers."""

from __future__ import annotations

import math
from collections.abc import Sequence

from evocore.evaluation import Candidate
from evocore.exceptions import ConfigurationError
from evocore.policies import MultiFidelityPolicy


class EvaluationScheduler:
    """Schedule candidates across multi-fidelity rungs."""

    def __init__(self, policy: MultiFidelityPolicy) -> None:
        self.policy = policy

    def rung_after(self, completed_rung: str) -> str | None:
        """Return the next rung name after a completed rung."""
        names = self.policy.rung_names
        if completed_rung not in names:
            raise ConfigurationError(f"unknown rung: {completed_rung!r}")
        index = names.index(completed_rung)
        if index + 1 >= len(names):
            return None
        return names[index + 1]

    def assign_rung(self, candidates: Sequence[Candidate], *, rung_name: str) -> list[Candidate]:
        """Assign a rung to candidates selected for evaluation."""
        if rung_name not in self.policy.rung_names:
            raise ConfigurationError(f"unknown rung: {rung_name!r}")
        assigned = list(candidates)
        for candidate in assigned:
            candidate.rung = rung_name
            candidate.status = "racing"
        return assigned

    @staticmethod
    def _score_for_rung(candidate: Candidate, rung_name: str) -> float:
        score = candidate.scores.get(rung_name)
        if score is None or score.score is None:
            return float("-inf")
        return float(score.score)

    def promote(self, candidates: Sequence[Candidate], *, completed_rung: str) -> list[Candidate]:
        """Promote top candidates plus deterministic audit samples."""
        if completed_rung not in self.policy.rung_names:
            raise ConfigurationError(f"unknown rung: {completed_rung!r}")

        rung = self.policy.rungs[self.policy.rung_names.index(completed_rung)]
        ranked = sorted(
            candidates,
            key=lambda candidate: self._score_for_rung(candidate, completed_rung),
            reverse=True,
        )
        promote_count = max(1, int(math.ceil(len(ranked) * rung.promote_fraction)))
        exploration_count = int(math.floor(len(ranked) * self.policy.exploration_fraction))
        audit_count = int(math.floor(len(ranked) * self.policy.audit_fraction))
        promoted = list(ranked[:promote_count])
        promoted_ids = {candidate.candidate_id for candidate in promoted}

        if exploration_count > 0 and len(ranked) > promote_count:
            exploration_pool = [
                candidate
                for candidate in reversed(ranked)
                if candidate.candidate_id not in promoted_ids
            ]
            exploration = exploration_pool[:exploration_count]
            promoted.extend(exploration)
            promoted_ids.update(candidate.candidate_id for candidate in exploration)

        if audit_count > 0 and len(ranked) > len(promoted):
            audit_pool = [
                candidate
                for candidate in ranked[promote_count:]
                if candidate.candidate_id not in promoted_ids
            ]
            audit = audit_pool[:audit_count]
            promoted.extend(audit)
            promoted_ids.update(candidate.candidate_id for candidate in audit)

        for candidate in ranked:
            if candidate.candidate_id in promoted_ids:
                candidate.status = "promoted"
            else:
                candidate.status = "eliminated"
        return promoted
