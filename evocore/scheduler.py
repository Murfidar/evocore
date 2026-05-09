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

    def promote(self, candidates: Sequence[Candidate], *, completed_rung: str) -> list[Candidate]:
        """Promote the top candidate fraction after a completed rung."""
        if completed_rung not in self.policy.rung_names:
            raise ConfigurationError(f"unknown rung: {completed_rung!r}")

        rung = self.policy.rungs[self.policy.rung_names.index(completed_rung)]
        ranked = sorted(
            candidates, key=lambda candidate: candidate.best_observed_score(), reverse=True
        )
        promote_count = max(1, int(math.ceil(len(ranked) * rung.promote_fraction)))
        promoted = ranked[:promote_count]
        promoted_ids = {candidate.candidate_id for candidate in promoted}
        for candidate in ranked:
            if candidate.candidate_id in promoted_ids:
                candidate.status = "promoted"
            else:
                candidate.status = "eliminated"
        return promoted
