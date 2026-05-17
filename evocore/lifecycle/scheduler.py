"""vNext multi-fidelity schedulers."""

from __future__ import annotations

import math
from collections.abc import Sequence

from evocore.core.errors import ConfigurationError
from evocore.lifecycle.policies import BudgetPolicy
from evocore.lifecycle.records import Candidate


class BudgetScheduler:
    """Schedule candidates across multi-fidelity stages."""

    def __init__(self, policy: BudgetPolicy) -> None:
        self.policy = policy

    def stage_after(self, completed_stage: str) -> str | None:
        """Return the next stage name after a completed stage."""
        names = self.policy.stage_names
        if completed_stage not in names:
            raise ConfigurationError(f"unknown stage: {completed_stage!r}")
        index = names.index(completed_stage)
        if index + 1 >= len(names):
            return None
        return names[index + 1]

    def assign_stage(self, candidates: Sequence[Candidate], *, stage_name: str) -> list[Candidate]:
        """Assign a stage to candidates selected for evaluation."""
        if stage_name not in self.policy.stage_names:
            raise ConfigurationError(f"unknown stage: {stage_name!r}")
        assigned = list(candidates)
        for candidate in assigned:
            candidate.stage = stage_name
            candidate.status = "racing"
        return assigned

    @staticmethod
    def _score_for_stage(candidate: Candidate, stage_name: str) -> float:
        score = candidate.scores.get(stage_name)
        if score is None or score.score is None:
            return float("-inf")
        return float(score.score)

    def promote(self, candidates: Sequence[Candidate], *, completed_stage: str) -> list[Candidate]:
        """Promote top candidates plus deterministic audit samples."""
        if completed_stage not in self.policy.stage_names:
            raise ConfigurationError(f"unknown stage: {completed_stage!r}")

        stage = self.policy.stages[self.policy.stage_names.index(completed_stage)]
        ranked = sorted(
            candidates,
            key=lambda candidate: self._score_for_stage(candidate, completed_stage),
            reverse=True,
        )
        promote_count = max(1, int(math.ceil(len(ranked) * stage.promote_fraction)))
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
