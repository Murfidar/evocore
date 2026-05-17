import pytest

from evocore.core.errors import ConfigurationError
from evocore.lifecycle import (
    BudgetPolicy,
    BudgetScheduler,
    Candidate,
    EvaluationRecord,
    EvaluationStage,
)


def _candidate(index: int) -> Candidate:
    return Candidate(
        candidate_id=f"c-{index}", genes=[float(index)], origin="random", event_index=0
    )


def test_policy_requires_unique_rung_names_and_max_evaluations() -> None:
    policy = BudgetPolicy(
        stages=[
            EvaluationStage("cheap", budget=0.10, promote_fraction=0.5, confidence="partial"),
            EvaluationStage("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full"),
        ],
        max_evaluations=32,
        batch_size=8,
        exploration_fraction=0.10,
        audit_fraction=0.05,
    )

    assert policy.max_evaluations == 32
    assert policy.stage_names == ("cheap", "full")
    assert policy.final_stage.name == "full"


def test_policy_rejects_duplicate_rung_names() -> None:
    with pytest.raises(ConfigurationError, match="duplicate stage"):
        BudgetPolicy(
            stages=[
                EvaluationStage("cheap", budget=0.10, promote_fraction=0.5, confidence="partial"),
                EvaluationStage(
                    "cheap", budget=1.0, promote_fraction=1.0, confidence="trusted_full"
                ),
            ],
            max_evaluations=16,
        )


def test_policy_rejects_missing_trusted_full_rung() -> None:
    with pytest.raises(ConfigurationError, match="trusted_full"):
        BudgetPolicy(
            stages=[
                EvaluationStage("cheap", budget=0.10, promote_fraction=0.5, confidence="partial")
            ],
            max_evaluations=16,
        )


def test_policy_rejects_invalid_budget_and_fractions() -> None:
    with pytest.raises(ConfigurationError, match="max_evaluations"):
        BudgetPolicy(
            stages=[
                EvaluationStage(
                    "full", budget=1.0, promote_fraction=1.0, confidence="trusted_full"
                )
            ],
            max_evaluations=0,
        )

    with pytest.raises(ConfigurationError, match="exploration_fraction"):
        BudgetPolicy(
            stages=[
                EvaluationStage(
                    "full", budget=1.0, promote_fraction=1.0, confidence="trusted_full"
                )
            ],
            max_evaluations=1,
            exploration_fraction=1.5,
        )


def test_policy_requires_trusted_full_rung_to_be_final() -> None:
    with pytest.raises(ConfigurationError, match="final stage"):
        BudgetPolicy(
            stages=[
                EvaluationStage(
                    "full", budget=1.0, promote_fraction=1.0, confidence="trusted_full"
                ),
                EvaluationStage("audit", budget=1.0, promote_fraction=1.0, confidence="partial"),
            ],
            max_evaluations=16,
        )


def test_policy_rejects_multiple_trusted_full_rungs() -> None:
    with pytest.raises(ConfigurationError, match="exactly one trusted_full"):
        BudgetPolicy(
            stages=[
                EvaluationStage("cheap", budget=0.1, promote_fraction=0.5, confidence="partial"),
                EvaluationStage(
                    "full_a", budget=1.0, promote_fraction=1.0, confidence="trusted_full"
                ),
                EvaluationStage(
                    "full_b", budget=1.0, promote_fraction=1.0, confidence="trusted_full"
                ),
            ],
            max_evaluations=16,
        )


def test_policy_rejects_legacy_full_evaluation_budget_name() -> None:
    with pytest.raises(TypeError, match="full_evaluation_budget"):
        BudgetPolicy(
            stages=[
                EvaluationStage(
                    "full", budget=1.0, promote_fraction=1.0, confidence="trusted_full"
                )
            ],
            full_evaluation_budget=16,
        )


def test_single_full_uses_max_evaluations_and_rejects_budget() -> None:
    policy = BudgetPolicy.single_full(max_evaluations=12, batch_size=4)

    assert policy.max_evaluations == 12
    assert policy.batch_size == 4
    assert policy.stage_names == ("full",)

    with pytest.raises(ConfigurationError, match="max_evaluations"):
        BudgetPolicy.single_full(budget=12, batch_size=4)


def test_scheduler_promotes_top_fraction_by_previous_rung_score() -> None:
    policy = BudgetPolicy(
        stages=[
            EvaluationStage("cheap", budget=0.10, promote_fraction=0.4, confidence="partial"),
            EvaluationStage("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full"),
        ],
        max_evaluations=10,
    )
    scheduler = BudgetScheduler(policy)
    candidates = [_candidate(index) for index in range(5)]
    for index, candidate in enumerate(candidates):
        candidate.apply_record(
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                score=float(index),
                confidence="partial",
                stage="cheap",
                cost=0.1,
            )
        )

    promoted = scheduler.promote(candidates, completed_stage="cheap")

    assert [candidate.candidate_id for candidate in promoted] == ["c-4", "c-3"]
    assert all(candidate.status == "promoted" for candidate in promoted)


def test_scheduler_promotes_by_completed_rung_score_not_best_observed_score() -> None:
    policy = BudgetPolicy(
        stages=[
            EvaluationStage("cheap", budget=0.10, promote_fraction=0.5, confidence="partial"),
            EvaluationStage("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full"),
        ],
        max_evaluations=10,
        exploration_fraction=0.0,
    )
    scheduler = BudgetScheduler(policy)
    candidates = [_candidate(0), _candidate(1)]
    candidates[0].apply_record(
        EvaluationRecord(
            candidate_id="c-0",
            score=1.0,
            confidence="partial",
            stage="cheap",
            cost=0.1,
        )
    )
    candidates[0].apply_record(
        EvaluationRecord(
            candidate_id="c-0",
            score=100.0,
            confidence="surrogate",
            stage="surrogate",
            cost=0.0,
        )
    )
    candidates[1].apply_record(
        EvaluationRecord(
            candidate_id="c-1",
            score=2.0,
            confidence="partial",
            stage="cheap",
            cost=0.1,
        )
    )

    promoted = scheduler.promote(candidates, completed_stage="cheap")

    assert [candidate.candidate_id for candidate in promoted] == ["c-1"]


def test_scheduler_exploration_fraction_adds_tail_candidates() -> None:
    policy = BudgetPolicy(
        stages=[
            EvaluationStage("cheap", budget=0.10, promote_fraction=0.25, confidence="partial"),
            EvaluationStage("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full"),
        ],
        max_evaluations=10,
        exploration_fraction=0.25,
    )
    scheduler = BudgetScheduler(policy)
    candidates = [_candidate(index) for index in range(8)]
    for index, candidate in enumerate(candidates):
        candidate.apply_record(
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                score=float(index),
                confidence="partial",
                stage="cheap",
                cost=0.1,
            )
        )

    promoted = scheduler.promote(candidates, completed_stage="cheap")

    assert [candidate.candidate_id for candidate in promoted] == ["c-7", "c-6", "c-0", "c-1"]


def test_scheduler_assigns_first_rung_to_new_candidates() -> None:
    policy = BudgetPolicy(
        stages=[
            EvaluationStage("cheap", budget=0.10, promote_fraction=0.5, confidence="partial"),
            EvaluationStage("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full"),
        ],
        max_evaluations=10,
    )
    scheduler = BudgetScheduler(policy)
    candidates = [_candidate(0), _candidate(1)]

    assigned = scheduler.assign_stage(candidates, stage_name="cheap")

    assert [candidate.stage for candidate in assigned] == ["cheap", "cheap"]
    assert [candidate.status for candidate in assigned] == ["racing", "racing"]


def test_scheduler_counts_eliminated_candidates() -> None:
    policy = BudgetPolicy(
        stages=[
            EvaluationStage("cheap", budget=0.10, promote_fraction=0.5, confidence="partial"),
            EvaluationStage("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full"),
        ],
        max_evaluations=10,
    )
    scheduler = BudgetScheduler(policy)
    candidates = [_candidate(index) for index in range(4)]
    for index, candidate in enumerate(candidates):
        candidate.apply_record(
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                score=float(index),
                confidence="partial",
                stage="cheap",
                cost=0.1,
            )
        )

    promoted = scheduler.promote(candidates, completed_stage="cheap")
    eliminated = [candidate for candidate in candidates if candidate.status == "eliminated"]

    assert len(promoted) == 2
    assert [candidate.candidate_id for candidate in eliminated] == ["c-0", "c-1"]


def test_scheduler_audit_fraction_promotes_one_low_ranked_candidate() -> None:
    policy = BudgetPolicy(
        stages=[
            EvaluationStage("cheap", budget=0.10, promote_fraction=0.25, confidence="partial"),
            EvaluationStage("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full"),
        ],
        max_evaluations=10,
        audit_fraction=0.25,
    )
    scheduler = BudgetScheduler(policy)
    candidates = [_candidate(index) for index in range(8)]
    for index, candidate in enumerate(candidates):
        candidate.apply_record(
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                score=float(index),
                confidence="partial",
                stage="cheap",
                cost=0.1,
            )
        )

    promoted = scheduler.promote(candidates, completed_stage="cheap")
    promoted_ids = {candidate.candidate_id for candidate in promoted}

    assert {"c-7", "c-6"}.issubset(promoted_ids)
    assert len(promoted) == 4
    # Top 2 by score (c-7, c-6) + 2 audit from remaining (c-5, c-4)
    assert {"c-5", "c-4"}.issubset(promoted_ids)
