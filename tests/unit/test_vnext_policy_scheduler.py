import pytest
from evocore.scheduler import EvaluationScheduler

from evocore.evaluation import Candidate, EvaluationRecord, Rung
from evocore.exceptions import ConfigurationError
from evocore.policies import MultiFidelityPolicy


def _candidate(index: int) -> Candidate:
    return Candidate(
        candidate_id=f"c-{index}", genes=[float(index)], origin="random", event_index=0
    )


def test_policy_requires_unique_rung_names_and_full_budget() -> None:
    policy = MultiFidelityPolicy(
        rungs=[
            Rung("cheap", budget=0.10, promote_fraction=0.5, confidence="partial"),
            Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full"),
        ],
        full_evaluation_budget=32,
        batch_size=8,
        exploration_fraction=0.10,
        audit_fraction=0.05,
    )

    assert policy.rung_names == ("cheap", "full")
    assert policy.final_rung.name == "full"


def test_policy_rejects_duplicate_rung_names() -> None:
    with pytest.raises(ConfigurationError, match="duplicate rung"):
        MultiFidelityPolicy(
            rungs=[
                Rung("cheap", budget=0.10, promote_fraction=0.5, confidence="partial"),
                Rung("cheap", budget=1.0, promote_fraction=1.0, confidence="trusted_full"),
            ],
            full_evaluation_budget=16,
        )


def test_policy_rejects_missing_trusted_full_rung() -> None:
    with pytest.raises(ConfigurationError, match="trusted_full"):
        MultiFidelityPolicy(
            rungs=[Rung("cheap", budget=0.10, promote_fraction=0.5, confidence="partial")],
            full_evaluation_budget=16,
        )


def test_policy_rejects_invalid_budget_and_fractions() -> None:
    with pytest.raises(ConfigurationError, match="full_evaluation_budget"):
        MultiFidelityPolicy(
            rungs=[Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full")],
            full_evaluation_budget=0,
        )

    with pytest.raises(ConfigurationError, match="exploration_fraction"):
        MultiFidelityPolicy(
            rungs=[Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full")],
            full_evaluation_budget=1,
            exploration_fraction=1.5,
        )


def test_scheduler_promotes_top_fraction_by_previous_rung_score() -> None:
    policy = MultiFidelityPolicy(
        rungs=[
            Rung("cheap", budget=0.10, promote_fraction=0.4, confidence="partial"),
            Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full"),
        ],
        full_evaluation_budget=10,
    )
    scheduler = EvaluationScheduler(policy)
    candidates = [_candidate(index) for index in range(5)]
    for index, candidate in enumerate(candidates):
        candidate.apply_record(
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                score=float(index),
                confidence="partial",
                rung="cheap",
                cost=0.1,
            )
        )

    promoted = scheduler.promote(candidates, completed_rung="cheap")

    assert [candidate.candidate_id for candidate in promoted] == ["c-4", "c-3"]
    assert all(candidate.status == "promoted" for candidate in promoted)


def test_scheduler_assigns_first_rung_to_new_candidates() -> None:
    policy = MultiFidelityPolicy(
        rungs=[
            Rung("cheap", budget=0.10, promote_fraction=0.5, confidence="partial"),
            Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full"),
        ],
        full_evaluation_budget=10,
    )
    scheduler = EvaluationScheduler(policy)
    candidates = [_candidate(0), _candidate(1)]

    assigned = scheduler.assign_rung(candidates, rung_name="cheap")

    assert [candidate.rung for candidate in assigned] == ["cheap", "cheap"]
    assert [candidate.status for candidate in assigned] == ["racing", "racing"]


def test_scheduler_counts_eliminated_candidates() -> None:
    policy = MultiFidelityPolicy(
        rungs=[
            Rung("cheap", budget=0.10, promote_fraction=0.5, confidence="partial"),
            Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full"),
        ],
        full_evaluation_budget=10,
    )
    scheduler = EvaluationScheduler(policy)
    candidates = [_candidate(index) for index in range(4)]
    for index, candidate in enumerate(candidates):
        candidate.apply_record(
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                score=float(index),
                confidence="partial",
                rung="cheap",
                cost=0.1,
            )
        )

    promoted = scheduler.promote(candidates, completed_rung="cheap")
    eliminated = [candidate for candidate in candidates if candidate.status == "eliminated"]

    assert len(promoted) == 2
    assert [candidate.candidate_id for candidate in eliminated] == ["c-0", "c-1"]


def test_scheduler_audit_fraction_promotes_one_low_ranked_candidate() -> None:
    policy = MultiFidelityPolicy(
        rungs=[
            Rung("cheap", budget=0.10, promote_fraction=0.25, confidence="partial"),
            Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full"),
        ],
        full_evaluation_budget=10,
        audit_fraction=0.25,
    )
    scheduler = EvaluationScheduler(policy)
    candidates = [_candidate(index) for index in range(8)]
    for index, candidate in enumerate(candidates):
        candidate.apply_record(
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                score=float(index),
                confidence="partial",
                rung="cheap",
                cost=0.1,
            )
        )

    promoted = scheduler.promote(candidates, completed_rung="cheap")
    promoted_ids = {candidate.candidate_id for candidate in promoted}

    assert {"c-7", "c-6"}.issubset(promoted_ids)
    assert len(promoted) == 4
    # Top 2 by score (c-7, c-6) + 2 audit from remaining (c-5, c-4)
    assert {"c-5", "c-4"}.issubset(promoted_ids)

