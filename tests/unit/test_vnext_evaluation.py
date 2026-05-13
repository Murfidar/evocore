import pytest

from evocore.evaluation import (
    Candidate,
    EvaluationRecord,
    OptimizationTelemetry,
    Rung,
)
from evocore.exceptions import ConfigurationError, FitnessError


def test_rung_requires_valid_budget_and_promotion_fraction() -> None:
    Rung("cheap", budget=0.25, promote_fraction=0.5, confidence="partial")

    with pytest.raises(ConfigurationError, match="rung name"):
        Rung("", budget=0.25, promote_fraction=0.5, confidence="partial")

    with pytest.raises(ConfigurationError, match="budget"):
        Rung("bad_budget", budget=0.0, promote_fraction=0.5, confidence="partial")

    with pytest.raises(ConfigurationError, match="promote_fraction"):
        Rung("bad_fraction", budget=0.25, promote_fraction=1.5, confidence="partial")


def test_candidate_applies_trusted_record_and_tracks_score() -> None:
    candidate = Candidate(
        candidate_id="c-1",
        genes=[1.0, 2],
        params={"x": 1.0, "mode": 2},
        origin="random",
        event_index=0,
    )
    record = EvaluationRecord(
        candidate_id="c-1",
        score=0.75,
        confidence="trusted_full",
        rung="full_snapshot",
        cost=1.0,
        metrics={"trade_count": 12},
    )

    candidate.apply_record(record)

    assert candidate.status == "trusted"
    assert candidate.confidence == "trusted_full"
    assert candidate.rung == "full_snapshot"
    assert candidate.cost == 1.0
    assert candidate.scores["full_snapshot"].score == 0.75
    assert candidate.metadata["metrics"]["trade_count"] == 12


def test_candidate_rejects_record_for_different_candidate() -> None:
    candidate = Candidate(candidate_id="left", genes=[1.0], origin="random", event_index=0)
    record = EvaluationRecord(
        candidate_id="right",
        score=1.0,
        confidence="trusted_full",
        rung="full",
        cost=1.0,
    )

    with pytest.raises(FitnessError, match="does not match candidate"):
        candidate.apply_record(record)


def test_candidate_and_record_expose_batch_ids() -> None:
    candidate = Candidate(
        candidate_id="c-1",
        batch_id="b-1",
        genes=[1.0],
        origin="random",
        event_index=0,
    )
    record = EvaluationRecord(
        candidate_id="c-1",
        batch_id="b-1",
        score=1.0,
        confidence="trusted_full",
        rung="full",
        cost=1.0,
    )

    candidate.apply_record(record)

    assert candidate.batch_id == "b-1"
    assert record.batch_id == "b-1"
    assert candidate.status == "trusted"


def test_candidate_rejects_record_for_different_batch() -> None:
    candidate = Candidate(
        candidate_id="c-1",
        batch_id="b-left",
        genes=[1.0],
        origin="random",
        event_index=0,
    )
    record = EvaluationRecord(
        candidate_id="c-1",
        batch_id="b-right",
        score=1.0,
        confidence="trusted_full",
        rung="full",
        cost=1.0,
    )

    with pytest.raises(FitnessError, match="batch_id"):
        candidate.apply_record(record)


def test_surrogate_score_does_not_mark_candidate_trusted() -> None:
    candidate = Candidate(candidate_id="c-2", genes=[0.0], origin="random", event_index=0)
    candidate.apply_record(
        EvaluationRecord(
            candidate_id="c-2",
            score=0.2,
            confidence="surrogate",
            rung="surrogate",
            cost=0.0,
            metrics={"model": "baseline"},
        )
    )

    assert candidate.status == "screened"
    assert candidate.confidence == "surrogate"
    assert "surrogate" in candidate.scores


def test_rejected_record_can_omit_score() -> None:
    record = EvaluationRecord(
        candidate_id="bad",
        score=None,
        confidence="rejected",
        rung="cheap",
        cost=0.0,
        metrics={"reason": "no_signals"},
    )

    assert record.score is None


def test_non_rejected_record_requires_finite_score() -> None:
    with pytest.raises(FitnessError, match="finite score"):
        EvaluationRecord(
            candidate_id="nan",
            score=float("nan"),
            confidence="partial",
            rung="cheap",
            cost=0.1,
        )


def test_telemetry_records_counts_and_costs() -> None:
    telemetry = OptimizationTelemetry()
    telemetry.record_proposed(5)
    telemetry.record_screened(2)
    telemetry.record_partial(3, rung="cheap", cost=0.6)
    telemetry.record_full(1, rung="full", cost=1.0)
    telemetry.record_promoted(2, rung="cheap")
    telemetry.record_eliminated(1, rung="cheap")

    assert telemetry.total_candidates_proposed == 5
    assert telemetry.candidates_screened == 2
    assert telemetry.candidates_partial_evaluated == 3
    assert telemetry.candidates_full_evaluated == 1
    assert telemetry.promoted_by_rung["cheap"] == 2
    assert telemetry.eliminated_by_rung["cheap"] == 1
    assert telemetry.cost_by_rung["cheap"] == pytest.approx(0.6)
    assert telemetry.cost_by_rung["full"] == pytest.approx(1.0)
