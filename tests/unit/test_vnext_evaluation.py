import pytest

from evocore.core.errors import ConfigurationError, FitnessError
from evocore.lifecycle import (
    Candidate,
    EvaluationContext,
    EvaluationRecord,
    EvaluationStage,
    OptimizationTelemetry,
    OptimizerStateSummary,
    UpdateResult,
    score_for_direction,
)


def test_rung_requires_valid_budget_and_promotion_fraction() -> None:
    EvaluationStage("cheap", budget=0.25, promote_fraction=0.5, confidence="partial")

    with pytest.raises(ConfigurationError, match="EvaluationStage name"):
        EvaluationStage("", budget=0.25, promote_fraction=0.5, confidence="partial")

    with pytest.raises(ConfigurationError, match="budget"):
        EvaluationStage("bad_budget", budget=0.0, promote_fraction=0.5, confidence="partial")

    with pytest.raises(ConfigurationError, match="promote_fraction"):
        EvaluationStage("bad_fraction", budget=0.25, promote_fraction=1.5, confidence="partial")


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
        stage="full_snapshot",
        cost=1.0,
        metrics={"trade_count": 12},
    )

    candidate.apply_record(record)

    assert candidate.status == "trusted"
    assert candidate.confidence == "trusted_full"
    assert candidate.stage == "full_snapshot"
    assert candidate.cost == 1.0
    assert candidate.scores["full_snapshot"].score == 0.75
    assert candidate.metadata["metrics"]["trade_count"] == 12


def test_candidate_rejects_record_for_different_candidate() -> None:
    candidate = Candidate(candidate_id="left", genes=[1.0], origin="random", event_index=0)
    record = EvaluationRecord(
        candidate_id="right",
        score=1.0,
        confidence="trusted_full",
        stage="full",
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
        stage="full",
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
        stage="full",
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
            stage="surrogate",
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
        stage="cheap",
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
            stage="cheap",
            cost=0.1,
        )


def test_telemetry_records_counts_and_costs() -> None:
    telemetry = OptimizationTelemetry()
    telemetry.record_proposed(5)
    telemetry.record_screened(2)
    telemetry.record_partial(3, stage="cheap", cost=0.6)
    telemetry.record_full(1, stage="full", cost=1.0)
    telemetry.record_promoted(2, stage="cheap")
    telemetry.record_eliminated(1, stage="cheap")

    assert telemetry.total_candidates_proposed == 5
    assert telemetry.candidates_screened == 2
    assert telemetry.candidates_partial_evaluated == 3
    assert telemetry.candidates_full_evaluated == 1
    assert telemetry.promoted_by_stage["cheap"] == 2
    assert telemetry.eliminated_by_stage["cheap"] == 1
    assert telemetry.cost_by_stage["cheap"] == pytest.approx(0.6)
    assert telemetry.cost_by_stage["full"] == pytest.approx(1.0)


def test_telemetry_records_unique_candidate_hashes_for_proposals() -> None:
    telemetry = OptimizationTelemetry()
    candidates = [
        Candidate(candidate_id="c-1", genes=[1.0, 2, True], origin="random", event_index=0),
        Candidate(candidate_id="c-2", genes=[1.0, 2, True], origin="random", event_index=0),
        Candidate(candidate_id="c-3", genes=[1.0, 3, True], origin="random", event_index=0),
    ]

    telemetry.record_proposed_candidates(candidates)

    assert telemetry.total_candidates_proposed == 3
    assert len(telemetry.unique_candidate_hashes) == 2


def test_telemetry_to_dict_exports_sorted_hashes_and_unique_count() -> None:
    telemetry = OptimizationTelemetry()
    telemetry.total_candidates_proposed = 3
    telemetry.unique_candidate_hashes.update({"hash-b", "hash-a"})
    telemetry.candidates_screened = 1
    telemetry.candidates_partial_evaluated = 2
    telemetry.candidates_full_evaluated = 3
    telemetry.candidates_cached = 1
    telemetry.promoted_by_stage = {"cheap": 2}
    telemetry.eliminated_by_stage = {"cheap": 1}
    telemetry.cost_by_stage = {"full": 2.0, "cheap": 0.5}

    assert telemetry.to_dict() == {
        "total_candidates_proposed": 3,
        "unique_candidate_hashes": ["hash-a", "hash-b"],
        "unique_candidate_count": 2,
        "candidates_screened": 1,
        "candidates_partial_evaluated": 2,
        "candidates_full_evaluated": 3,
        "candidates_cached": 1,
        "promoted_by_stage": {"cheap": 2},
        "eliminated_by_stage": {"cheap": 1},
        "cost_by_stage": {"cheap": 0.5, "full": 2.0},
    }


def test_telemetry_to_json_is_deterministic() -> None:
    telemetry = OptimizationTelemetry()
    telemetry.unique_candidate_hashes.update({"z", "a"})
    telemetry.cost_by_stage = {"full": 1.0, "cheap": 0.25}

    first = telemetry.to_json()
    second = telemetry.to_json()

    assert first == second
    assert '"unique_candidate_hashes": ["a", "z"]' in first


def test_evaluation_record_preserves_metadata() -> None:
    record = EvaluationRecord(
        candidate_id="c-1",
        score=1.25,
        confidence="trusted_full",
        stage="full",
        cost=1.0,
        metrics={"loss": 0.2},
        metadata={"source": "unit"},
        batch_id="b-1",
    )

    assert record.metadata["source"] == "unit"
    assert record.metrics["loss"] == pytest.approx(0.2)


def test_evaluation_record_preserves_positional_batch_id() -> None:
    record = EvaluationRecord(
        "c-1",
        1.25,
        "trusted_full",
        "full",
        1.0,
        {"loss": 0.2},
        "b-1",
    )

    assert record.metrics["loss"] == pytest.approx(0.2)
    assert record.batch_id == "b-1"
    assert record.metadata == {}


def test_evaluation_context_carries_batch_rung_direction_and_budget() -> None:
    stage = EvaluationStage("cheap", budget=0.25, promote_fraction=0.5, confidence="partial")

    context = EvaluationContext(
        stage=stage,
        batch_id="b-1",
        event_index=3,
        direction="minimize",
        budget=0.25,
        metadata={"phase": "screen"},
    )

    assert context.stage is stage
    assert context.batch_id == "b-1"
    assert context.event_index == 3
    assert context.direction == "minimize"
    assert context.budget == pytest.approx(0.25)
    assert context.metadata["phase"] == "screen"


def test_tell_result_and_state_summary_have_stable_fields() -> None:
    telemetry = OptimizationTelemetry()
    tell_result = UpdateResult(
        accepted_count=3,
        trusted_count=1,
        partial_count=1,
        surrogate_count=0,
        cached_count=0,
        rejected_count=1,
        best_candidate_id="c-2",
        best_score=2.5,
        consumed_batch_ids=("b-1",),
        pending_batch_ids=("b-2",),
        telemetry=telemetry,
    )
    state = OptimizerStateSummary(
        best_candidate_id="c-2",
        best_score=2.5,
        event_index=4,
        pending_batch_ids=("b-2",),
        trusted_count=5,
        telemetry=telemetry,
    )

    assert tell_result.accepted_count == 3
    assert tell_result.cached_count == 0
    assert tell_result.consumed_batch_ids == ("b-1",)
    assert state.best_candidate_id == "c-2"
    assert state.pending_batch_ids == ("b-2",)


def test_candidate_best_observed_score_honors_direction() -> None:
    candidate = Candidate(candidate_id="c-1", genes=[1.0], batch_id="b-1")
    candidate.apply_record(
        EvaluationRecord(
            candidate_id="c-1",
            score=10.0,
            confidence="partial",
            stage="cheap",
            cost=0.1,
        )
    )
    candidate.apply_record(
        EvaluationRecord(
            candidate_id="c-1",
            score=2.0,
            confidence="trusted_full",
            stage="full",
            cost=1.0,
        )
    )

    assert candidate.best_observed_score("maximize") == pytest.approx(10.0)
    assert candidate.best_observed_score("minimize") == pytest.approx(2.0)
    assert candidate.comparison_score("maximize") == pytest.approx(10.0)
    assert candidate.comparison_score("minimize") == pytest.approx(-2.0)


def test_score_for_direction_rejects_invalid_direction() -> None:
    assert score_for_direction(3.0, "maximize") == pytest.approx(3.0)
    assert score_for_direction(3.0, "minimize") == pytest.approx(-3.0)

    with pytest.raises(ConfigurationError, match="direction"):
        score_for_direction(3.0, "lowest")  # type: ignore[arg-type]


def test_rejected_record_rejects_score() -> None:
    with pytest.raises(FitnessError, match="rejected"):
        EvaluationRecord(
            candidate_id="bad",
            score=0.0,
            confidence="rejected",
            stage="full",
            cost=0.0,
            metadata={"reason": "constraint_violation"},
        )


def test_telemetry_records_cached_without_full_evaluation_count() -> None:
    telemetry = OptimizationTelemetry()

    telemetry.record_cached(2, stage="full", cost=0.0)

    assert telemetry.candidates_cached == 2
    assert telemetry.candidates_full_evaluated == 0
    assert telemetry.to_dict()["candidates_cached"] == 2
