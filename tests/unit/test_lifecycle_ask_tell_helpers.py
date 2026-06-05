from __future__ import annotations

import pytest

from evocore import FitnessError, Gene, GeneSpace
from evocore.lifecycle import (
    Candidate,
    CandidateBatch,
    EvaluationRecord,
    EvaluationStage,
    EventHistory,
    OptimizationTelemetry,
)
from evocore.lifecycle.ask_tell_helpers import (
    append_candidate_ask_events,
    append_candidate_tell_event,
    candidate_and_batch_for_record,
    evaluation_context_for_candidates,
    record_evaluation_telemetry,
    validate_evaluator_records,
)


def _space() -> GeneSpace:
    return GeneSpace([Gene("x", "float", -1.0, 1.0)])


def _candidate(candidate_id: str = "c-1", batch_id: str = "b-1") -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        genes=[0.25],
        batch_id=batch_id,
        origin="random",
        event_index=7,
        generation=2,
    )


def _record(candidate: Candidate, *, score: float | None = 1.0) -> EvaluationRecord:
    return EvaluationRecord(
        candidate_id=candidate.candidate_id,
        batch_id=candidate.batch_id,
        score=score,
        confidence="trusted_full" if score is not None else "rejected",
        stage="full",
        cost=1.5,
        metrics={"loss": 0.1} if score is not None else {},
        metadata={"source": "unit"},
    )


def test_append_candidate_ask_events_matches_optimizer_event_shape() -> None:
    events = EventHistory()
    candidate = _candidate()

    append_candidate_ask_events(events, [candidate], _space())

    rows = events.to_rows()
    assert len(rows) == 1
    assert rows[0]["event_index"] == 0
    assert rows[0]["event_type"] == "ask"
    assert rows[0]["batch_id"] == "b-1"
    assert rows[0]["candidate_id"] == "c-1"
    assert rows[0]["candidate_hash"] == _space().value_hash(candidate.genes)
    assert rows[0]["genes"] == [0.25]


def test_append_candidate_tell_event_merges_extra_metadata() -> None:
    events = EventHistory()
    candidate = _candidate()
    record = _record(candidate)
    candidate.apply_record(record)

    append_candidate_tell_event(
        events,
        candidate,
        record,
        _space(),
        "minimize",
        metadata={"accepted_for_state": True},
    )

    row = events.to_rows()[0]
    assert row["event_type"] == "tell"
    assert row["raw_score"] == pytest.approx(1.0)
    assert row["comparison_score"] == pytest.approx(-1.0)
    assert row["status"] == "trusted"
    assert row["metrics"] == {"loss": 0.1}
    assert row["metadata"] == {"source": "unit", "accepted_for_state": True}


def test_candidate_and_batch_for_record_rejects_unknown_ids() -> None:
    candidate = _candidate()
    batch = CandidateBatch(batch_id="b-1", candidate_ids=("c-1",))

    found_candidate, found_batch = candidate_and_batch_for_record(
        _record(candidate),
        {"c-1": candidate},
        {"b-1": batch},
    )

    assert found_candidate is candidate
    assert found_batch is batch

    with pytest.raises(FitnessError, match="unknown candidate_id"):
        candidate_and_batch_for_record(
            EvaluationRecord(
                candidate_id="missing",
                batch_id="b-1",
                score=1.0,
                confidence="trusted_full",
                stage="full",
            ),
            {"c-1": candidate},
            {"b-1": batch},
        )


def test_record_evaluation_telemetry_updates_counts_and_returns_label() -> None:
    telemetry = OptimizationTelemetry()
    candidate = _candidate()

    label = record_evaluation_telemetry(telemetry, _record(candidate))

    assert label == "trusted"
    assert telemetry.candidates_full_evaluated == 1
    assert telemetry.cost_by_stage["full"] == pytest.approx(1.5)


def test_evaluation_context_for_candidates_requires_one_batch() -> None:
    stage = EvaluationStage("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full")

    context = evaluation_context_for_candidates(
        [_candidate()],
        stage,
        direction="maximize",
        fallback_event_index=99,
        batch_error_message="Assigned candidates must belong to exactly one batch.",
    )

    assert context.batch_id == "b-1"
    assert context.event_index == 7
    assert context.budget == pytest.approx(1.0)

    with pytest.raises(FitnessError, match="exactly one batch"):
        evaluation_context_for_candidates(
            [_candidate("c-1", "b-1"), _candidate("c-2", "b-2")],
            stage,
            direction="maximize",
            fallback_event_index=99,
            batch_error_message="Assigned candidates must belong to exactly one batch.",
        )


def test_validate_evaluator_records_rejects_missing_unexpected_duplicate_and_batch_mismatch() -> (
    None
):
    assigned = [_candidate("c-1", "b-1"), _candidate("c-2", "b-1")]

    validate_evaluator_records(
        assigned,
        [_record(assigned[0]), _record(assigned[1])],
        batch_error_message="Assigned candidates must belong to exactly one batch.",
    )

    with pytest.raises(FitnessError, match="missing evaluation records"):
        validate_evaluator_records(
            assigned,
            [_record(assigned[0])],
            batch_error_message="Assigned candidates must belong to exactly one batch.",
        )

    with pytest.raises(FitnessError, match="unknown evaluation records"):
        validate_evaluator_records(
            assigned,
            [
                _record(assigned[0]),
                _record(assigned[1]),
                EvaluationRecord(
                    candidate_id="c-3",
                    batch_id="b-1",
                    score=1.0,
                    confidence="trusted_full",
                    stage="full",
                ),
            ],
            batch_error_message="Assigned candidates must belong to exactly one batch.",
        )

    with pytest.raises(FitnessError, match="duplicate evaluation records"):
        validate_evaluator_records(
            assigned,
            [_record(assigned[0]), _record(assigned[0]), _record(assigned[1])],
            batch_error_message="Assigned candidates must belong to exactly one batch.",
        )

    with pytest.raises(FitnessError, match="record batch_id"):
        validate_evaluator_records(
            assigned,
            [
                _record(assigned[0]),
                EvaluationRecord(
                    candidate_id="c-2",
                    batch_id="wrong",
                    score=1.0,
                    confidence="trusted_full",
                    stage="full",
                ),
            ],
            batch_error_message="Assigned candidates must belong to exactly one batch.",
        )
