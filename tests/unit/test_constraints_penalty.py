from evocore import Candidate, EvaluationRecord, Gene, GeneSpace
from evocore.lifecycle import (
    STATE_UPDATE_CONFIDENCES,
    TRUSTED_CONFIDENCES,
    constraint_penalty_record,
    is_state_update_confidence,
    is_trusted_confidence,
)
from evocore.search_space import ActiveGeneProjection, ConstraintViolation


def test_constraint_penalty_is_state_update_but_not_trusted() -> None:
    assert "constraint_penalty" in STATE_UPDATE_CONFIDENCES
    assert "constraint_penalty" not in TRUSTED_CONFIDENCES
    assert is_state_update_confidence("constraint_penalty")
    assert not is_trusted_confidence("constraint_penalty")


def test_constraint_penalty_record_has_finite_score_zero_cost_and_metadata() -> None:
    candidate = Candidate("c-1", [2.0], batch_id="b-1")
    record = constraint_penalty_record(
        candidate=candidate,
        stage="projection",
        direction="maximize",
        violations=[ConstraintViolation(code="bad", message="invalid", names=("x",))],
    )

    assert isinstance(record, EvaluationRecord)
    assert record.confidence == "constraint_penalty"
    assert record.score is not None
    assert record.score < 0.0
    assert record.cost == 0.0
    assert record.metadata["constraint_violations"][0]["code"] == "bad"


def test_penalty_candidate_status_is_eliminated() -> None:
    candidate = Candidate("c-1", [2.0], batch_id="b-1")
    candidate.apply_record(
        EvaluationRecord(
            candidate_id="c-1",
            batch_id="b-1",
            score=-1.0e300,
            confidence="constraint_penalty",
            stage="projection",
        )
    )

    assert candidate.status == "eliminated"
    assert candidate.best_state_score("maximize") == -1.0e300


def test_constraint_validator_metadata_round_trips_through_projection() -> None:
    def validate(params):
        if params["fast"] >= params["slow"]:
            return [
                ConstraintViolation(
                    code="ordering",
                    message="fast must be below slow",
                    names=("fast", "slow"),
                )
            ]
        return []

    projection = ActiveGeneProjection(
        source_space=GeneSpace(
            [
                Gene("fast", "float", 1.0, 20.0),
                Gene("slow", "float", 1.0, 40.0),
            ]
        ),
        active_names=["fast", "slow"],
        validators=[validate],
        schema_id="constraints",
        schema_version="1",
    )

    result = projection.reconstruct([10.0, 5.0])

    assert result.valid is False
    assert result.violations[0].code == "ordering"
