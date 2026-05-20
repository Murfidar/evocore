import pytest

from evocore import CheckpointError
from evocore.lifecycle import (
    Candidate,
    CandidateBatch,
    EvaluationRecord,
    EventHistory,
    EventRecord,
    OptimizationTelemetry,
    batch_from_checkpoint,
    batch_to_checkpoint,
    candidate_from_checkpoint,
    candidate_to_checkpoint,
    event_history_from_checkpoint,
    event_history_to_checkpoint,
    telemetry_from_checkpoint,
    telemetry_to_checkpoint,
)


def _record(candidate_id: str, *, batch_id: str, stage: str = "full") -> EvaluationRecord:
    return EvaluationRecord(
        candidate_id=candidate_id,
        batch_id=batch_id,
        score=1.5,
        confidence="trusted_full",
        stage=stage,
        cost=2.0,
        metrics={"loss": 0.25},
        metadata={"worker": "a"},
    )


def test_candidate_checkpoint_round_trip_preserves_scores_and_metadata() -> None:
    candidate = Candidate(
        candidate_id="c-1",
        genes=[0.25, 2, True],
        batch_id="b-1",
        params={"x": 0.25, "n": 2, "enabled": True},
        origin="random",
        parents=("p-1",),
        event_index=3,
        generation=7,
        metadata={"source": "unit"},
    )
    candidate.apply_record(_record("c-1", batch_id="b-1"))

    payload = candidate_to_checkpoint(candidate)
    restored = candidate_from_checkpoint(payload)

    assert restored.candidate_id == candidate.candidate_id
    assert restored.genes == candidate.genes
    assert restored.params == candidate.params
    assert restored.batch_id == candidate.batch_id
    assert restored.parents == candidate.parents
    assert restored.event_index == candidate.event_index
    assert restored.generation == candidate.generation
    assert restored.status == "trusted"
    assert restored.confidence == "trusted_full"
    assert restored.cost == pytest.approx(2.0)
    assert restored.scores["full"].score == pytest.approx(1.5)
    assert restored.scores["full"].metrics == {"loss": 0.25}
    assert restored.metadata["source"] == "unit"


def test_batch_checkpoint_round_trip_preserves_candidate_order_records_and_consumed() -> None:
    batch = CandidateBatch(
        batch_id="b-1",
        candidate_ids=("c-1", "c-2"),
        continuous_samples_by_id={"c-1": [0.1], "c-2": [0.2]},
    )
    batch.accept_record(_record("c-1", batch_id="b-1"))
    batch.consumed = True

    payload = batch_to_checkpoint(batch)
    restored = batch_from_checkpoint(payload)

    assert restored.batch_id == "b-1"
    assert restored.candidate_ids == ("c-1", "c-2")
    assert restored.consumed is True
    assert restored.continuous_samples_by_id == {"c-1": [0.1], "c-2": [0.2]}
    assert list(restored.records_by_key) == [("c-1", "full")]
    assert restored.records_by_key[("c-1", "full")].score == pytest.approx(1.5)


def test_batch_checkpoint_rejects_record_for_candidate_outside_batch() -> None:
    payload = {
        "batch_id": "b-1",
        "candidate_ids": ["c-1"],
        "records": [
            {
                "candidate_id": "c-2",
                "batch_id": "b-1",
                "score": 1.0,
                "confidence": "trusted_full",
                "stage": "full",
                "cost": 0.0,
                "metrics": {},
                "metadata": {},
            }
        ],
        "consumed": False,
        "continuous_samples_by_id": {},
    }

    with pytest.raises(CheckpointError, match="does not belong to batch"):
        batch_from_checkpoint(payload)


def test_telemetry_checkpoint_round_trip_restores_unique_hash_set() -> None:
    telemetry = OptimizationTelemetry()
    telemetry.record_proposed(3)
    telemetry.unique_candidate_hashes.update({"h2", "h1"})
    telemetry.record_full(2, stage="full", cost=5.0)
    telemetry.record_cached(1, stage="cache", cost=0.5)

    restored = telemetry_from_checkpoint(telemetry_to_checkpoint(telemetry))

    assert restored.total_candidates_proposed == 3
    assert restored.unique_candidate_hashes == {"h1", "h2"}
    assert restored.candidates_full_evaluated == 2
    assert restored.candidates_cached == 1
    assert restored.cost_by_stage == {"cache": 0.5, "full": 5.0}


def test_event_history_checkpoint_round_trip_preserves_append_order() -> None:
    history = EventHistory()
    history.append(EventRecord(event_index=0, event_type="ask", batch_id="b-1"))
    history.append(
        EventRecord(
            event_index=1,
            event_type="tell",
            batch_id="b-1",
            candidate_id="c-1",
            raw_score=1.5,
            comparison_score=1.5,
            status="trusted",
        )
    )

    restored = event_history_from_checkpoint(event_history_to_checkpoint(history))

    assert restored.to_rows() == history.to_rows()
