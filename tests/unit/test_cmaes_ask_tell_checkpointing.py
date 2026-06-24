import pytest

from evocore import (
    CheckpointError,
    CMAESOptimizer,
    EvaluationRecord,
    FitnessError,
    Gene,
    GeneSpace,
)


def _space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("period", "int", 2, 20),
        ]
    )


def _optimizer(**overrides) -> CMAESOptimizer:
    params = {
        "population_size": 4,
        "max_generations": 5,
        "seed": 7,
    }
    params.update(overrides)
    return CMAESOptimizer(_space(), **params)


def _score(candidate) -> float:
    return -sum(float(value) ** 2 for value in candidate.genes)


def _trusted_records(candidates) -> list[EvaluationRecord]:
    return [
        EvaluationRecord(
            candidate_id=candidate.candidate_id,
            batch_id=candidate.batch_id,
            score=_score(candidate),
            confidence="trusted_full",
            stage="full",
            cost=1.0,
        )
        for candidate in candidates
    ]


def _partial_records(candidates) -> list[EvaluationRecord]:
    return [
        EvaluationRecord(
            candidate_id=candidate.candidate_id,
            batch_id=candidate.batch_id,
            score=_score(candidate),
            confidence="partial",
            stage="cheap",
            cost=0.1,
        )
        for candidate in candidates
    ]


def test_cma_ask_tell_checkpoint_after_ask_contains_pending_batch_and_rust_state() -> None:
    optimizer = _optimizer()
    candidates = optimizer.ask()

    snapshot = optimizer.ask_tell_checkpoint(metadata={"reason": "unit"})
    payload = snapshot.to_dict()
    state_payload = payload["state"]["payload"]

    assert payload["optimizer"]["optimizer_type"] == "CMAESOptimizer"
    assert payload["position"]["mode"] == "ask_tell"
    assert payload["position"]["event_index"] == 1
    assert payload["position"]["pending_batch_ids"] == [candidates[0].batch_id]
    assert payload["metadata"] == {"reason": "unit"}
    assert state_payload["state_kind"] == "cmaes_ask_tell"
    assert state_payload["event_index"] == 1
    assert state_payload["best_candidate_id"] is None
    assert state_payload["cmaes_state"]["schema_version"] == 1
    assert state_payload["cmaes_state"]["optimizer_type"] == "cmaes"
    assert state_payload["cmaes_state"]["state"]["generation"] == 0
    assert set(state_payload["candidates_by_id"]) == {
        candidate.candidate_id for candidate in candidates
    }
    assert list(state_payload["batches_by_id"]) == [candidates[0].batch_id]


def test_cma_resume_ask_tell_checkpoint_after_ask_accepts_pending_records(tmp_path) -> None:
    source = _optimizer()
    candidates = source.ask()
    checkpoint_path = tmp_path / "cmaes-ask-tell.evocore-checkpoint.json"
    source.save_checkpoint(checkpoint_path, source.ask_tell_checkpoint())

    restored = _optimizer()
    summary = restored.resume_ask_tell_checkpoint(checkpoint_path)
    result = restored.tell(_trusted_records(candidates))

    assert summary.pending_batch_ids == (candidates[0].batch_id,)
    assert result.trusted_count == 4
    assert result.consumed_batch_ids == (candidates[0].batch_id,)
    assert result.pending_batch_ids == ()
    assert restored.generation == 1
    assert restored.state_summary().trusted_count == 4
    assert restored.best_candidate is not None


def test_cma_resume_ask_tell_checkpoint_after_partial_tell_accepts_missing_records(
    tmp_path,
) -> None:
    source = _optimizer()
    candidates = source.ask()
    records = _trusted_records(candidates)
    source.tell(records[:2])
    checkpoint_path = tmp_path / "cmaes-partial.evocore-checkpoint.json"
    source.save_checkpoint(checkpoint_path, source.ask_tell_checkpoint())

    restored = _optimizer()
    summary = restored.resume_ask_tell_checkpoint(checkpoint_path)
    result = restored.tell(records[2:])

    assert summary.best_candidate_id == candidates[0].candidate_id
    assert summary.pending_batch_ids == (candidates[0].batch_id,)
    assert result.accepted_count == 2
    assert result.consumed_batch_ids == (candidates[0].batch_id,)
    assert result.pending_batch_ids == ()
    assert restored.generation == 1
    assert restored.state_summary().trusted_count == 4


def test_cma_resume_ask_tell_checkpoint_next_ask_matches_uninterrupted(tmp_path) -> None:
    source = _optimizer()
    first_batch = source.ask()
    source.tell(_trusted_records(first_batch))
    checkpoint_path = tmp_path / "cmaes-complete.evocore-checkpoint.json"
    source.save_checkpoint(checkpoint_path, source.ask_tell_checkpoint())

    restored = _optimizer()
    restored.resume_ask_tell_checkpoint(checkpoint_path)
    restored_next = restored.ask()
    source_next = source.ask()

    assert [candidate.candidate_id for candidate in restored_next] == [
        candidate.candidate_id for candidate in source_next
    ]
    assert [candidate.batch_id for candidate in restored_next] == [
        candidate.batch_id for candidate in source_next
    ]
    assert [candidate.genes for candidate in restored_next] == [
        candidate.genes for candidate in source_next
    ]

    with pytest.raises(FitnessError, match="consumed"):
        restored.tell([_trusted_records(first_batch)[0]])


def test_margin_cma_resume_next_ask_matches_uninterrupted(tmp_path) -> None:
    space = GeneSpace([Gene("x", "int", 0, 3), Gene("y", "float", -1.0, 1.0)])
    uninterrupted = CMAESOptimizer(
        space,
        population_size=4,
        seed=12,
        integer_strategy="margin",
    )
    restored = CMAESOptimizer(
        space,
        population_size=4,
        seed=12,
        integer_strategy="margin",
    )

    batch = uninterrupted.ask()
    uninterrupted.tell(
        [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=float(index),
                confidence="trusted_full",
                stage="full",
            )
            for index, candidate in enumerate(batch)
        ]
    )
    checkpoint = uninterrupted.ask_tell_checkpoint()
    restored.resume_ask_tell_checkpoint(checkpoint.to_dict())

    assert [candidate.genes for candidate in restored.ask()] == [
        candidate.genes for candidate in uninterrupted.ask()
    ]


def test_cma_resume_partial_confidence_records_keeps_batch_pending(tmp_path) -> None:
    source = _optimizer()
    candidates = source.ask()
    partial_result = source.tell(_partial_records(candidates))
    checkpoint_path = tmp_path / "cmaes-partial-confidence.evocore-checkpoint.json"
    source.save_checkpoint(checkpoint_path, source.ask_tell_checkpoint())

    restored = _optimizer()
    summary = restored.resume_ask_tell_checkpoint(checkpoint_path)
    result = restored.tell(_trusted_records(candidates))

    assert partial_result.consumed_batch_ids == ()
    assert partial_result.pending_batch_ids == (candidates[0].batch_id,)
    assert summary.pending_batch_ids == (candidates[0].batch_id,)
    assert result.consumed_batch_ids == (candidates[0].batch_id,)
    assert result.pending_batch_ids == ()


def test_cma_resume_ask_tell_checkpoint_rejects_config_mismatch(tmp_path) -> None:
    source = _optimizer()
    source.ask()
    checkpoint_path = tmp_path / "cmaes-config.evocore-checkpoint.json"
    source.save_checkpoint(checkpoint_path, source.ask_tell_checkpoint())

    mismatched = _optimizer(population_size=6)

    with pytest.raises(CheckpointError, match="optimizer_config_hash"):
        mismatched.resume_ask_tell_checkpoint(checkpoint_path)


def test_cma_resume_ask_tell_checkpoint_rejects_seed_and_direction_mismatch(tmp_path) -> None:
    source = _optimizer()
    source.ask()
    checkpoint_path = tmp_path / "cmaes-identity.evocore-checkpoint.json"
    source.save_checkpoint(checkpoint_path, source.ask_tell_checkpoint())

    with pytest.raises(CheckpointError, match="seed"):
        _optimizer(seed=999).resume_ask_tell_checkpoint(checkpoint_path)

    with pytest.raises(CheckpointError, match="direction"):
        _optimizer(direction="minimize").resume_ask_tell_checkpoint(checkpoint_path)


def test_cma_resume_ask_tell_checkpoint_rejects_wrong_state_kind() -> None:
    source = _optimizer()
    source.ask()
    payload = source.ask_tell_checkpoint().to_dict()
    payload["state"]["payload"]["state_kind"] = "ga_ask_tell"

    with pytest.raises(CheckpointError, match="CMA-ES ask/tell resume"):
        source.resume_ask_tell_checkpoint(payload)


def test_cma_resume_ask_tell_checkpoint_rejects_unknown_best_candidate() -> None:
    source = _optimizer()
    candidates = source.ask()
    source.tell([_trusted_records(candidates)[0]])
    payload = source.ask_tell_checkpoint().to_dict()
    payload["state"]["payload"]["best_candidate_id"] = "c-missing"

    with pytest.raises(CheckpointError, match="best_candidate_id"):
        source.resume_ask_tell_checkpoint(payload)


def test_cma_resume_ask_tell_checkpoint_rejects_batch_unknown_candidate_reference() -> None:
    source = _optimizer()
    candidates = source.ask()
    payload = source.ask_tell_checkpoint().to_dict()
    batch_payload = payload["state"]["payload"]["batches_by_id"][candidates[0].batch_id]
    batch_payload["candidate_ids"].append("c-missing")

    with pytest.raises(CheckpointError, match="references unknown candidate_id"):
        source.resume_ask_tell_checkpoint(payload)


def test_cma_resume_ask_tell_checkpoint_rejects_missing_cmaes_state() -> None:
    source = _optimizer()
    source.ask()
    payload = source.ask_tell_checkpoint().to_dict()
    del payload["state"]["payload"]["cmaes_state"]

    with pytest.raises(CheckpointError, match="cmaes_state"):
        source.resume_ask_tell_checkpoint(payload)


def test_cma_resume_ask_tell_checkpoint_rejects_malformed_cmaes_state() -> None:
    source = _optimizer()
    source.ask()
    payload = source.ask_tell_checkpoint().to_dict()
    payload["state"]["payload"]["cmaes_state"]["schema_version"] = 999

    with pytest.raises(CheckpointError, match="cmaes_state"):
        source.resume_ask_tell_checkpoint(payload)


def test_cma_resume_restores_events_as_audit_data_without_replay(tmp_path) -> None:
    source = _optimizer()
    candidates = source.ask()
    records = _trusted_records(candidates)
    source.tell(records[:1])
    source_event_rows = source.events.to_rows()
    checkpoint_path = tmp_path / "cmaes-events.evocore-checkpoint.json"
    source.save_checkpoint(checkpoint_path, source.ask_tell_checkpoint())

    restored = _optimizer()
    summary = restored.resume_ask_tell_checkpoint(checkpoint_path)
    result = restored.tell(records[1:])

    assert restored.events.to_rows()[: len(source_event_rows)] == source_event_rows
    assert summary.trusted_count == 1
    assert result.trusted_count == 3
    assert restored.state_summary().trusted_count == 4
