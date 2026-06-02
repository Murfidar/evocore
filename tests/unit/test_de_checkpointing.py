import copy

import pytest

from evocore import DifferentialEvolutionOptimizer, EvaluationRecord, Gene, GeneSpace
from evocore.core.errors import CheckpointError


def _space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("period", "int", 2, 20),
            Gene("enabled", "bool"),
        ]
    )


def _records(candidates, scores):
    return [
        EvaluationRecord(
            candidate_id=candidate.candidate_id,
            batch_id=candidate.batch_id,
            score=score,
            confidence="trusted_full",
            stage="full",
        )
        for candidate, score in zip(candidates, scores, strict=False)
    ]


def _target_genes(engine: DifferentialEvolutionOptimizer) -> list[list[object]]:
    return [
        list(engine._candidates_by_id[candidate_id].genes)
        for candidate_id in engine._target_candidate_ids
    ]


def _state_tuple(engine: DifferentialEvolutionOptimizer) -> tuple[object, ...]:
    summary = engine.state_summary()
    return (
        engine.generation,
        summary.best_candidate_id,
        summary.best_score,
        summary.trusted_count,
        tuple(summary.pending_batch_ids),
        tuple(engine._target_candidate_ids),
        tuple(tuple(values) for values in _target_genes(engine)),
    )


def test_de_checkpoint_restores_after_initial_ask() -> None:
    engine = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)
    candidates = engine.ask()
    snapshot = engine.ask_tell_checkpoint(metadata={"phase": "submitted"})

    restored = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)
    summary = restored.resume_ask_tell_checkpoint(snapshot.to_dict())

    assert summary.pending_batch_ids == (candidates[0].batch_id,)
    assert restored.tell(_records(candidates, [0, 1, 2, 3, 4, 5])).trusted_count == 6


def test_de_checkpoint_restores_after_partial_initial_tell() -> None:
    engine = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)
    candidates = engine.ask()
    engine.tell(_records(candidates[:3], [0, 1, 2]))
    snapshot = engine.ask_tell_checkpoint()

    restored = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)
    restored.resume_ask_tell_checkpoint(snapshot.to_dict())
    result = restored.tell(_records(candidates[3:], [3, 4, 5]))

    assert result.best_score == pytest.approx(5.0)
    assert restored.state_summary().trusted_count == 6


def test_de_checkpoint_resume_matches_uninterrupted_initialization_completion() -> None:
    original = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)
    candidates = original.ask()
    original.tell(_records(candidates[:2], [1.0, 2.0]))
    snapshot = original.ask_tell_checkpoint()

    uninterrupted = copy.deepcopy(original)
    restored = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)
    restored.resume_ask_tell_checkpoint(snapshot.to_dict())

    remaining_records = _records(candidates[2:], [3.0, 4.0, 5.0, 6.0])
    uninterrupted_result = uninterrupted.tell(remaining_records)
    restored_result = restored.tell(remaining_records)

    assert _state_tuple(restored) == _state_tuple(uninterrupted)
    assert restored_result.best_score == pytest.approx(uninterrupted_result.best_score)
    assert restored_result.state_accepted_count == uninterrupted_result.state_accepted_count
    assert [decision.accepted_for_state for decision in restored_result.acceptance_decisions] == [
        decision.accepted_for_state for decision in uninterrupted_result.acceptance_decisions
    ]


def test_de_checkpoint_resume_matches_uninterrupted_trial_completion() -> None:
    original = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)
    targets = original.ask()
    original.tell(_records(targets, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
    trials = original.ask()
    original.tell(_records(trials[:2], [100.0, -100.0]))
    snapshot = original.ask_tell_checkpoint()

    uninterrupted = copy.deepcopy(original)
    restored = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)
    restored.resume_ask_tell_checkpoint(snapshot.to_dict())

    remaining_records = _records(trials[2:], [-101.0, 99.0, -102.0, 98.0])
    uninterrupted_result = uninterrupted.tell(remaining_records)
    restored_result = restored.tell(remaining_records)

    assert _state_tuple(restored) == _state_tuple(uninterrupted)
    assert restored_result.consumed_batch_ids == uninterrupted_result.consumed_batch_ids
    assert restored_result.state_accepted_count == uninterrupted_result.state_accepted_count
    assert [
        (decision.accepted_for_state, decision.reason, decision.target_slot)
        for decision in restored_result.acceptance_decisions
    ] == [
        (decision.accepted_for_state, decision.reason, decision.target_slot)
        for decision in uninterrupted_result.acceptance_decisions
    ]


def test_de_checkpoint_restores_pending_trial_mapping() -> None:
    engine = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)
    targets = engine.ask()
    engine.tell(_records(targets, [0, 1, 2, 3, 4, 5]))
    trials = engine.ask()
    snapshot = engine.ask_tell_checkpoint()

    restored = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)
    restored.resume_ask_tell_checkpoint(snapshot.to_dict())
    result = restored.tell(_records([trials[0]], [100.0]))

    assert result.acceptance_decisions[0].accepted_for_state is True
    assert result.acceptance_decisions[0].target_slot == 0


def test_de_checkpoint_rejects_wrong_optimizer_identity() -> None:
    engine = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)
    snapshot = engine.ask_tell_checkpoint().to_dict()
    snapshot["optimizer"]["optimizer_type"] = "GeneticAlgorithmOptimizer"
    restored = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)

    with pytest.raises(CheckpointError, match="optimizer_type"):
        restored.resume_ask_tell_checkpoint(snapshot)


@pytest.mark.parametrize(
    "field",
    [
        "event_index",
        "generation",
        "candidates_by_id",
        "batches_by_id",
        "target_candidate_ids",
        "trial_target_slots",
        "trial_target_candidate_ids",
        "telemetry",
        "events",
    ],
)
def test_de_checkpoint_rejects_missing_required_payload_fields(field: str) -> None:
    engine = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)
    engine.ask()
    payload = engine.ask_tell_checkpoint().to_dict()
    del payload["state"]["payload"][field]

    restored = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)

    with pytest.raises(CheckpointError, match=field):
        restored.resume_ask_tell_checkpoint(payload)


def test_de_checkpoint_rejects_target_population_larger_than_config() -> None:
    engine = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)
    candidates = engine.ask()
    engine.tell(_records(candidates, [0, 1, 2, 3, 4, 5]))
    payload = engine.ask_tell_checkpoint().to_dict()
    payload["state"]["payload"]["target_candidate_ids"].append(candidates[0].candidate_id)

    restored = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)

    with pytest.raises(CheckpointError, match="target_candidate_ids"):
        restored.resume_ask_tell_checkpoint(payload)


def test_de_checkpoint_rejects_trial_mapping_with_wrong_target_slot() -> None:
    engine = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)
    candidates = engine.ask()
    engine.tell(_records(candidates, [0, 1, 2, 3, 4, 5]))
    trial = engine.ask()[0]
    payload = engine.ask_tell_checkpoint().to_dict()
    payload["state"]["payload"]["trial_target_slots"][trial.candidate_id] = 99

    restored = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)

    with pytest.raises(CheckpointError, match="target slot"):
        restored.resume_ask_tell_checkpoint(payload)


def test_de_checkpoint_rejects_trial_mapping_with_mismatched_target_id() -> None:
    engine = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)
    candidates = engine.ask()
    engine.tell(_records(candidates, [0, 1, 2, 3, 4, 5]))
    trial = engine.ask()[0]
    payload = engine.ask_tell_checkpoint().to_dict()
    payload["state"]["payload"]["trial_target_candidate_ids"][trial.candidate_id] = candidates[
        1
    ].candidate_id

    restored = DifferentialEvolutionOptimizer(_space(), population_size=6, seed=42)

    with pytest.raises(CheckpointError, match="target_candidate_id"):
        restored.resume_ask_tell_checkpoint(payload)
