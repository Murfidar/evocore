import copy

import pytest

from evocore import (
    DifferentialEvolutionOptimizer,
    EvaluationRecord,
    Gene,
    GeneSpace,
)
from evocore.core.errors import CheckpointError
from evocore.optimizers.de.adaptive import JDETrialParameters


def _space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("period", "int", 2, 20),
            Gene("enabled", "bool"),
        ]
    )


def _records(candidates, scores, confidence="trusted_full", stage="full"):
    return [
        EvaluationRecord(
            candidate_id=candidate.candidate_id,
            batch_id=candidate.batch_id,
            score=score,
            confidence=confidence,
            stage=stage,
        )
        for candidate, score in zip(candidates, scores, strict=False)
    ]


def _trusted_jde_engine() -> DifferentialEvolutionOptimizer:
    engine = DifferentialEvolutionOptimizer(
        _space(),
        population_size=6,
        strategy="jde-rand1bin",
        mutation_factor=0.5,
        crossover_rate=0.9,
        seed=42,
    )
    targets = engine.ask()
    engine.tell(_records(targets, [0, 1, 2, 3, 4, 5]))
    return engine


def _state_tuple(engine: DifferentialEvolutionOptimizer):
    state = engine._de_strategy_state
    return (
        tuple(engine._target_candidate_ids),
        tuple(round(value, 12) for value in state.f_by_slot),
        tuple(round(value, 12) for value in state.cr_by_slot),
        tuple(sorted(state.pending_trial_params)),
        engine.state_summary().best_candidate_id,
        engine.state_summary().trusted_count,
    )


def test_jde_trial_metadata_is_deterministic_for_same_seed() -> None:
    left = _trusted_jde_engine()
    right = _trusted_jde_engine()

    left_trials = left.ask()
    right_trials = right.ask()

    left_metadata = [
        (
            trial.metadata["strategy"],
            trial.metadata["target_slot"],
            trial.metadata["adaptive_slot"],
            trial.metadata["mutation_factor"],
            trial.metadata["crossover_rate"],
            trial.genes,
        )
        for trial in left_trials
    ]
    right_metadata = [
        (
            trial.metadata["strategy"],
            trial.metadata["target_slot"],
            trial.metadata["adaptive_slot"],
            trial.metadata["mutation_factor"],
            trial.metadata["crossover_rate"],
            trial.genes,
        )
        for trial in right_trials
    ]

    assert left_metadata == right_metadata
    assert {trial.metadata["strategy"] for trial in left_trials} == {"jde-rand1bin"}


def test_jde_acceptance_commits_trial_parameters() -> None:
    engine = _trusted_jde_engine()
    trial = engine.ask()[0]
    slot = trial.metadata["target_slot"]
    engine._de_strategy_state.pending_trial_params[trial.candidate_id] = JDETrialParameters(
        target_slot=slot,
        mutation_factor=0.37,
        crossover_rate=0.41,
    )

    result = engine.tell(_records([trial], [100.0]))

    assert result.acceptance_decisions[0].accepted_for_state is True
    assert engine._de_strategy_state.f_by_slot[slot] == pytest.approx(0.37)
    assert engine._de_strategy_state.cr_by_slot[slot] == pytest.approx(0.41)
    assert trial.candidate_id not in engine._de_strategy_state.pending_trial_params


def test_jde_rejection_preserves_previous_parameters() -> None:
    engine = _trusted_jde_engine()
    trial = engine.ask()[0]
    slot = trial.metadata["target_slot"]
    original = (
        engine._de_strategy_state.f_by_slot[slot],
        engine._de_strategy_state.cr_by_slot[slot],
    )
    engine._de_strategy_state.pending_trial_params[trial.candidate_id] = JDETrialParameters(
        target_slot=slot,
        mutation_factor=0.22,
        crossover_rate=0.33,
    )

    result = engine.tell(_records([trial], [-100.0]))

    assert result.acceptance_decisions[0].accepted_for_state is False
    assert engine._de_strategy_state.f_by_slot[slot] == pytest.approx(original[0])
    assert engine._de_strategy_state.cr_by_slot[slot] == pytest.approx(original[1])
    assert trial.candidate_id not in engine._de_strategy_state.pending_trial_params


def test_jde_partial_records_do_not_adapt_or_clear_pending_trial() -> None:
    engine = _trusted_jde_engine()
    trial = engine.ask()[0]
    slot = trial.metadata["target_slot"]
    original = (
        engine._de_strategy_state.f_by_slot[slot],
        engine._de_strategy_state.cr_by_slot[slot],
    )

    result = engine.tell(_records([trial], [10.0], confidence="partial", stage="cheap"))

    assert result.partial_count == 1
    assert engine._de_strategy_state.f_by_slot[slot] == pytest.approx(original[0])
    assert engine._de_strategy_state.cr_by_slot[slot] == pytest.approx(original[1])
    assert trial.candidate_id in engine._de_strategy_state.pending_trial_params


def test_jde_rejected_record_preserves_previous_parameters_and_clears_pending() -> None:
    engine = _trusted_jde_engine()
    trial = engine.ask()[0]
    slot = trial.metadata["target_slot"]
    original = (
        engine._de_strategy_state.f_by_slot[slot],
        engine._de_strategy_state.cr_by_slot[slot],
    )

    result = engine.tell(
        _records([trial], [None], confidence="rejected", stage="cheap__de_screened_out")
    )

    assert result.rejected_count == 1
    assert engine._de_strategy_state.f_by_slot[slot] == pytest.approx(original[0])
    assert engine._de_strategy_state.cr_by_slot[slot] == pytest.approx(original[1])
    assert trial.candidate_id not in engine._de_strategy_state.pending_trial_params


def test_jde_checkpoint_resume_matches_uninterrupted_pending_trial() -> None:
    original = _trusted_jde_engine()
    trials = original.ask()
    original.tell(_records(trials[:2], [100.0, -100.0]))
    snapshot = original.ask_tell_checkpoint()

    uninterrupted = copy.deepcopy(original)
    restored = DifferentialEvolutionOptimizer(
        _space(),
        population_size=6,
        strategy="jde-rand1bin",
        mutation_factor=0.5,
        crossover_rate=0.9,
        seed=42,
    )
    restored.resume_ask_tell_checkpoint(snapshot.to_dict())

    remaining_records = _records(trials[2:], [-101.0, 99.0, -102.0, 98.0])
    uninterrupted_result = uninterrupted.tell(remaining_records)
    restored_result = restored.tell(remaining_records)

    assert _state_tuple(restored) == _state_tuple(uninterrupted)
    assert restored_result.state_accepted_count == uninterrupted_result.state_accepted_count


def test_jde_checkpoint_rejects_missing_strategy_state() -> None:
    engine = _trusted_jde_engine()
    engine.ask()
    payload = engine.ask_tell_checkpoint().to_dict()
    del payload["state"]["payload"]["strategy_state"]
    restored = DifferentialEvolutionOptimizer(
        _space(),
        population_size=6,
        strategy="jde-rand1bin",
        mutation_factor=0.5,
        crossover_rate=0.9,
        seed=42,
    )

    with pytest.raises(CheckpointError, match="strategy_state"):
        restored.resume_ask_tell_checkpoint(payload)
