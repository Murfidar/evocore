import copy

import pytest

from evocore import (
    BudgetPolicy,
    DifferentialEvolutionOptimizer,
    EvaluationRecord,
    EvaluationStage,
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


def test_jde_trial_ask_registers_rust_proposed_pending_params() -> None:
    optimizer = DifferentialEvolutionOptimizer(
        _space(),
        population_size=6,
        strategy="jde-rand1bin",
        seed=42,
    )
    initial = optimizer.ask()
    optimizer.tell(
        [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=float(index),
                confidence="trusted_full",
                stage="full",
            )
            for index, candidate in enumerate(initial)
        ]
    )

    trials = optimizer.ask(3)

    assert len(trials) == 3
    for trial in trials:
        assert trial.candidate_id in optimizer._de_strategy_state.pending_trial_params
        pending = optimizer._de_strategy_state.pending_trial_params[trial.candidate_id]
        assert pending.target_slot == trial.metadata["adaptive_slot"]
        assert pending.mutation_factor == trial.metadata["mutation_factor"]
        assert pending.crossover_rate == trial.metadata["crossover_rate"]


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


def _pending_jde_checkpoint_payload():
    engine = _trusted_jde_engine()
    trials = engine.ask()
    return engine.ask_tell_checkpoint().to_dict(), trials


def _new_jde_optimizer() -> DifferentialEvolutionOptimizer:
    return DifferentialEvolutionOptimizer(
        _space(),
        population_size=6,
        strategy="jde-rand1bin",
        mutation_factor=0.5,
        crossover_rate=0.9,
        seed=42,
    )


def test_jde_checkpoint_rejects_missing_pending_strategy_params() -> None:
    payload, trials = _pending_jde_checkpoint_payload()
    del payload["state"]["payload"]["strategy_state"]["pending_trial_params"][
        trials[0].candidate_id
    ]

    with pytest.raises(CheckpointError, match="pending_trial_params"):
        _new_jde_optimizer().resume_ask_tell_checkpoint(payload)


def test_jde_checkpoint_rejects_unknown_pending_strategy_params() -> None:
    payload, _ = _pending_jde_checkpoint_payload()
    pending = payload["state"]["payload"]["strategy_state"]["pending_trial_params"]
    pending["c-unknown"] = {
        "target_slot": 0,
        "mutation_factor": 0.5,
        "crossover_rate": 0.9,
    }

    with pytest.raises(CheckpointError, match="pending_trial_params"):
        _new_jde_optimizer().resume_ask_tell_checkpoint(payload)


def test_jde_checkpoint_rejects_pending_strategy_slot_mismatch() -> None:
    payload, trials = _pending_jde_checkpoint_payload()
    payload["state"]["payload"]["strategy_state"]["pending_trial_params"][trials[0].candidate_id][
        "target_slot"
    ] = 1

    with pytest.raises(CheckpointError, match="target_slot"):
        _new_jde_optimizer().resume_ask_tell_checkpoint(payload)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("mutation_factor", float("nan"), "mutation_factor"),
        ("mutation_factor", -0.1, "mutation_factor"),
        ("crossover_rate", float("inf"), "crossover_rate"),
        ("crossover_rate", 1.1, "crossover_rate"),
    ],
)
def test_jde_checkpoint_rejects_invalid_pending_strategy_parameters(
    key: str,
    value: float,
    message: str,
) -> None:
    payload, trials = _pending_jde_checkpoint_payload()
    payload["state"]["payload"]["strategy_state"]["pending_trial_params"][trials[0].candidate_id][
        key
    ] = value

    with pytest.raises(CheckpointError, match=message):
        _new_jde_optimizer().resume_ask_tell_checkpoint(payload)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("f_by_slot", float("nan"), "f_by_slot"),
        ("f_by_slot", -0.1, "f_by_slot"),
        ("cr_by_slot", float("inf"), "cr_by_slot"),
        ("cr_by_slot", 1.1, "cr_by_slot"),
    ],
)
def test_jde_checkpoint_rejects_invalid_committed_strategy_parameters(
    key: str,
    value: float,
    message: str,
) -> None:
    payload, _ = _pending_jde_checkpoint_payload()
    payload["state"]["payload"]["strategy_state"][key][0] = value

    with pytest.raises(CheckpointError, match=message):
        _new_jde_optimizer().resume_ask_tell_checkpoint(payload)


class TwoStageSphere:
    def evaluate(self, candidates, context):
        assert context.stage is not None
        scale = 0.5 if context.stage.name == "cheap" else 1.0
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=-scale * sum(float(value) ** 2 for value in candidate.genes),
                confidence=context.stage.confidence,
                stage=context.stage.name,
                cost=context.stage.budget,
            )
            for candidate in candidates
        ]


def test_jde_policy_run_keeps_strategy_state_consistent() -> None:
    policy = BudgetPolicy(
        stages=[
            EvaluationStage("cheap", budget=0.1, promote_fraction=0.5, confidence="partial"),
            EvaluationStage("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full"),
        ],
        max_evaluations=18,
        batch_size=6,
        exploration_fraction=0.0,
        audit_fraction=0.0,
    )
    optimizer = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=3,
        strategy="jde-rand1bin",
        seed=42,
    )

    result = optimizer.run(TwoStageSphere(), policy=policy)

    assert result.optimizer_type == "DifferentialEvolutionOptimizer"
    assert result.reproducibility.optimizer_config["parameters"]["strategy"] == "jde-rand1bin"
    assert len(optimizer._de_strategy_state.f_by_slot) == 6
    assert len(optimizer._de_strategy_state.cr_by_slot) == 6
    assert not optimizer._de_strategy_state.pending_trial_params
