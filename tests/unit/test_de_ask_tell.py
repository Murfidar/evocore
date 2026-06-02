import pytest

from evocore import DifferentialEvolutionOptimizer, EvaluationRecord, Gene, GeneSpace
from evocore.core.errors import ConfigurationError, FitnessError


def _mixed_space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("period", "int", 2, 20),
            Gene("enabled", "bool"),
            Gene("fixed", "float", 1.5, 1.5),
        ]
    )


def _records(candidates, scores, confidence="trusted_full"):
    return [
        EvaluationRecord(
            candidate_id=candidate.candidate_id,
            batch_id=candidate.batch_id,
            score=score,
            confidence=confidence,
            stage="full",
        )
        for candidate, score in zip(candidates, scores, strict=False)
    ]


def test_de_initial_ask_returns_valid_decoded_candidates() -> None:
    engine = DifferentialEvolutionOptimizer(_mixed_space(), population_size=6, seed=42)

    candidates = engine.ask()

    assert len(candidates) == 6
    assert {candidate.batch_id for candidate in candidates} == {candidates[0].batch_id}
    assert [candidate.origin for candidate in candidates] == ["random"] * 6
    for candidate in candidates:
        assert isinstance(candidate.genes[0], float)
        assert isinstance(candidate.genes[1], int)
        assert type(candidate.genes[2]) is bool
        assert candidate.genes[3] == pytest.approx(1.5)
        _mixed_space().validate_genes(candidate.genes)


def test_de_initial_tell_fills_target_population_and_best_candidate() -> None:
    engine = DifferentialEvolutionOptimizer(_mixed_space(), population_size=6, seed=42)
    candidates = engine.ask()

    result = engine.tell(_records(candidates, [0.0, 1.0, 5.0, 2.0, 3.0, 4.0]))

    assert result.accepted_count == 6
    assert result.state_accepted_count == 6
    assert len(result.acceptance_decisions) == 6
    assert all(decision.accepted_for_state for decision in result.acceptance_decisions)
    assert result.best_candidate_id == candidates[2].candidate_id
    assert result.best_score == pytest.approx(5.0)
    assert engine.state_summary().trusted_count == 6
    assert engine.state_summary().pending_batch_ids == ()


def test_de_ask_rejects_non_positive_count() -> None:
    engine = DifferentialEvolutionOptimizer(_mixed_space(), population_size=6, seed=42)

    with pytest.raises(ConfigurationError, match="ask\\(n\\) requires n > 0"):
        engine.ask(0)


def test_de_tell_rejects_unknown_candidate() -> None:
    engine = DifferentialEvolutionOptimizer(_mixed_space(), population_size=6, seed=42)

    with pytest.raises(FitnessError, match="unknown candidate_id"):
        engine.tell(
            [
                EvaluationRecord(
                    candidate_id="missing",
                    batch_id="b-missing",
                    score=1.0,
                    confidence="trusted_full",
                    stage="full",
                )
            ]
        )


def _trusted_engine() -> tuple[DifferentialEvolutionOptimizer, list]:
    engine = DifferentialEvolutionOptimizer(_mixed_space(), population_size=6, seed=42)
    candidates = engine.ask()
    engine.tell(_records(candidates, [0.0, 1.0, 5.0, 2.0, 3.0, 4.0]))
    return engine, candidates


def test_de_trial_ask_returns_one_trial_per_target_with_mapping_metadata() -> None:
    engine, targets = _trusted_engine()

    trials = engine.ask()

    assert len(trials) == 6
    assert {trial.origin for trial in trials} == {"mutation"}
    assert set(engine._trial_target_slots) == {trial.candidate_id for trial in trials}
    assert set(engine._trial_target_candidate_ids) == {trial.candidate_id for trial in trials}
    assert {trial.metadata["target_candidate_id"] for trial in trials} == {
        target.candidate_id for target in targets
    }
    assert {trial.metadata["target_slot"] for trial in trials} == set(range(6))


def test_de_trial_generation_is_deterministic_for_same_seed_and_state() -> None:
    left, _ = _trusted_engine()
    right, _ = _trusted_engine()

    left_trials = left.ask()
    right_trials = right.ask()

    assert [trial.genes for trial in left_trials] == [trial.genes for trial in right_trials]
    assert [trial.metadata["target_slot"] for trial in left_trials] == [
        trial.metadata["target_slot"] for trial in right_trials
    ]


def test_de_trial_generation_preserves_gene_types_and_fixed_values() -> None:
    engine, _ = _trusted_engine()

    trials = engine.ask()

    for trial in trials:
        assert isinstance(trial.genes[0], float)
        assert isinstance(trial.genes[1], int)
        assert type(trial.genes[2]) is bool
        assert trial.genes[3] == pytest.approx(1.5)
        _mixed_space().validate_genes(trial.genes)


def test_de_tell_replaces_target_when_trial_is_better() -> None:
    engine, targets = _trusted_engine()
    trials = engine.ask()
    trial = trials[0]
    target_id = trial.metadata["target_candidate_id"]
    target_slot = trial.metadata["target_slot"]

    result = engine.tell(_records([trial], [100.0]))

    assert result.state_accepted_count == 1
    assert result.acceptance_decisions[0].accepted_for_state is True
    assert result.acceptance_decisions[0].reason == "trial_replaced_target"
    assert result.acceptance_decisions[0].target_candidate_id == target_id
    assert result.acceptance_decisions[0].target_slot == target_slot
    assert engine._target_candidate_ids[target_slot] == trial.candidate_id
    assert targets[0].candidate_id not in engine._target_candidate_ids


def test_de_tell_keeps_target_when_trial_is_worse() -> None:
    engine, targets = _trusted_engine()
    trials = engine.ask()
    trial = trials[2]
    target_slot = trial.metadata["target_slot"]
    target_id = trial.metadata["target_candidate_id"]

    result = engine.tell(_records([trial], [-100.0]))

    assert result.state_accepted_count == 0
    assert result.acceptance_decisions[0].accepted_for_state is False
    assert result.acceptance_decisions[0].reason == "trial_kept_target"
    assert result.acceptance_decisions[0].target_candidate_id == target_id
    assert result.acceptance_decisions[0].target_slot == target_slot
    assert engine._target_candidate_ids[target_slot] == targets[target_slot].candidate_id


def test_de_tell_replaces_on_equal_score() -> None:
    engine, _ = _trusted_engine()
    trials = engine.ask()
    trial = trials[1]

    result = engine.tell(_records([trial], [1.0]))

    assert result.acceptance_decisions[0].accepted_for_state is True
    assert result.acceptance_decisions[0].reason == "trial_replaced_target"


def test_de_minimize_replaces_when_trial_score_is_lower() -> None:
    engine = DifferentialEvolutionOptimizer(
        _mixed_space(),
        population_size=6,
        seed=42,
        direction="minimize",
    )
    targets = engine.ask()
    engine.tell(_records(targets, [10.0, 9.0, 8.0, 7.0, 6.0, 5.0]))
    trial = engine.ask()[0]

    result = engine.tell(_records([trial], [1.0]))

    assert result.acceptance_decisions[0].accepted_for_state is True
    assert result.best_candidate_id == trial.candidate_id
    assert result.best_score == pytest.approx(1.0)


def test_de_minimize_keeps_target_when_trial_score_is_higher() -> None:
    engine = DifferentialEvolutionOptimizer(
        _mixed_space(),
        population_size=6,
        seed=42,
        direction="minimize",
    )
    targets = engine.ask()
    engine.tell(_records(targets, [10.0, 9.0, 8.0, 7.0, 6.0, 5.0]))
    trial = engine.ask()[0]
    target_slot = trial.metadata["target_slot"]

    result = engine.tell(_records([trial], [100.0]))

    assert result.state_accepted_count == 0
    assert result.acceptance_decisions[0].accepted_for_state is False
    assert result.acceptance_decisions[0].reason == "trial_kept_target"
    assert engine._target_candidate_ids[target_slot] == trial.metadata["target_candidate_id"]


def test_de_cached_records_are_state_eligible_for_initialization_and_trial_replacement() -> None:
    engine = DifferentialEvolutionOptimizer(_mixed_space(), population_size=6, seed=42)
    targets = engine.ask()
    init_result = engine.tell(
        _records(targets, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], confidence="cached")
    )
    trial = engine.ask()[0]

    replacement_result = engine.tell(_records([trial], [100.0], confidence="cached"))

    assert init_result.cached_count == 6
    assert init_result.state_accepted_count == 6
    assert replacement_result.cached_count == 1
    assert replacement_result.state_accepted_count == 1
    assert replacement_result.acceptance_decisions[0].reason == "trial_replaced_target"


def test_de_partial_and_surrogate_records_do_not_replace_trial_targets() -> None:
    engine, targets = _trusted_engine()
    trials = engine.ask()

    result = engine.tell(
        [
            EvaluationRecord(
                candidate_id=trials[0].candidate_id,
                batch_id=trials[0].batch_id,
                score=100.0,
                confidence="partial",
                stage="screen",
            ),
            EvaluationRecord(
                candidate_id=trials[1].candidate_id,
                batch_id=trials[1].batch_id,
                score=100.0,
                confidence="surrogate",
                stage="surrogate",
            ),
        ]
    )

    assert result.partial_count == 1
    assert result.surrogate_count == 1
    assert result.state_accepted_count == 0
    assert result.acceptance_decisions == ()
    assert engine._target_candidate_ids == [target.candidate_id for target in targets]


def test_de_rejects_state_record_for_consumed_trial_batch() -> None:
    engine, _ = _trusted_engine()
    trials = engine.ask()
    rejected = [
        EvaluationRecord(
            candidate_id=trial.candidate_id,
            batch_id=trial.batch_id,
            score=None,
            confidence="rejected",
            stage="full",
            metadata={"reason": "constraint"},
        )
        for trial in trials
    ]
    engine.tell(rejected)

    with pytest.raises(FitnessError, match="already been consumed"):
        engine.tell(
            [
                EvaluationRecord(
                    candidate_id=trials[0].candidate_id,
                    batch_id=trials[0].batch_id,
                    score=100.0,
                    confidence="trusted_full",
                    stage="retry",
                )
            ]
        )


def test_de_rejected_trial_does_not_replace_target_and_can_complete_batch() -> None:
    engine, targets = _trusted_engine()
    trials = engine.ask()
    rejected = [
        EvaluationRecord(
            candidate_id=trial.candidate_id,
            batch_id=trial.batch_id,
            score=None,
            confidence="rejected",
            stage="full",
            metadata={"reason": "constraint"},
        )
        for trial in trials
    ]

    result = engine.tell(rejected)

    assert result.rejected_count == len(trials)
    assert result.state_accepted_count == 0
    assert result.consumed_batch_ids == (trials[0].batch_id,)
    assert engine._target_candidate_ids == [target.candidate_id for target in targets]
    assert engine.generation == 1
