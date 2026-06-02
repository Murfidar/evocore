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
