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
