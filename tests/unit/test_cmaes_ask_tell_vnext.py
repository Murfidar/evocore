import pytest

from evocore import CMAESEngine, EvaluationRecord, GeneDef, GeneSpace
from evocore.exceptions import FitnessError


def _space() -> GeneSpace:
    return GeneSpace(
        [
            GeneDef("x", "float", -5.0, 5.0),
            GeneDef("period", "int", 2, 20),
        ]
    )


def test_cma_ask_returns_candidate_batch() -> None:
    engine = CMAESEngine(_space(), population_size=4, seed=7)

    candidates = engine.ask()

    assert len(candidates) == 4
    assert len({candidate.candidate_id for candidate in candidates}) == 4
    assert len({candidate.batch_id for candidate in candidates}) == 1
    assert all(candidate.params is not None for candidate in candidates)


def test_cma_tell_ignores_partial_records_for_state_update() -> None:
    engine = CMAESEngine(_space(), population_size=4, seed=7)
    candidates = engine.ask()
    generation_before = engine.generation

    summary = engine.tell(
        [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                score=1.0,
                confidence="partial",
                rung="cheap",
                cost=0.1,
            )
            for candidate in candidates
        ]
    )

    assert summary.partial_count == 4
    assert engine.generation == generation_before


def test_cma_tell_trusted_records_updates_state() -> None:
    engine = CMAESEngine(_space(), population_size=4, seed=7)
    candidates = engine.ask()

    summary = engine.tell(
        [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                score=-sum(float(value) ** 2 for value in candidate.genes),
                confidence="trusted_full",
                rung="full",
                cost=1.0,
            )
            for candidate in candidates
        ]
    )

    assert summary.trusted_count == 4
    assert engine.generation == 1


def test_cma_tell_accumulates_trusted_records_across_partial_calls() -> None:
    engine = CMAESEngine(_space(), population_size=4, seed=7)
    candidates = engine.ask()
    records = [
        EvaluationRecord(
            candidate_id=candidate.candidate_id,
            batch_id=candidate.batch_id,
            score=-sum(float(value) ** 2 for value in candidate.genes),
            confidence="trusted_full",
            rung="full",
            cost=1.0,
        )
        for candidate in candidates
    ]

    first = engine.tell(records[:2])

    assert first.trusted_count == 2
    assert engine.generation == 0

    second = engine.tell(records[2:])

    assert second.trusted_count == 2
    assert engine.generation == 1


def test_cma_tell_completes_batch_out_of_order() -> None:
    engine = CMAESEngine(_space(), population_size=4, seed=7)
    candidates = engine.ask()
    records = [
        EvaluationRecord(
            candidate_id=candidate.candidate_id,
            batch_id=candidate.batch_id,
            score=-sum(float(value) ** 2 for value in candidate.genes),
            confidence="trusted_full",
            rung="full",
            cost=1.0,
        )
        for candidate in candidates
    ]

    engine.tell([records[2], records[0]])
    assert engine.generation == 0

    engine.tell([records[3], records[1]])
    assert engine.generation == 1


def test_cma_tell_rejects_duplicate_trusted_record_after_batch_consumed() -> None:
    engine = CMAESEngine(_space(), population_size=4, seed=7)
    candidates = engine.ask()
    records = [
        EvaluationRecord(
            candidate_id=candidate.candidate_id,
            batch_id=candidate.batch_id,
            score=-sum(float(value) ** 2 for value in candidate.genes),
            confidence="trusted_full",
            rung="full",
            cost=1.0,
        )
        for candidate in candidates
    ]

    engine.tell(records)

    with pytest.raises(FitnessError, match="consumed"):
        engine.tell([records[0]])
