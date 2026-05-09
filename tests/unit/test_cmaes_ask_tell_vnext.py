from evocore import CMAESEngine, EvaluationRecord, GeneDef, GeneSpace


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
