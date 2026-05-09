from evocore import EvaluationRecord, GAEngine, GeneDef, GeneSpace


def _space() -> GeneSpace:
    return GeneSpace(
        [
            GeneDef("x", "float", -5.0, 5.0),
            GeneDef("mode", "int", 0, 3),
        ]
    )


def test_ga_ask_returns_candidates_with_params_and_ids() -> None:
    engine = GAEngine(_space(), population_size=6, generations=5, seed=123)

    candidates = engine.ask(4)

    assert len(candidates) == 4
    assert len({candidate.candidate_id for candidate in candidates}) == 4
    assert all(candidate.params is not None for candidate in candidates)
    assert all(candidate.origin == "random" for candidate in candidates)


def test_ga_tell_trusted_records_builds_trusted_population() -> None:
    engine = GAEngine(_space(), population_size=4, generations=5, seed=123)
    candidates = engine.ask(4)
    records = [
        EvaluationRecord(
            candidate_id=candidate.candidate_id,
            score=float(index),
            confidence="trusted_full",
            rung="full",
            cost=1.0,
        )
        for index, candidate in enumerate(candidates)
    ]

    summary = engine.tell(records)

    assert summary.trusted_count == 4
    assert engine.vnext_telemetry.candidates_full_evaluated == 4
    assert engine.best_candidate.candidate_id == candidates[-1].candidate_id


def test_ga_tell_surrogate_records_do_not_build_trusted_population() -> None:
    engine = GAEngine(_space(), population_size=4, generations=5, seed=123)
    candidates = engine.ask(4)

    engine.tell(
        [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                score=100.0,
                confidence="surrogate",
                rung="surrogate",
                cost=0.0,
            )
            for candidate in candidates
        ]
    )

    assert engine.vnext_telemetry.candidates_screened == 4
    assert engine.best_candidate is None
