import pytest

from evocore import (
    EvaluationRecord,
    Evaluator,
    GAEngine,
    GeneDef,
    GeneSpace,
    MultiFidelityPolicy,
)
from evocore.exceptions import FitnessError


class SphereEvaluator(Evaluator):
    def evaluate(self, candidates, rung):
        confidence = rung.confidence
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                score=-sum(float(value) ** 2 for value in candidate.genes),
                confidence=confidence,
                rung=rung.name,
                cost=rung.budget,
            )
            for candidate in candidates
        ]


class DroppingEvaluator(Evaluator):
    def evaluate(self, candidates, rung):
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=1.0,
                confidence=rung.confidence,
                rung=rung.name,
                cost=rung.budget,
            )
            for candidate in candidates[:-1]
        ]


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


def test_ga_ask_assigns_stable_batch_id_per_batch() -> None:
    engine = GAEngine(_space(), population_size=6, generations=5, seed=123)

    first = engine.ask(3)
    second = engine.ask(2)

    assert len({candidate.batch_id for candidate in first}) == 1
    assert len({candidate.batch_id for candidate in second}) == 1
    assert first[0].batch_id != second[0].batch_id
    assert first[0].batch_id.startswith("b-")


def test_ga_ask_populates_unique_candidate_hash_telemetry() -> None:
    engine = GAEngine(_space(), population_size=6, generations=5, seed=123)

    candidates = engine.ask(4)

    assert engine.vnext_telemetry.total_candidates_proposed == 4
    assert len(engine.vnext_telemetry.unique_candidate_hashes) == len(
        {tuple(candidate.genes) for candidate in candidates}
    )


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


def test_ga_tell_accepts_partial_records_for_one_batch() -> None:
    engine = GAEngine(_space(), population_size=4, generations=5, seed=123)
    candidates = engine.ask(4)
    records = [
        EvaluationRecord(
            candidate_id=candidate.candidate_id,
            batch_id=candidate.batch_id,
            score=float(index),
            confidence="trusted_full",
            rung="full",
            cost=1.0,
        )
        for index, candidate in enumerate(candidates)
    ]

    first = engine.tell(records[:2])
    second = engine.tell(records[2:])

    assert first.trusted_count == 2
    assert second.trusted_count == 2
    assert engine.vnext_telemetry.candidates_full_evaluated == 4
    assert engine.best_candidate.candidate_id == candidates[-1].candidate_id


def test_ga_tell_rejects_duplicate_candidate_rung_record() -> None:
    engine = GAEngine(_space(), population_size=4, generations=5, seed=123)
    candidate = engine.ask(1)[0]
    record = EvaluationRecord(
        candidate_id=candidate.candidate_id,
        batch_id=candidate.batch_id,
        score=1.0,
        confidence="trusted_full",
        rung="full",
        cost=1.0,
    )

    engine.tell([record])

    with pytest.raises(FitnessError, match="already has"):
        engine.tell([record])


def test_ga_tell_rejects_explicit_batch_mismatch() -> None:
    engine = GAEngine(_space(), population_size=4, generations=5, seed=123)
    candidate = engine.ask(1)[0]

    with pytest.raises(FitnessError, match="batch_id"):
        engine.tell(
            [
                EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    batch_id="b-wrong",
                    score=1.0,
                    confidence="trusted_full",
                    rung="full",
                    cost=1.0,
                )
            ]
        )


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


def test_ga_run_uses_policy_and_returns_vnext_telemetry() -> None:
    engine = GAEngine(_space(), population_size=6, generations=20, seed=123)
    policy = MultiFidelityPolicy.single_full(budget=12, batch_size=4)

    result = engine.run(SphereEvaluator(), policy=policy)

    assert result.n_evaluations == 12
    assert result.best_individual.fitness_valid
    assert result.telemetry.candidates_full_evaluated == 12
    assert result.stop_reason == "max_evaluations"


def test_ga_run_rejects_evaluator_that_omits_assigned_records() -> None:
    engine = GAEngine(_space(), population_size=4, generations=5, seed=123)

    with pytest.raises(FitnessError, match="missing evaluation records"):
        engine.run(
            DroppingEvaluator(),
            policy=MultiFidelityPolicy.single_full(budget=4, batch_size=4),
        )


def test_ga_run_resets_vnext_state_for_repeated_runs() -> None:
    engine = GAEngine(_space(), population_size=4, generations=5, seed=123)
    policy = MultiFidelityPolicy.single_full(budget=8, batch_size=4)

    first = engine.run(SphereEvaluator(), policy=policy)
    second = engine.run(SphereEvaluator(), policy=policy)

    assert first.n_evaluations == 8
    assert second.n_evaluations == 8
    assert len(second.final_population) == 8
    assert second.telemetry.candidates_full_evaluated == 8
