import pytest

from evocore import CMAESOptimizer, EvaluationRecord, Gene, GeneSpace
from evocore.core.errors import FitnessError


def _space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("period", "int", 2, 20),
        ]
    )


def test_cma_ask_returns_candidate_batch() -> None:
    engine = CMAESOptimizer(_space(), population_size=4, seed=7)

    candidates = engine.ask()

    assert len(candidates) == 4
    assert len({candidate.candidate_id for candidate in candidates}) == 4
    assert len({candidate.batch_id for candidate in candidates}) == 1
    assert all(candidate.params is not None for candidate in candidates)


def test_cma_ask_keeps_continuous_samples_separate_from_repaired_candidate_genes() -> None:
    space = GeneSpace([Gene("period", "int", 5, 20), Gene("x", "float", -1.0, 1.0)])
    engine = CMAESOptimizer(space, population_size=6, seed=42)

    candidates = engine.ask()
    batch = engine._batches_by_id[candidates[0].batch_id]

    assert set(batch.continuous_samples_by_id) == {
        candidate.candidate_id for candidate in candidates
    }
    for candidate in candidates:
        continuous = batch.continuous_samples_by_id[candidate.candidate_id]
        assert isinstance(candidate.genes[0], int)
        assert isinstance(continuous[0], float)


def test_cma_constraint_penalties_complete_batch_without_trusted_candidates() -> None:
    engine = CMAESOptimizer(_space(), population_size=4, seed=7)
    candidates = engine.ask()
    records = [
        EvaluationRecord(
            candidate_id=candidate.candidate_id,
            batch_id=candidate.batch_id,
            score=-1.0e300,
            confidence="constraint_penalty",
            stage="projection",
        )
        for candidate in candidates
    ]

    result = engine.tell(records)

    assert result.state_accepted_count == len(candidates)
    assert result.trusted_count == 0
    assert result.consumed_batch_ids == (candidates[0].batch_id,)
    assert engine.candidate_snapshot(scope="trusted").candidates == ()


def test_cma_ask_records_append_only_ask_events() -> None:
    space = _space()
    engine = CMAESOptimizer(space, population_size=4, seed=7)

    candidates = engine.ask()

    assert len(engine.events) == 4
    rows = engine.events.to_rows()
    assert [row["event_index"] for row in rows] == [0, 1, 2, 3]
    assert all(row["event_type"] == "ask" for row in rows)
    assert rows[0]["batch_id"] == candidates[0].batch_id
    assert rows[0]["candidate_id"] == candidates[0].candidate_id
    assert rows[0]["candidate_hash"] == space.value_hash(candidates[0].genes)
    assert engine.vnext_telemetry.unique_candidate_hashes == {
        space.value_hash(candidate.genes) for candidate in candidates
    }


def test_cma_tell_ignores_partial_records_for_state_update() -> None:
    engine = CMAESOptimizer(_space(), population_size=4, seed=7)
    candidates = engine.ask()
    generation_before = engine.generation

    summary = engine.tell(
        [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                score=1.0,
                confidence="partial",
                stage="cheap",
                cost=0.1,
            )
            for candidate in candidates
        ]
    )

    assert summary.partial_count == 4
    assert engine.generation == generation_before


def test_cma_tell_trusted_records_updates_state() -> None:
    engine = CMAESOptimizer(_space(), population_size=4, seed=7)
    candidates = engine.ask()

    summary = engine.tell(
        [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                score=-sum(float(value) ** 2 for value in candidate.genes),
                confidence="trusted_full",
                stage="full",
                cost=1.0,
            )
            for candidate in candidates
        ]
    )

    assert summary.trusted_count == 4
    assert engine.generation == 1


def test_cma_tell_records_raw_and_comparison_scores_for_minimize() -> None:
    engine = CMAESOptimizer(_space(), population_size=4, seed=7, direction="minimize")
    candidates = engine.ask()

    engine.tell(
        [
            EvaluationRecord(
                candidate_id=candidates[0].candidate_id,
                batch_id=candidates[0].batch_id,
                score=3.0,
                confidence="trusted_full",
                stage="full",
                cost=1.0,
            )
        ]
    )

    row = engine.events.to_rows()[-1]
    assert row["event_type"] == "tell"
    assert row["raw_score"] == pytest.approx(3.0)
    assert row["comparison_score"] == pytest.approx(-3.0)
    assert row["status"] == "trusted"


def test_cma_tell_accumulates_trusted_records_across_partial_calls() -> None:
    engine = CMAESOptimizer(_space(), population_size=4, seed=7)
    candidates = engine.ask()
    records = [
        EvaluationRecord(
            candidate_id=candidate.candidate_id,
            batch_id=candidate.batch_id,
            score=-sum(float(value) ** 2 for value in candidate.genes),
            confidence="trusted_full",
            stage="full",
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
    engine = CMAESOptimizer(_space(), population_size=4, seed=7)
    candidates = engine.ask()
    records = [
        EvaluationRecord(
            candidate_id=candidate.candidate_id,
            batch_id=candidate.batch_id,
            score=-sum(float(value) ** 2 for value in candidate.genes),
            confidence="trusted_full",
            stage="full",
            cost=1.0,
        )
        for candidate in candidates
    ]

    engine.tell([records[2], records[0]])
    assert engine.generation == 0

    engine.tell([records[3], records[1]])
    assert engine.generation == 1


def test_cma_tell_rejects_duplicate_trusted_record_after_batch_consumed() -> None:
    engine = CMAESOptimizer(_space(), population_size=4, seed=7)
    candidates = engine.ask()
    records = [
        EvaluationRecord(
            candidate_id=candidate.candidate_id,
            batch_id=candidate.batch_id,
            score=-sum(float(value) ** 2 for value in candidate.genes),
            confidence="trusted_full",
            stage="full",
            cost=1.0,
        )
        for candidate in candidates
    ]

    engine.tell(records)

    with pytest.raises(FitnessError, match="consumed"):
        engine.tell([records[0]])


def test_cma_tell_empty_records_returns_noop_tell_result() -> None:
    engine = CMAESOptimizer(_space(), population_size=4, seed=7)

    result = engine.tell([])

    assert result.accepted_count == 0
    assert result.trusted_count == 0
    assert result.pending_batch_ids == ()


def test_cma_tell_rejects_unknown_explicit_batch_id() -> None:
    engine = CMAESOptimizer(_space(), population_size=4, seed=7)
    candidate = engine.ask()[0]

    with pytest.raises(FitnessError, match="unknown batch_id"):
        engine.tell(
            [
                EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    batch_id="b-missing",
                    score=1.0,
                    confidence="trusted_full",
                    stage="full",
                    cost=1.0,
                )
            ]
        )


def test_cma_state_summary_reports_best_and_pending_batches() -> None:
    engine = CMAESOptimizer(_space(), population_size=4, seed=7)
    candidates = engine.ask()

    before = engine.state_summary()

    assert before.best_candidate_id is None
    assert before.pending_batch_ids == (candidates[0].batch_id,)

    engine.tell(
        [
            EvaluationRecord(
                candidate_id=candidates[0].candidate_id,
                batch_id=candidates[0].batch_id,
                score=3.0,
                confidence="trusted_full",
                stage="full",
                cost=1.0,
            )
        ]
    )

    after = engine.state_summary()

    assert after.best_candidate_id == candidates[0].candidate_id
    assert after.best_score == pytest.approx(3.0)
    assert after.trusted_count == 1


def test_cma_best_state_ignores_partial_scores() -> None:
    engine = CMAESOptimizer(_space(), population_size=4, seed=7)
    candidates = engine.ask()

    engine.tell(
        [
            EvaluationRecord(
                candidate_id=candidates[0].candidate_id,
                batch_id=candidates[0].batch_id,
                score=999.0,
                confidence="partial",
                stage="cheap",
                cost=0.1,
            )
        ]
    )
    result = engine.tell(
        [
            EvaluationRecord(
                candidate_id=candidates[0].candidate_id,
                batch_id=candidates[0].batch_id,
                score=0.0,
                confidence="trusted_full",
                stage="full",
                cost=1.0,
            ),
            EvaluationRecord(
                candidate_id=candidates[1].candidate_id,
                batch_id=candidates[1].batch_id,
                score=10.0,
                confidence="trusted_full",
                stage="full",
                cost=1.0,
            ),
        ]
    )

    assert result.best_candidate_id == candidates[1].candidate_id
    assert result.best_score == pytest.approx(10.0)


def test_cma_cached_records_are_eligible_for_best_state_and_batch_update() -> None:
    engine = CMAESOptimizer(_space(), population_size=4, seed=7)
    candidates = engine.ask()

    result = engine.tell(
        [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=10.0 + index,
                confidence="cached",
                stage="full",
                cost=0.0,
            )
            for index, candidate in enumerate(candidates)
        ]
    )

    assert result.cached_count == 4
    assert result.consumed_batch_ids == (candidates[0].batch_id,)
    assert engine.generation == 1
    assert engine.vnext_telemetry.candidates_cached == 4
    assert engine.vnext_telemetry.candidates_full_evaluated == 0
    assert result.best_candidate_id == candidates[-1].candidate_id
    assert result.best_score == pytest.approx(13.0)


def test_cma_minimize_direction_tracks_lowest_trusted_score() -> None:
    engine = CMAESOptimizer(_space(), population_size=4, seed=7, direction="minimize")
    candidates = engine.ask()

    result = engine.tell(
        [
            EvaluationRecord(
                candidate_id=candidates[0].candidate_id,
                batch_id=candidates[0].batch_id,
                score=10.0,
                confidence="trusted_full",
                stage="full",
                cost=1.0,
            ),
            EvaluationRecord(
                candidate_id=candidates[1].candidate_id,
                batch_id=candidates[1].batch_id,
                score=2.0,
                confidence="trusted_full",
                stage="full",
                cost=1.0,
            ),
        ]
    )

    assert engine.best_candidate is not None
    assert engine.best_candidate.candidate_id == candidates[1].candidate_id
    assert result.best_score == pytest.approx(2.0)
