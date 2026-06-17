from evocore import (
    EvaluationRecord,
    ExternalStateOptimizer,
    Gene,
    GeneSpace,
    GeneticAlgorithmOptimizer,
    WarmStartRecord,
)


def _space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("n", "int", 1, 10),
            Gene("enabled", "bool"),
        ]
    )


def _optimizer() -> GeneticAlgorithmOptimizer:
    return GeneticAlgorithmOptimizer(_space(), population_size=4, max_generations=3, seed=123)


def _records() -> list[WarmStartRecord]:
    return [
        WarmStartRecord(params={"x": 1.0, "n": 3, "enabled": True}, score=10.0),
        WarmStartRecord(params={"x": 2.0, "n": 4, "enabled": False}, score=20.0),
        WarmStartRecord(params={"x": -1.0, "n": 5, "enabled": True}, score=5.0),
    ]


def test_ga_implements_external_state_protocol() -> None:
    optimizer = _optimizer()

    assert isinstance(optimizer, ExternalStateOptimizer)
    capabilities = optimizer.external_state_capabilities()
    assert capabilities.warm_start_before_ask is True
    assert capabilities.warm_start_after_ask is True
    assert capabilities.proposed_candidate_injection is True
    assert capabilities.population_snapshots is True


def test_ga_warm_start_state_populates_trusted_snapshot_and_next_ask() -> None:
    optimizer = _optimizer()

    result = optimizer.warm_start(_records())

    assert result.accepted_count == 3
    assert result.cached_count == 3
    assert result.state_accepted_count == 3
    assert result.best_score == 20.0

    trusted = optimizer.candidate_snapshot(scope="trusted")
    assert trusted.trusted_count == 3
    assert [item.score for item in optimizer.top_candidates(2)] == [20.0, 10.0]

    next_batch = optimizer.ask(4)
    assert {candidate.origin for candidate in next_batch} == {"mutation"}


def test_ga_warm_start_tracked_does_not_seed_reproduction_state() -> None:
    optimizer = _optimizer()

    result = optimizer.warm_start(_records(), mode="tracked")

    assert result.accepted_count == 3
    assert result.state_accepted_count == 0
    assert optimizer.state_summary().trusted_count == 0
    assert len(optimizer.candidate_snapshot(scope="scored").candidates) == 3
    assert optimizer.candidate_snapshot(scope="trusted").candidates == ()

    batch = optimizer.ask(4)
    assert {candidate.origin for candidate in batch} == {"random"}


def test_ga_warm_start_skips_duplicate_values_by_default() -> None:
    optimizer = _optimizer()
    duplicate = WarmStartRecord(params={"x": 1.0, "n": 3, "enabled": True}, score=12.0)

    result = optimizer.warm_start([_records()[0], duplicate])
    snapshot = optimizer.candidate_snapshot(scope="trusted")

    assert result.accepted_count == 1
    assert snapshot.trusted_count == 1
    assert len(snapshot.candidates) == 1


def test_ga_inject_candidates_creates_pending_batch_for_later_tell() -> None:
    optimizer = _optimizer()

    injected = optimizer.inject_candidates(
        [WarmStartRecord(params={"x": 0.5, "n": 2, "enabled": True}, score=0.0)],
        metadata={"candidate_source": "domain_seed"},
    )

    assert len(injected.accepted) == 1
    candidate_id = injected.accepted[0].candidate_id
    assert optimizer.state_summary().pending_batch_ids == (injected.accepted[0].batch_id,)

    result = optimizer.tell(
        [
            EvaluationRecord(
                candidate_id=candidate_id,
                batch_id=injected.accepted[0].batch_id,
                score=7.0,
                confidence="trusted_full",
                stage="full",
            )
        ]
    )

    assert result.state_accepted_count == 1
    assert optimizer.top_candidates(1)[0].candidate_id == candidate_id


def test_ga_external_state_checkpoint_round_trip(tmp_path) -> None:
    source = _optimizer()
    source.warm_start(_records())
    checkpoint_path = tmp_path / "ga-external-state.evocore-checkpoint.json"
    source.save_checkpoint(checkpoint_path, source.ask_tell_checkpoint(metadata={"phase": "warm"}))

    restored = _optimizer()
    restored.resume_ask_tell_checkpoint(checkpoint_path)

    assert [item.values for item in restored.top_candidates(2)] == [
        (2.0, 4, False),
        (1.0, 3, True),
    ]
    assert restored.state_summary().trusted_count == 3
