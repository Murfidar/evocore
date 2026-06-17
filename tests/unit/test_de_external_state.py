import pytest

from evocore import (
    DifferentialEvolutionOptimizer,
    EvaluationRecord,
    ExternalStateOptimizer,
    Gene,
    GeneSpace,
    WarmStartRecord,
)
from evocore.core import ConfigurationError


def _space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("y", "float", -5.0, 5.0),
        ]
    )


def _optimizer() -> DifferentialEvolutionOptimizer:
    return DifferentialEvolutionOptimizer(_space(), population_size=4, max_generations=3, seed=456)


def _records() -> list[WarmStartRecord]:
    return [
        WarmStartRecord(params={"x": 1.0, "y": 1.0}, score=10.0),
        WarmStartRecord(params={"x": 2.0, "y": 2.0}, score=20.0),
        WarmStartRecord(params={"x": -1.0, "y": -1.0}, score=5.0),
        WarmStartRecord(params={"x": 3.0, "y": 3.0}, score=15.0),
        WarmStartRecord(params={"x": 4.0, "y": 4.0}, score=30.0),
    ]


def test_de_implements_external_state_protocol() -> None:
    optimizer = _optimizer()

    assert isinstance(optimizer, ExternalStateOptimizer)
    capabilities = optimizer.external_state_capabilities()
    assert capabilities.warm_start_before_ask is True
    assert capabilities.warm_start_after_ask is False
    assert capabilities.proposed_candidate_injection is True
    assert capabilities.population_snapshots is True


def test_de_warm_start_state_fills_target_slots_by_rank() -> None:
    optimizer = _optimizer()

    result = optimizer.warm_start(_records())

    assert result.accepted_count == 4
    assert result.cached_count == 4
    assert result.state_accepted_count == 4
    assert optimizer.state_summary().trusted_count == 4
    assert [item.score for item in optimizer.top_candidates(4)] == [30.0, 20.0, 15.0, 10.0]

    batch = optimizer.ask(4)
    assert {candidate.origin for candidate in batch} == {"mutation"}


def test_de_state_warm_start_rejects_after_ask() -> None:
    optimizer = _optimizer()
    optimizer.ask(2)

    with pytest.raises(ConfigurationError, match="before DE target initialization"):
        optimizer.warm_start(_records())


def test_de_tracked_warm_start_after_initialization_does_not_change_targets() -> None:
    optimizer = _optimizer()
    first_batch = optimizer.ask(4)
    optimizer.tell(
        [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=float(index),
                confidence="trusted_full",
                stage="full",
            )
            for index, candidate in enumerate(first_batch)
        ]
    )

    result = optimizer.warm_start(_records(), mode="tracked")

    assert result.state_accepted_count == 0
    assert optimizer.state_summary().trusted_count == 4
    assert len(optimizer.candidate_snapshot(scope="scored").candidates) == 9


def test_de_inject_candidates_before_initialization_creates_initial_pending_batch() -> None:
    optimizer = _optimizer()

    injected = optimizer.inject_candidates(
        [WarmStartRecord(params={"x": 0.25, "y": 0.5}, score=0.0)],
        metadata={"candidate_source": "domain_seed"},
    )

    assert len(injected.accepted) == 1
    accepted = injected.accepted[0]
    assert optimizer.state_summary().pending_batch_ids == (accepted.batch_id,)

    result = optimizer.tell(
        [
            EvaluationRecord(
                candidate_id=accepted.candidate_id,
                batch_id=accepted.batch_id,
                score=8.0,
                confidence="cached",
                stage="search_memory",
            )
        ]
    )

    assert result.state_accepted_count == 1
    assert optimizer.state_summary().trusted_count == 1


def test_de_proposed_injection_rejects_after_target_population_is_full() -> None:
    optimizer = _optimizer()
    optimizer.warm_start(_records())

    with pytest.raises(ConfigurationError, match="target population is full"):
        optimizer.inject_candidates(
            [WarmStartRecord(params={"x": 0.0, "y": 0.0}, score=0.0)],
            mode="proposed",
        )


def test_de_proposed_injection_rejects_when_batch_would_overfill_targets() -> None:
    optimizer = _optimizer()
    initial = optimizer.ask(2)
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

    with pytest.raises(ConfigurationError, match="remaining target slot"):
        optimizer.inject_candidates(_records()[:3], mode="proposed")


def test_de_external_state_checkpoint_round_trip(tmp_path) -> None:
    source = _optimizer()
    source.warm_start(_records())
    checkpoint_path = tmp_path / "de-external-state.evocore-checkpoint.json"
    source.save_checkpoint(checkpoint_path, source.ask_tell_checkpoint(metadata={"phase": "warm"}))

    restored = _optimizer()
    restored.resume_ask_tell_checkpoint(checkpoint_path)

    assert [item.score for item in restored.top_candidates(2)] == [30.0, 20.0]
    assert restored.state_summary().trusted_count == 4
