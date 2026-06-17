import pytest

from evocore import CMAESOptimizer, ExternalStateOptimizer, Gene, GeneSpace, WarmStartRecord
from evocore.core import ConfigurationError
from evocore.search_space import encode_gene_values


def _space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("y", "float", -5.0, 5.0),
        ]
    )


def _optimizer() -> CMAESOptimizer:
    return CMAESOptimizer(_space(), population_size=4, max_generations=3, seed=789)


def _records() -> list[WarmStartRecord]:
    return [
        WarmStartRecord(params={"x": 1.0, "y": 1.0}, score=10.0),
        WarmStartRecord(params={"x": 2.0, "y": 2.0}, score=20.0),
        WarmStartRecord(params={"x": -1.0, "y": -1.0}, score=5.0),
    ]


def test_cmaes_implements_external_state_protocol() -> None:
    optimizer = _optimizer()

    assert isinstance(optimizer, ExternalStateOptimizer)
    capabilities = optimizer.external_state_capabilities()
    assert capabilities.warm_start_before_ask is True
    assert capabilities.warm_start_after_ask is False
    assert capabilities.proposed_candidate_injection is False
    assert capabilities.tracked_only_injection is True


def test_cmaes_warm_start_state_sets_initial_mean_from_best_record() -> None:
    optimizer = _optimizer()

    result = optimizer.warm_start(_records(), cma_mean_strategy="best")

    assert result.accepted_count == 3
    assert result.state_accepted_count == 3
    assert result.best_score == 20.0
    assert optimizer.initial_mean == encode_gene_values(_space(), [2.0, 2.0])
    assert [item.score for item in optimizer.top_candidates(2)] == [20.0, 10.0]


def test_cmaes_warm_start_state_sets_initial_mean_from_top_k_centroid() -> None:
    optimizer = _optimizer()

    optimizer.warm_start(_records(), cma_mean_strategy="top_k_centroid", top_k=2)

    assert optimizer.initial_mean == encode_gene_values(_space(), [1.5, 1.5])


def test_cmaes_state_warm_start_rejects_after_state_exists() -> None:
    optimizer = _optimizer()
    optimizer.ask()

    with pytest.raises(ConfigurationError, match="before the first CMA-ES ask"):
        optimizer.warm_start(_records())


def test_cmaes_tracked_warm_start_after_start_records_scores_only() -> None:
    optimizer = _optimizer()
    optimizer.ask()

    result = optimizer.warm_start(_records(), mode="tracked")

    assert result.state_accepted_count == 0
    assert optimizer.initial_mean is None
    assert len(optimizer.candidate_snapshot(scope="known").candidates) == 7
    assert len(optimizer.candidate_snapshot(scope="scored").candidates) == 3


def test_cmaes_injection_supports_tracked_only() -> None:
    optimizer = _optimizer()

    with pytest.raises(ConfigurationError, match="tracked"):
        optimizer.inject_candidates(_records(), mode="proposed")

    result = optimizer.inject_candidates(_records(), mode="tracked")

    assert len(result.accepted) == 3
    assert optimizer.state_summary().pending_batch_ids == ()


def test_cmaes_external_state_checkpoint_round_trip(tmp_path) -> None:
    source = _optimizer()
    source.warm_start(_records(), cma_mean_strategy="best")
    checkpoint_path = tmp_path / "cmaes-external-state.evocore-checkpoint.json"
    source.save_checkpoint(checkpoint_path, source.ask_tell_checkpoint(metadata={"phase": "warm"}))

    restored = CMAESOptimizer(
        _space(),
        population_size=4,
        max_generations=3,
        seed=789,
        initial_mean=encode_gene_values(_space(), [2.0, 2.0]),
    )
    restored.resume_ask_tell_checkpoint(checkpoint_path)

    assert [item.score for item in restored.top_candidates(2)] == [20.0, 10.0]
    assert restored.state_summary().trusted_count == 3
