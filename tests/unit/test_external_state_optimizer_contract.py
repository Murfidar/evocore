import pytest

from evocore import (
    CMAESOptimizer,
    DifferentialEvolutionOptimizer,
    ExternalStateOptimizer,
    Gene,
    GeneSpace,
    GeneticAlgorithmOptimizer,
    WarmStartRecord,
)


def _space() -> GeneSpace:
    return GeneSpace([Gene("x", "float", -5.0, 5.0), Gene("y", "float", -5.0, 5.0)])


def _warm_records() -> list[WarmStartRecord]:
    return [
        WarmStartRecord(params={"x": 1.0, "y": 1.0}, score=10.0, metadata={"source": "elite_a"}),
        WarmStartRecord(params={"x": 2.0, "y": 2.0}, score=20.0, metadata={"source": "elite_b"}),
    ]


@pytest.fixture(params=["ga", "de", "cmaes"])
def optimizer(request):
    space = _space()
    if request.param == "ga":
        return GeneticAlgorithmOptimizer(space, population_size=4, seed=11)
    if request.param == "de":
        return DifferentialEvolutionOptimizer(space, population_size=4, seed=11)
    return CMAESOptimizer(space, population_size=4, seed=11)


def test_all_phase_1_optimizers_support_external_state_protocol(optimizer) -> None:
    assert isinstance(optimizer, ExternalStateOptimizer)

    capabilities = optimizer.external_state_capabilities()
    assert capabilities.population_snapshots is True
    assert capabilities.top_candidate_snapshots is True
    assert capabilities.cached_record_helpers is True


def test_warm_start_top_candidates_preserve_metadata(optimizer) -> None:
    result = optimizer.warm_start(_warm_records())

    assert result.accepted_count == 2
    top = optimizer.top_candidates(1)

    assert len(top) == 1
    assert top[0].score == 20.0
    assert top[0].metadata["record_metadata"]["source"] == "elite_b"


def test_tracked_warm_start_is_visible_as_scored_not_trusted(optimizer) -> None:
    optimizer.warm_start(_warm_records(), mode="tracked")

    assert optimizer.candidate_snapshot(scope="trusted").candidates == ()
    assert len(optimizer.candidate_snapshot(scope="scored").candidates) == 2


def test_user_metadata_cannot_override_internal_tracked_mode(optimizer) -> None:
    optimizer.warm_start(
        [
            WarmStartRecord(
                params={"x": 1.0, "y": 1.0},
                score=10.0,
                metadata={"external_state_mode": "state", "source": "archive"},
            )
        ],
        mode="tracked",
    )

    scored = optimizer.candidate_snapshot(scope="scored").candidates
    assert len(scored) == 1
    assert scored[0].metadata["external_state_mode"] == "tracked"
    assert scored[0].metadata["source"] == "archive"
    assert optimizer.candidate_snapshot(scope="trusted").candidates == ()


def test_duplicate_warm_start_values_are_skipped_consistently(optimizer) -> None:
    records = _warm_records()
    result = optimizer.warm_start([records[0], records[0]])

    assert result.accepted_count == 1
