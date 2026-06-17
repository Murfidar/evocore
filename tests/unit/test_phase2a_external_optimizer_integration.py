import pytest

from evocore import (
    CandidateArchive,
    CMAESOptimizer,
    DifferentialEvolutionOptimizer,
    GeneSpace,
    GeneticAlgorithmOptimizer,
    WarmStartRecord,
)


@pytest.fixture(params=["ga", "de", "cmaes"])
def optimizer(request):
    space = GeneSpace.uniform(-5.0, 5.0, 2)
    if request.param == "ga":
        return GeneticAlgorithmOptimizer(space, population_size=4, seed=10)
    if request.param == "de":
        return DifferentialEvolutionOptimizer(space, population_size=4, seed=10)
    return CMAESOptimizer(space, population_size=4, seed=10)


def test_archive_accepts_phase1_snapshots_from_all_optimizers(optimizer) -> None:
    optimizer.warm_start(
        [
            WarmStartRecord(values=(1.0, 1.0), score=1.0, metadata={"family": "a"}),
            WarmStartRecord(values=(2.0, 2.0), score=2.0, metadata={"family": "b"}),
        ]
    )
    archive = CandidateArchive()

    archive.add_population(optimizer.candidate_snapshot(scope="trusted"), source="trusted")

    records = archive.to_warm_start_records(k=1)
    assert len(records) == 1
    assert records[0].score == 2.0
