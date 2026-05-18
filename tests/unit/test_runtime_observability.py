import logging

import pytest

import evocore
from evocore import CMAESOptimizer, FitnessError, GeneSpace, GeneticAlgorithmOptimizer


def sphere(ind):
    return -sum(x * x for x in ind.values)


def non_finite_once(_ind):
    return float("nan")


def test_version_export_is_string():
    assert isinstance(evocore.__version__, str)
    assert evocore.__version__


def test_version_export_is_in_all():
    assert "__version__" in evocore.__all__


def test_ga_logs_generation_progress(caplog):
    engine = GeneticAlgorithmOptimizer(
        GeneSpace.uniform(-1.0, 1.0, 2), population_size=6, max_generations=1, seed=7
    )

    with caplog.at_level(logging.INFO, logger="evocore"):
        engine._run_from_population(engine._initial_population(), sphere, start_generation=0)

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "evocore.optimizers.ga.generation_loop"
    ]
    assert any("GA generation=0" in message for message in messages)


def test_ga_non_finite_fitness_raises_without_warning_log(caplog) -> None:
    engine = GeneticAlgorithmOptimizer(
        GeneSpace.uniform(-1.0, 1.0, 2), population_size=4, max_generations=1, seed=7
    )

    with (
        caplog.at_level(logging.WARNING, logger="evocore"),
        pytest.raises(FitnessError, match="finite"),
    ):
        engine._run_from_population(
            engine._initial_population(), non_finite_once, start_generation=0
        )

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "evocore.optimizers.ga.engine"
    ]
    assert not any("assigned fitness=-inf" in message for message in messages)


def test_cmaes_logs_generation_progress(caplog):
    engine = CMAESOptimizer(
        GeneSpace.uniform(-1.0, 1.0, 2), population_size=6, max_generations=1, seed=7
    )

    with caplog.at_level(logging.INFO, logger="evocore"):
        engine.run(sphere)

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "evocore.optimizers.cmaes.engine"
    ]
    assert any("CMA-ES generation=0" in message for message in messages)
