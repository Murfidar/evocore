from __future__ import annotations

import pytest

from evocore import _core
from evocore.optimizers.de.strategies import (
    SUPPORTED_DE_STRATEGIES,
    supported_strategy_names,
)

STRATEGY_CONTRACT = {
    "rand1bin": {
        "donor_count": 3,
        "difference_pair_count": 1,
        "uses_best_slot": False,
        "base_is_target": False,
        "adaptive": False,
    },
    "best1bin": {
        "donor_count": 3,
        "difference_pair_count": 1,
        "uses_best_slot": True,
        "base_is_target": False,
        "adaptive": False,
    },
    "rand2bin": {
        "donor_count": 5,
        "difference_pair_count": 2,
        "uses_best_slot": False,
        "base_is_target": False,
        "adaptive": False,
    },
    "current-to-best1bin": {
        "donor_count": 3,
        "difference_pair_count": 1,
        "uses_best_slot": True,
        "base_is_target": True,
        "adaptive": False,
    },
    "jde-rand1bin": {
        "donor_count": 3,
        "difference_pair_count": 1,
        "uses_best_slot": False,
        "base_is_target": False,
        "adaptive": True,
    },
}


def _population(size: int) -> list[list[float]]:
    return [[float(index), float(index + 1)] for index in range(size)]


def _scores(size: int) -> list[float]:
    return [float(index) for index in range(size)]


def _jde_state(size: int) -> dict[str, list[float]]:
    return {"f_by_slot": [0.5] * size, "cr_by_slot": [0.9] * size}


def _generate(strategy: str, population_size: int):
    jde_state = _jde_state(population_size) if strategy == "jde-rand1bin" else None
    return _core.de_generate_trials(
        _population(population_size),
        _scores(population_size),
        [(-10.0, 10.0), (-10.0, 10.0)],
        ["float", "float"],
        strategy,
        0.7,
        0.9,
        42,
        0,
        [0],
        "maximize",
        jde_state,
    )


def test_python_strategy_names_have_contract_entries() -> None:
    assert set(supported_strategy_names()) == set(STRATEGY_CONTRACT)


@pytest.mark.parametrize("strategy", supported_strategy_names())
def test_rust_accepts_every_python_strategy_at_min_population(strategy: str) -> None:
    spec = SUPPORTED_DE_STRATEGIES[strategy]

    proposals = _generate(strategy, spec.min_population_size)

    assert len(proposals) == 1
    metadata = proposals[0]["metadata"]
    expected = STRATEGY_CONTRACT[strategy]
    assert metadata["strategy"] == strategy
    assert len(metadata["donor_slots"]) == expected["donor_count"]
    assert len(metadata["difference_pairs"]) == expected["difference_pair_count"]
    assert ("best_slot" in metadata) is expected["uses_best_slot"]
    if expected["base_is_target"]:
        assert metadata["base_slot"] == metadata["target_slot"]
    assert ("adaptive_slot" in metadata) is expected["adaptive"]
    assert ("mutation_factor" in metadata) is expected["adaptive"]
    assert ("crossover_rate" in metadata) is expected["adaptive"]


@pytest.mark.parametrize("strategy", supported_strategy_names())
def test_rust_min_population_matches_python_spec(strategy: str) -> None:
    spec = SUPPORTED_DE_STRATEGIES[strategy]

    with pytest.raises(ValueError, match=f"at least {spec.min_population_size}"):
        _generate(strategy, spec.min_population_size - 1)


def test_rust_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="Unknown DE strategy"):
        _generate("not-a-strategy", 6)


def test_rust_requires_jde_state_for_adaptive_strategy() -> None:
    with pytest.raises(ValueError, match="jde_state is required"):
        _core.de_generate_trials(
            _population(4),
            _scores(4),
            [(-10.0, 10.0), (-10.0, 10.0)],
            ["float", "float"],
            "jde-rand1bin",
            0.7,
            0.9,
            42,
            0,
            [0],
            "maximize",
        )
